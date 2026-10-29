"""GLM-5.3-Flash weight-cache spec: which HF tensors become which cache tensors, with what dtype/transform.

Shared by convert_weights.py (writer), spot_check.py (verifier) and anyone who needs the name mapping.
Pure python + torch; no ttnn import here.

Conventions (milestone 1, see .agents/skills/glm53-weights/SKILL.md):
- Cache key = HF name without the `model.language_model.` prefix (`lm_head.weight` unchanged).
- Every 2-D linear weight is stored transposed to [in, out] (so `y = x @ W`), TILE_LAYOUT.
- Routed experts: `layers.L.mlp.experts.E.gate_up_proj` = [4096, 2*2048] (cols 0:2048 gate, 2048:4096 up),
  `layers.L.mlp.experts.E.down_proj` = [2048, 4096]; both bfloat4_b.
- kv_b_proj [64*(256+256), 512] is split into W_uk [64, 256, 512] (q_nope @ W_uk[h] -> latent) and
  W_uv [64, 512, 256] (latent_out @ W_uv[h] -> v), bf16.
- Depthwise conv taps [C, 1, K] -> [K, C] (row j multiplies input row t-(K-1)+j), bf16.
- 1-D tensors keep their HF shape (TILE pads to [32, N] internally); f32 ones stay float32.
- Embedding is the one ROW_MAJOR tensor: ttnn.embedding asserts ROW_MAJOR weights
  (ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:34).
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field

PREFIX = "model.language_model."
NUM_LAYERS = 45  # + MTP layer 45 (converted too; out of scope for milestone 1 but cheap)
MTP_LAYER = 45

BFP4 = "bfloat4_b"
BFP8 = "bfloat8_b"
BF16 = "bfloat16"
FP32 = "float32"


@dataclass
class Task:
    name: str  # cache key
    path: str  # relative to cache dir
    dtype: str  # bfloat4_b | bfloat8_b | bfloat16 | float32
    layout: str  # TILE | ROW_MAJOR
    transform: str  # see apply_transform in convert_weights.py
    sources: list[str]  # HF tensor names (weights only; *_scale_inv companions implied for fp8)
    group: str  # reporting group: routed_expert | linear | norm | ...
    layer: int | None = None
    shape: list[int] = field(default_factory=list)  # logical cache shape


def read_headers(snapshot: str) -> dict[str, tuple[str, list[int], str]]:
    """HF tensor name -> (safetensors dtype, shape, shard file). Reads only the JSON headers."""
    idx = json.load(open(f"{snapshot}/model.safetensors.index.json"))["weight_map"]
    info = {}
    for shard in sorted(set(idx.values())):
        with open(f"{snapshot}/{shard}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                info[k] = (v["dtype"], v["shape"], shard)
    return info


def _strip(hf: str) -> str:
    return hf[len(PREFIX) :] if hf.startswith(PREFIX) else hf


def _layer_path(layer: int, leaf: str) -> str:
    return f"layers/{layer:02d}/{leaf}.tensorbin"


# Linear-weight leaf names (relative to layers.L.) and their cache dtype.
_LINEAR_DTYPE = {
    # KDA (bf16 in checkpoint)
    "self_attn.q_proj.weight": BFP8,
    "self_attn.k_proj.weight": BFP8,
    "self_attn.v_proj.weight": BFP8,
    "self_attn.o_proj.weight": BFP8,  # KDA o_proj bf16 [4096,8192] and MLA o_proj fp8 [4096,16384]
    "self_attn.f_a_proj.weight": BF16,
    "self_attn.f_b_proj.weight": BF16,
    "self_attn.g_a_proj.weight": BF16,
    "self_attn.g_b_proj.weight": BF16,
    "self_attn.b_proj.weight": BF16,
    # MLA (fp8)
    "self_attn.q_a_proj.weight": BFP8,
    "self_attn.q_b_proj.weight": BFP8,
    "self_attn.kv_a_proj_with_mqa.weight": BFP8,
    # DSA indexer (bf16)
    "self_attn.indexer.wq_b.weight": BF16,
    "self_attn.indexer.wk.weight": BF16,
    "self_attn.indexer.weights_proj.weight": BF16,
    "self_attn.indexer.index_kpool_compress_gate": BF16,  # used as F.linear(x, gate): [128,4096] -> [4096,128]
    # dense MLP (layers 0-2) and shared expert (fp8)
    "mlp.gate_proj.weight": BFP8,
    "mlp.up_proj.weight": BFP8,
    "mlp.down_proj.weight": BFP8,
    "mlp.shared_experts.gate_proj.weight": BFP8,
    "mlp.shared_experts.up_proj.weight": BFP8,
    "mlp.shared_experts.down_proj.weight": BFP8,
    # router (bf16, computed in fp32 by HF)
    "mlp.gate.weight": BF16,
    # mHC projections (bf16) [24, 16384] -> [16384, 24]
    "hc_attn_fn": BF16,
    "hc_ffn_fn": BF16,
    # MTP
    "eh_proj.weight": BFP8,
}

_VECTOR_DTYPE = {
    "input_layernorm.weight": BF16,
    "post_attention_layernorm.weight": BF16,
    "self_attn.o_norm.weight": BF16,
    "self_attn.q_a_layernorm.weight": BF16,
    "self_attn.kv_a_layernorm.weight": BF16,
    "self_attn.indexer.k_norm.weight": BF16,
    "self_attn.indexer.k_norm.bias": BF16,
    "self_attn.A_log": FP32,
    "self_attn.dt_bias": FP32,
    "mlp.gate.e_score_correction_bias": FP32,
    "hc_attn_base": FP32,
    "hc_attn_scale": FP32,
    "hc_ffn_base": FP32,
    "hc_ffn_scale": FP32,
    "enorm.weight": BF16,
    "hnorm.weight": BF16,
    "shared_head.norm.weight": BF16,
}

_GROUP = {
    BFP4: "bfp4",
    BFP8: "bfp8",
    BF16: "bf16",
    FP32: "fp32",
}


def build_tasks(headers: dict) -> list[Task]:
    tasks: list[Task] = []
    seen_sources: set[str] = set()
    layer_re = re.compile(r"^layers\.(\d+)\.(.+)$")
    expert_re = re.compile(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

    def shape_of(hf):
        return list(headers[hf][1])

    names = sorted(k for k in headers if not k.startswith("model.visual") and not k.endswith("_scale_inv"))
    for hf in names:
        key = _strip(hf)
        if hf == "lm_head.weight":
            r, c = shape_of(hf)
            tasks.append(Task(key, "lm_head.weight.tensorbin", BFP8, "TILE", "linear_t", [hf], "lm_head", None, [c, r]))
            continue
        if key == "embed_tokens.weight":
            tasks.append(
                Task(key, "embed_tokens.weight.tensorbin", BF16, "ROW_MAJOR", "copy", [hf], "embedding", None, shape_of(hf))
            )
            continue
        if key == "norm.weight":
            tasks.append(Task(key, "norm.weight.tensorbin", BF16, "TILE", "copy", [hf], "norm", None, shape_of(hf)))
            continue
        m = layer_re.match(key)
        if not m:
            raise KeyError(f"unhandled tensor {hf}")
        layer, leaf = int(m.group(1)), m.group(2)
        em = expert_re.match(leaf)
        if em:
            eid, which = int(em.group(1)), em.group(2)
            if which == "up":
                continue  # fused into gate_up below
            if which == "gate":
                g = hf
                u = hf.replace(".gate_proj.", ".up_proj.")
                gr, gc = shape_of(g)
                tasks.append(
                    Task(
                        f"layers.{layer}.mlp.experts.{eid}.gate_up_proj",
                        _layer_path(layer, f"experts/{eid:03d}.gate_up_proj"),
                        BFP4,
                        "TILE",
                        "experts_gate_up",
                        [g, u],
                        "routed_expert",
                        layer,
                        [gc, 2 * gr],
                    )
                )
            else:
                r, c = shape_of(hf)
                tasks.append(
                    Task(
                        f"layers.{layer}.mlp.experts.{eid}.down_proj",
                        _layer_path(layer, f"experts/{eid:03d}.down_proj"),
                        BFP4,
                        "TILE",
                        "linear_t",
                        [hf],
                        "routed_expert",
                        layer,
                        [c, r],
                    )
                )
            continue
        if leaf == "self_attn.kv_b_proj.weight":
            # [H*(nope+v), latent] -> per head [nope+v, latent]
            tasks.append(
                Task(f"layers.{layer}.self_attn.kv_b_proj.W_uk", _layer_path(layer, "self_attn.kv_b_proj.W_uk"),
                     BF16, "TILE", "kv_b_uk", [hf], "mla_kv_b", layer, [64, 256, 512])
            )
            tasks.append(
                Task(f"layers.{layer}.self_attn.kv_b_proj.W_uv", _layer_path(layer, "self_attn.kv_b_proj.W_uv"),
                     BF16, "TILE", "kv_b_uv", [hf], "mla_kv_b", layer, [64, 512, 256])
            )
            continue
        if leaf in ("self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"):
            c, _, k = shape_of(hf)
            tasks.append(
                Task(key, _layer_path(layer, leaf), BF16, "TILE", "conv_taps", [hf], "kda_conv", layer, [k, c])
            )
            continue
        if leaf == "self_attn.indexer.index_kpool_compress_ape":
            tasks.append(Task(key, _layer_path(layer, leaf), BF16, "TILE", "copy", [hf], "indexer", layer, shape_of(hf)))
            continue
        if leaf in _LINEAR_DTYPE:
            r, c = shape_of(hf)
            tasks.append(
                Task(key, _layer_path(layer, leaf), _LINEAR_DTYPE[leaf], "TILE", "linear_t", [hf], "linear", layer, [c, r])
            )
            continue
        if leaf in _VECTOR_DTYPE:
            tasks.append(
                Task(key, _layer_path(layer, leaf), _VECTOR_DTYPE[leaf], "TILE", "copy", [hf], "vector", layer, shape_of(hf))
            )
            continue
        raise KeyError(f"unhandled tensor {hf} (leaf {leaf})")

    # every non-visual weight must be consumed exactly once
    for t in tasks:
        seen_sources.update(t.sources)
    missing = [n for n in names if n not in seen_sources]
    if missing:
        raise RuntimeError(f"{len(missing)} HF tensors not mapped, e.g. {missing[:5]}")
    return tasks


def dtype_group(dtype: str) -> str:
    return _GROUP[dtype]
