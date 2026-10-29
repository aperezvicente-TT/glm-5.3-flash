"""Pure-numpy reader for single-buffer host `.tensorbin` files written by ttnn.dump_tensor.

File = [u64 header_size][flatbuffer header (header_size bytes)][raw buffer]. For TILE layout the raw buffer is
32x32 tiles in row-major tile order, each tile = 4 faces of 16x16 (f0 r0-15/c0-15, f1 r0-15/c16-31,
f2 r16-31/c0-15, f3 r16-31/c16-31), row-major inside a face. bfp8_b / bfp4_b tiles are
[64 shared exponents (one per face row of 16 values, face-major)] + mantissas in the same element order:
bfp8_b one byte (sign<<7 | 7-bit magnitude, value = m * 2^(e-127-6)), bfp4_b one nibble per value
(low nibble first; sign<<3 | 3-bit magnitude, value = m * 2^(e-127-2)). Matches
tt_metal/impl/data_format/{bfloat8,bfloat4,blockfloat_common}.cpp unpackers.

No ttnn import: decoding a bfp tensor through ttnn.to_torch instantiates MetalContext (opens the UMD driver),
which CPU-only stages must not do.
"""

from __future__ import annotations

import math
import struct

import numpy as np

TILE = 32
FACE = 16
TILE_BYTES = {"bfloat8_b": 64 + 1024, "bfloat4_b": 64 + 512, "bfloat16": 2048, "float32": 4096}


def raw_data(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        (hdr,) = struct.unpack("<Q", f.read(8))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return np.asarray(mm[8 + hdr :])


def _dims(shape):
    shape = list(shape)
    if len(shape) == 1:
        return 1, 1, shape[0]
    return int(np.prod(shape[:-2])) if len(shape) > 2 else 1, shape[-2], shape[-1]


def _tiles_to_matrix(tiles: np.ndarray, batch: int, hp: int, wp: int) -> np.ndarray:
    """tiles [ntiles, 1024] (element order: face-major, row-major inside face) -> [batch, hp, wp]."""
    ht, wt = hp // TILE, wp // TILE
    t = tiles.reshape(batch, ht, wt, 2, 2, FACE, FACE)  # face row, face col, r, c
    t = t.transpose(0, 1, 3, 5, 2, 4, 6)  # batch, ht, fr, r, wt, fc, c
    return t.reshape(batch, hp, wp)


def _decode_bfp(data: np.ndarray, ntiles: int, bits: int) -> np.ndarray:
    tb = TILE_BYTES["bfloat8_b" if bits == 8 else "bfloat4_b"]
    d = data[: ntiles * tb].reshape(ntiles, tb)
    exps = d[:, :64].astype(np.int32)  # [ntiles, 64] one per 16 values
    man = d[:, 64:]
    if bits == 8:
        sign = (man >> 7).astype(np.int32)
        mag = (man & 0x7F).astype(np.float32)
        shift = 6
    else:
        lo = man & 0x0F
        hi = man >> 4
        nib = np.stack([lo, hi], axis=-1).reshape(ntiles, 1024)
        sign = (nib >> 3).astype(np.int32)
        mag = (nib & 0x7).astype(np.float32)
        shift = 2
    e = np.repeat(exps, 16, axis=1)  # [ntiles, 1024]
    val = np.ldexp(mag, e - 127 - shift)
    val = np.where(sign == 1, -val, val)
    val[mag == 0] = 0.0
    return val.astype(np.float32)


def decode(path: str, shape, dtype: str, layout: str = "TILE") -> np.ndarray:
    """Return the logical tensor as float32 numpy of `shape`."""
    data = raw_data(path)
    shape = list(shape)
    if layout == "ROW_MAJOR":
        n = int(np.prod(shape))
        if dtype == "bfloat16":
            u16 = data[: 2 * n].view(np.uint16).astype(np.uint32) << 16
            return u16.view(np.float32).reshape(shape)
        if dtype == "float32":
            return data[: 4 * n].view(np.float32).reshape(shape).copy()
        raise ValueError(dtype)
    batch, h, w = _dims(shape)
    hp, wp = math.ceil(h / TILE) * TILE, math.ceil(w / TILE) * TILE
    ntiles = batch * (hp // TILE) * (wp // TILE)
    expect = ntiles * TILE_BYTES[dtype]
    if data.size != expect:
        raise ValueError(f"{path}: data region {data.size} B != expected {expect} B for {shape} {dtype}")
    if dtype == "bfloat16":
        vals = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(ntiles, 1024)
    elif dtype == "float32":
        vals = data.view(np.float32).reshape(ntiles, 1024)
    elif dtype == "bfloat8_b":
        vals = _decode_bfp(data, ntiles, 8)
    elif dtype == "bfloat4_b":
        vals = _decode_bfp(data, ntiles, 4)
    else:
        raise ValueError(dtype)
    m = _tiles_to_matrix(vals, batch, hp, wp)[:, :h, :w]
    return np.ascontiguousarray(m).reshape(shape)


def pcc(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float((a * b).sum() / den)
