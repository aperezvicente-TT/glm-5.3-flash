"""Does a plugin worktree carry the plugin fixes the README lists as found and fixed? Each fix counts as present when
its commit is an ancestor of the worktree's HEAD, or a patch-equivalent commit is (git cherry: a pick under another
sha), or a named re-make is an ancestor. A fix with a content PROBE must also still be in HEAD's file (ancestry alone
would not notice a later revert: review round 14): "REVERTED" otherwise. Run at install (install_on_pass.sh) and at
launch (serve_window.sh, against the target's own GLM53_PLUGIN_SRC). The 1M / PMK lineage branched before the n refusal e79ee93 and served without it (review round 12): this check
names such a gap before an install. rc 1 when a required fix is missing.
A HEAD whose full sha is listed in plugin_fixes_verified.txt (beside this script) with the current list's fingerprint
passed this whole check on a host holding every fix object (program 2026-10-27T01:27:19Z: the Galaxy's bundle-built
clone holds only the served sha's ancestry, so a fix picked or re-made under another sha read UNKNOWN there). The result
depends only on HEAD's commit graph and the list, so it stands for that sha anywhere. The probes still run on HEAD.
An absent fix object with a content probe present in HEAD counts too ("by probe").
usage: plugin_fixes_check.py [--record] <plugin worktree> [<plugin worktree> ...]  (--record: append a passing HEAD)"""

import hashlib
import os
import subprocess
import sys
import time

# (sha, what[, the same fix re-made under other shas on another branch -- a different diff context that git cherry does
# not match; each named in its own commit message])
FIXES = [  # README "Defects found and fixed" + the review-round plugin fixes
    ("199510c", "LMCache page reads compiled after the trace capture (SlotOps page copies)"),
    ("e3e7072", "an unaligned prefill chunk start"),
    ("267ca90", "a preempted request's recompute off the grid"),
    ("86c1e5e", "the bounded KDA checkpoint store"),
    ("e79ee93", "n > max_num_seqs refused with a 400"),
    ("12928b3", "the partial-page h2d fix", ["d169b12"]),
    # review round 13 P2-1 step 3: the rest of the README list and the review-round plugin fixes, with the 1M / PMK
    # lineage's re-makes (same subject, another base) as alternates
    ("0da3370", "KDA checkpoints: the async writer + GLM53_KDA_CKPT_EVERY"),
    ("97df19b", "GLM53_KDA_CKPT_EVERY bounds the decode-time KDA checkpoints too", ["1de0211"]),
    ("86fbc6e", "GLM53_LMCACHE_MIN_STORE_TOKENS: short prompts neither stored nor checkpointed", ["66be963"]),
    ("d038ad7", "the prompt's resume point is the last grid boundary strictly before its end", ["859916d"]),
    ("736c6fb", "the provenance rule: no KDA checkpoint from a request recomputing LMCache's pages", ["75896ca"]),
    ("385d6b9", "LMCache's load window = the hit vLLM computes from", ["2bd7edf"]),
    ("5ea87ac", "a latched device-step fault closes the mesh at the latch (drain)", ["c485aad"]),
    # c026ad1 and 2b5afac are ONE fix (same subject and author time; review round 16 O5): one entry, 2b5afac its re-make
    ("c026ad1", "a structured /v1/completions request under --reasoning-parser is constrained from its first token",
     ["2b5afac"]),
    ("1306ec1", "the dev prefill hints act only on the alignment grid", ["6ffd8be", "dd5c54e"]),
]


# sha -> (file in the plugin worktree, a fixed string HEAD's version must contain)
PROBES = {  # 11 of the 15 fixes (rounds 15-16): an identifier the fix added, present in every carrying plugin's HEAD; e3e7072,
    # 267ca90, 12928b3 and 1306ec1 added no distinctive identifier and are checked by history only
    "199510c": ("src/vllm_tt_plugin/kv_connector.py", "def _slot_ops"),
    "86c1e5e": ("src/vllm_tt_plugin/kv_connector.py", "GLM53_KDA_CKPT_MAX_GB"),
    "e79ee93": ("src/vllm_tt_plugin/platform.py", "def _reject_parallel_samples_over_batch"),
    "0da3370": ("src/vllm_tt_plugin/kv_connector.py", "def _checkpoint_grid"),
    "97df19b": ("src/vllm_tt_plugin/kv_connector.py", "def decode_boundary_wanted"),
    "86fbc6e": ("src/vllm_tt_plugin/kv_connector.py", "def lmcache_min_store_tokens"),
    "d038ad7": ("src/vllm_tt_plugin/kv_connector.py", "STRICTLY before the prompt"),
    "736c6fb": ("src/vllm_tt_plugin/kv_connector.py", "def _mark_provenance"),
    "385d6b9": ("src/vllm_tt_plugin/kv_connector.py", "GLM53_LMCACHE_LOAD_ALIGNED"),
    "5ea87ac": ("src/vllm_tt_plugin/drain.py", "def mesh_close_marker"),
    "c026ad1": ("src/vllm_tt_plugin/scheduler.py", "def _raw_prompt_reasoning_state"),
}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


VERIFIED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugin_fixes_verified.txt")
LIST_FP = hashlib.sha256(repr((FIXES, sorted(PROBES.items()))).encode()).hexdigest()[:16]


def verified_heads() -> dict:
    """full sha -> the line that verified it, for lines carrying the current list's fingerprint"""
    out = {}
    try:
        for line in open(VERIFIED):
            f = line.split()
            if len(f) >= 3 and not line.startswith("#") and f[1] == LIST_FP:
                out[f[0]] = line.strip()
    except OSError:
        pass
    return out


record = "--record" in sys.argv[1:]
rc = 0
for repo in [a for a in sys.argv[1:] if a != "--record"]:
    head = git(repo, "rev-parse", "--short=7", "HEAD").stdout.strip()
    full = git(repo, "rev-parse", "HEAD").stdout.strip()
    print(f"== {repo} @ {head}")
    ver = verified_heads().get(full)
    repo_rc = 0
    for sha, what, *alt in FIXES:
        if git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            probe = PROBES.get(sha)
            if ver is not None and (not probe or probe[1] in git(repo, "show", f"HEAD:{probe[0]}").stdout):
                print(f"  present {sha} (verified for this sha: {ver.split(None, 2)[2]}) ({what})")
                continue
            if probe and probe[1] in git(repo, "show", f"HEAD:{probe[0]}").stdout:
                print(f"  present {sha} by probe ({probe[1]!r} in HEAD's {probe[0]}; the fix object is not here) ({what})")
                continue
            print(f"  UNKNOWN {sha} ({what}): not in this repository")
            rc = repo_rc = 1
            continue
        if git(repo, "merge-base", "--is-ancestor", sha, "HEAD").returncode == 0:
            how = ""
        else:
            out = git(repo, "cherry", "HEAD", sha, f"{sha}^").stdout.split()
            same = [a for a in (alt[0] if alt else []) if git(repo, "merge-base", "--is-ancestor", a, "HEAD").returncode == 0]
            if out[:1] == ["-"]:
                how = " as an equivalent pick"
            elif same:
                how = f" as {same[0]} (the same fix re-made)"
            else:
                print(f"  MISSING {sha} ({what})")
                rc = repo_rc = 1
                continue
        probe = PROBES.get(sha)
        if probe and probe[1] not in git(repo, "show", f"HEAD:{probe[0]}").stdout:
            print(f"  REVERTED {sha} ({what}): in the history{how}, but HEAD's {probe[0]} lacks {probe[1]!r}")
            rc = repo_rc = 1
            continue
        print(f"  present {sha}{how} ({what})")
    if record and repo_rc == 0 and full not in verified_heads():
        with open(VERIFIED, "a") as f:
            f.write(f"{full} {LIST_FP} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {os.uname().nodename} "
                    f"{os.path.basename(os.path.normpath(repo))} every listed fix present\n")
        print(f"  recorded {head} in {os.path.basename(VERIFIED)}")
sys.exit(rc)
