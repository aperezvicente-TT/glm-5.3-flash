#!/usr/bin/env bash
# ONE candidate model tree for a Galaxy bisect arm (program 2026-10-28T03:53:06Z arm g; any later --knob-tree arm), HOST ONLY,
# no device: glx_k4d7lg_ef_prep.sh's recipe for a single tree -- a sparse models/ worktree of the base checkout's repo at <sha>,
# from <bundle> (its prerequisites must already be in that repo: e.g. 14's glxk4d7g_armg.bundle needs bbba4e657c2 = the e / f
# prep's tt-metal-s9cfk4d7f), the base's build links, ttnn/ + tools/ farms with the JIT kernel sources HEAD carries over the base
# and 14b's dump_fast_dispatch_halted.py. Outside models/ the sha may differ from 8f28c0fd0a8 only in k4d7lg's JIT kernel sources
# and in 14b's cq_prefetch.cpp drain (3fc9517c3d0; program [session 10] 2026-10-29T00:24:57Z: composed into every served form):
# that file must be the drain blob; tt_metal/ is then a farm of the base's (that one file real), else a link to the base's.
# A tree that exists at <sha> is kept. Run OUTSIDE the namespace with the real paths (GLX_ROOT, default /opt/glm53).
# usage: glx_k4d7lg_tree_prep.sh <bundle> <branch in it> <sha11> <tree name under glm53/> <dir with dump_fast_dispatch_halted.py>
#   e.g. glx_k4d7lg_tree_prep.sh .../glxk4d7g_armg.bundle glm53/glm53-14-armg 3ca0f5dd41d tt-metal-s9cfk4d7g /opt/glm53/glm53/glxk4d7lg
set -euo pipefail
BUN=$(readlink -f "${1:?the bundle}"); TBR=${2:?the branch}; TSHA=${3:?the sha11}; TNAME=${4:?the tree name}; PKG=$(readlink -f "${5:?the dir with dump_fast_dispatch_halted.py}")
G=${GLX_ROOT:-/opt/glm53}; L=$G/glm53
GIT="git -c core.hooksPath=/dev/null"
DRAIN_BLOB=58c735376771d43db9306c31b23572427fbfadfe  # tt_metal/impl/dispatch/kernels/cq_prefetch.cpp at 14b's 3fc9517c3d0 (md5 40e8ebb3)
say() { echo "$(date -u +%FT%TZ) glxk4d7lg tree prep ($TNAME): $*"; }
t0=$(date +%s)
for d in $L/srv/s9k4d3:97f55f8 $L/LMCache-tt-s9k4d:0e45f0ec; do
  case "$(git -C "${d%:*}" rev-parse HEAD 2>/dev/null)" in ${d#*:}*) ;; *) say "${d%:*} is not at ${d#*:} (run glx_k4d7lg_prep.sh first)"; exit 2;; esac
done
# the base = a checkout whose HEAD (its working files, which the new trees link) is 8f28c0fd0a8 or 48d2fda005a (as glx_k4d7lg_prep.sh)
S=""
for c in ${GLX_BASE_TREE:-$L/tt-metal $L/tt-metal-s9cfk3}; do
  case "$(git -C "$c" rev-parse --short=11 HEAD 2>/dev/null)" in 8f28c0fd0a8|48d2fda005a) S=$c; break;; esac
  say "base candidate $c is at $(git -C "$c" rev-parse --short=11 HEAD 2>/dev/null || echo '?') (want 8f28c0fd0a8 or 48d2fda005a): skipped"
done
[ -n "$S" ] || { say "no base checkout at 8f28c0fd0a8 or 48d2fda005a (set GLX_BASE_TREE)"; exit 2; }
BASE=$(git -C "$S" rev-parse --short=11 HEAD)
if [ -e "$S/build_Release/lib/libtt_metal.so" ]; then BLD=build_Release
elif [ -e "$S/build/lib/libtt_metal.so" ]; then BLD=build
else say "base $S has no C++ build (build_Release/lib or build/lib)"; exit 2; fi
[ -e "$S/ttnn/ttnn/_ttnn.so" ] || { say "base $S has no ttnn/ttnn/_ttnn.so"; exit 2; }
git -C "$S" bundle verify "$BUN" > /dev/null
mktree() {  # <dir> <sha> <bundle branch>
  local D=$1 SHA=$2 BR=$3 REF=refs/glxk4d7lg/$(basename "$1")
  if [ -e "$D" ]; then
    [ "$(git -C "$D" rev-parse --short=11 HEAD)" = $SHA ] || { say "$D exists, not at $SHA"; exit 1; }
    say "model tree $D exists at $SHA: kept"; return 0
  fi
  git -C "$S" fetch -q "$BUN" "+refs/heads/$BR:$REF"
  [ "$(git -C "$S" rev-parse --short=11 $REF)" = $SHA ] || { say "the bundle's $BR is not $SHA"; exit 4; }
  # outside models/: only the JIT kernel sources (+ the two nanobind files the base build already compiled) may differ
  x=$(git -C "$S" diff --name-only $BASE $SHA -- . ':!models' ':!logs' | grep -vxE \
      'ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/[a-z_]+\.(cpp|hpp)|ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_sdpa_compute\.cpp|ttnn/cpp/ttnn-nanobind/(dram_image\.cpp|page_dma_driver\.hpp)|tt_metal/impl/dispatch/kernels/cq_prefetch\.cpp' || true)
  [ -z "$x" ] || { say "$SHA differs from the base $BASE outside models/ beyond the JIT kernels: $x"; exit 4; }
  if [ -n "$(git -C "$S" diff --name-only $BASE $SHA -- tt_metal)" ]; then
    b=$(git -C "$S" rev-parse $SHA:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp)
    [ $b = $DRAIN_BLOB ] || { say "$SHA's cq_prefetch.cpp is blob $b, not 14b's drain $DRAIN_BLOB"; exit 4; }
  fi
  $GIT -C "$S" worktree add -q --no-checkout --detach "$D" $REF
  ( cd "$D"; git sparse-checkout set --no-cone '/*' '!/*/' '/models/'; git read-tree -mu HEAD
    tmo=$(git diff --name-only $BASE HEAD -- tt_metal | tr '\n' ' ')  # the drain: tt_metal/ becomes a farm
    for p in "$S"/* "$S"/.cpmcache; do
      n=$(basename "$p"); [ -e "$p" ] || continue
      case "$n" in models|ttnn|tools|generated|build) continue;; esac
      [ "$n" = tt_metal ] && [ -n "$tmo" ] && continue
      [ -e "$D/$n" ] && continue
      if [ -L "$p" ]; then cp -P "$p" "$D/$n"; elif [ -d "$p" ]; then ln -s "$p" "$D/$n"; fi
    done
    if [ $BLD = build_Release ]; then ln -s build_Release "$D/build"; else ln -s "$S/build" "$D/build"; fi
    mkdir -p "$D/generated"
    over=$(git diff --name-only $BASE HEAD -- ttnn ':!ttnn/cpp/ttnn-nanobind' | tr '\n' ' ')
    python3 -I - "$S" "$D" "$PKG/dump_fast_dispatch_halted.py" $over $tmo <<'PY'
import os, subprocess, sys
base, dest, triage, over = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
files = {p: subprocess.run(["git", "-C", dest, "show", "HEAD:" + p], check=True, capture_output=True).stdout for p in over}
files["tools/triage/dump_fast_dispatch_halted.py"] = open(triage, "rb").read()
def farm(top):
    # every dir on the path to an override file is real; every other entry is a link to the base's
    chains = {os.path.dirname(p) for p in files if p.startswith(top + "/")}
    real = {top}
    for c in chains:
        while c and c != top:
            real.add(c); c = os.path.dirname(c)
    for d in sorted(real):
        os.makedirs(os.path.join(dest, d), exist_ok=True)
        for n in sorted(os.listdir(os.path.join(base, d))):
            rel = os.path.join(d, n)
            if rel in real or rel in files:
                continue
            os.symlink(os.path.join(base, rel), os.path.join(dest, rel))
for top in ("ttnn", "tools") + (("tt_metal",) if any(p.startswith("tt_metal/") for p in files) else ()):
    farm(top)
for p, b in files.items():
    with open(os.path.join(dest, p), "wb") as f:
        f.write(b)
print(f"farms ttnn/ + tools/{' + tt_metal/' if any(p.startswith('tt_metal/') for p in files) else ''}: {len(files)} real files ({', '.join(sorted(files))})")
PY
    bad=0; while IFS= read -r f; do cmp -s "$f" <(git show "HEAD:$f") || { say "farm file $f != HEAD"; bad=1; }; done < <(find ttnn $([ -n "$tmo" ] && echo tt_metal) -type f)
    [ $bad = 0 ] || exit 4
    [ -z "$(git status --porcelain -- models)" ] || { say "$D models/ not clean"; exit 4; }
    ls "$D"/build/lib/libtt_metal.so "$D"/ttnn/ttnn/_ttnn.so "$D"/tools/triage/triage.py "$D"/tt_metal/hw/firmware > /dev/null )
  say "model tree $D at $(git -C "$D" rev-parse --short=11 HEAD) over the base $S ($BASE); cq_prefetch md5 $(md5sum < "$D"/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp | cut -c1-8)"
}
mktree $L/$TNAME $TSHA $TBR
say "done in $(( $(date +%s) - t0 )) s"
