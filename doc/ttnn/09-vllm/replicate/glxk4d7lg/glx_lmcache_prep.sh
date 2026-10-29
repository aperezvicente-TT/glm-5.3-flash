#!/usr/bin/env bash
# An LMCache worktree for a Galaxy served form (program [program] 2026-10-28T19:22:27Z: f2i3c runs 12's capped cold tier, LMCache
# 943926cb), HOST ONLY: glx_k4d7lg_prep.sh's mkwt for one LMCache tree -- <bundle>'s head (12's bundles carry HEAD; a branch name works
# too) fetched into the box's LMCache repo (GLX_LMCACHE_REPO, default glm53/LMCache-tt-s12ns, which holds the prerequisite 0e45f0ec of
# glm53/LMCache-tt-s9k4d) under a private ref, a detached worktree glm53/<dir> at <sha8>; an existing one must be at <sha8>.
# Run OUTSIDE the namespace with the real paths (GLX_ROOT, default /opt/glm53).
# usage: glx_lmcache_prep.sh <bundle> <sha8> <dir under glm53/> [<ref in the bundle>, default HEAD]
#   e.g. glx_lmcache_prep.sh <dir>/glm53/glxk4d7lg/lmcache_s12cap_943926cb.bundle 943926cb LMCache-tt-s9cap
set -euo pipefail
BUN=$(readlink -f "${1:?bundle}"); SHA=${2:?sha8}; N=${3:?dir under glm53/}; REF=${4:-HEAD}
G=${GLX_ROOT:-/opt/glm53}; L=$G/glm53; D=$L/$N; REPO=${GLX_LMCACHE_REPO:-$L/LMCache-tt-s12ns}
say() { echo "$(date -u +%FT%TZ) glx lmcache prep ($N): $*"; }
if [ -e "$D" ]; then case "$(git -C "$D" rev-parse HEAD)" in $SHA*) say "$D exists at $SHA: kept"; exit 0;; esac; say "$D exists, not at $SHA"; exit 1; fi
git -C "$REPO" bundle verify "$BUN" > /dev/null
git -C "$REPO" fetch -q "$BUN" "+$REF:refs/glxlmc/$N"
git -c core.hooksPath=/dev/null -C "$REPO" worktree add -q --detach "$D" "refs/glxlmc/$N"
case "$(git -C "$D" rev-parse HEAD)" in $SHA*) say "$D at $SHA";; *) say "$D not at $SHA"; exit 4;; esac
[ -z "$(git -C "$D" status --porcelain --untracked-files=no)" ] || { say "$D is not clean"; exit 4; }
say "the served env names GLM53_LMCACHE_TREE=/opt/glm53/$N (the namespace path of $D)"
