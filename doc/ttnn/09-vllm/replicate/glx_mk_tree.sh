#!/usr/bin/env bash
# A served model tree on the Galaxy (glx_mk_k4b_tree.sh generalized for the fleet kit, program 2026-10-27T16:54:15Z: the
# k4d3 line), HOST ONLY: glm53/<dir> at <sha> (branch <branch>) from a code-only bundle whose prerequisites the box's
# glm53/tt-metal-s9cfk3 repo has (the k3 base 48d2fda005a + the k4b branch it already fetched). <sha> must change nothing outside models/ vs 48d2fda005a (checked), so the tree is
# composed as on host-8chip: a sparse worktree (top-level files + models/), the k3 tree's C++ build entries linked (the
# same C++ build), ttnn/ copied from the k3 tree (its symlink farm; its regular files checked against HEAD), generated/.
# usage: glx_mk_tree.sh <bundle> <branch> <sha> <dir>      (env: GLX_ROOT, default /opt/glm53)
#   e.g. glx_mk_tree.sh trees/s9_k4d3_f0cd246e247.bundle glm53/glm53-flash-s9-cf3hk4d3 f0cd246e247 tt-metal-s9cfk4d3
set -euo pipefail
BUN=$(readlink -f "${1:?bundle}"); BR=${2:?branch}; SHA=${3:?sha (11 chars)}; DN=${4:?tree dir name}
G=${GLX_ROOT:-/opt/glm53}; L=$G/glm53; S=$L/tt-metal-s9cfk3; D=$L/$DN
BASE=48d2fda005a; LBR=${GLX_LOCAL_BRANCH:-$BR}  # (a test name)
GIT="git -c core.hooksPath=/dev/null"
[ -e "$D" ] && { echo "$D exists"; exit 1; }
git -C "$S" rev-parse --git-dir > /dev/null || { echo "$S is not a git checkout"; exit 2; }
git -C "$S" cat-file -e "$BASE^{commit}" || { echo "$S's repo lacks the k3 base $BASE"; exit 2; }
git -C "$S" bundle verify "$BUN" > /dev/null
git -C "$S" fetch -q "$BUN" "+refs/heads/$BR:refs/heads/$LBR"
[ "$(git -C "$S" rev-parse --short=11 "$LBR")" = $SHA ] || { echo "the bundle's $BR is not $SHA"; exit 4; }
x=$(git -C "$S" diff --name-only $BASE $SHA -- . ':!models'); [ -z "$x" ] || { echo "$SHA changes files outside models/: $x"; exit 4; }
$GIT -C "$S" worktree add -q --no-checkout "$D" "$LBR"
cd "$D"; git sparse-checkout set --no-cone '/*' '!/*/' '/models/'; git read-tree -mu HEAD
# every top-level directory of the k3 tree except models/ and ttnn/: its symlink copied, or a link to the k3 tree's dir
for p in "$S"/* "$S"/.cpmcache; do
  n=$(basename "$p"); [ -e "$p" ] || continue
  case "$n" in models|ttnn|generated) continue;; esac
  [ -e "$D/$n" ] && continue
  if [ -L "$p" ]; then cp -P "$p" "$D/$n"; elif [ -d "$p" ]; then ln -s "$p" "$D/$n"; fi
done
cp -a "$S/ttnn" "$D/ttnn"; mkdir -p "$D/generated"
# the farm (as host-8chip's): links into the C++ build tree + the regular files the JIT compiles from the model tree
# (host-8chip k4b: 5 kernel sources); every regular file must equal HEAD, like mk_cfk4d_trees.sh's check
bad=0; n=0; while IFS= read -r f; do n=$((n + 1)); cmp -s "$f" <(git show "HEAD:$f" 2>/dev/null) || { echo "ttnn farm file $f != HEAD"; bad=1; }; done < <(find ttnn -type f)
echo "ttnn farm: $n regular files == HEAD"
[ $bad = 0 ] || exit 4
ls "$D"/build_Release/lib/libtt_metal.so > /dev/null 2>&1 || ls "$D"/build/lib/libtt_metal.so > /dev/null 2>&1 \
  || { echo "no libtt_metal.so under $D/build_Release/lib or $D/build/lib (link the k3 C++ build)"; exit 4; }
echo "tree $D at $(git rev-parse --short=11 HEAD) ($BR); links: $(find "$D" -maxdepth 1 -type l -printf '%f ' )"
