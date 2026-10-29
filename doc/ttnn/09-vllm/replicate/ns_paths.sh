#!/usr/bin/env bash
# Replication kit: run a command with the host-8chip path layout the workspace scripts expect, without editing them: a user +
# mount namespace binds the replica's dirs onto /opt/glm53 and /opt/glm53/kit{,-data}
# (the pattern host-p300's image mocks use, scripts/stage9_ops/mock_d0.sh).
# Where /opt/glm53/home (or a /mnt path) cannot be created, a tmpfs covers /home (or /mnt) and EVERY existing entry of the
# original is bound back under it (2026-10-27 01:36Z, the Galaxy's W4 attempt 3: the tmpfs hid the user's home, the venv's
# bin/python pointed at an interpreter there, and `python` resolved to /usr/bin/python without vllm).
# usage: R_GLM53=<dir for /opt/glm53> R_WS=<workspace checkout> R_DATA=<data dir> ns_paths.sh <command...>
set -u
: "${R_GLM53:?}" "${R_WS:?}" "${R_DATA:?}"
exec unshare -rm bash -c '
  cover() {  # cover <dir> <path inside it to create>: tmpfs over <dir>, its original entries bound back
    local d=$1 o; o=$(mktemp -d /tmp/ns_orig.XXXXXX) && mount --rbind "$d" "$o" && mount -t tmpfs none "$d" || return 9
    for e in "$o"/* "$o"/.[!.]*; do [ -e "$e" ] || continue
      if [ -d "$e" ]; then mkdir -p "$d/${e##*/}"; else touch "$d/${e##*/}"; fi; mount --rbind "$e" "$d/${e##*/}"; done
    mkdir -p "$2"; }
  mkdir -p /opt/glm53 2>/dev/null || cover /home /opt/glm53 || exit 9
  mount --bind "$0" /opt/glm53 || exit 9
  for p in /opt/glm53/kit /opt/glm53/data; do mkdir -p "$p" || exit 9; done
  mount --bind "$1" /opt/glm53/kit && mount --bind "$2" /opt/glm53/data || exit 9  # the share also lives under /opt/glm53/home/mpi-shfs on host-8chip (tp_image.py weights_identity reads weights/manifest.json there)
  for p in /opt/glm53/kit /opt/glm53/data; do
    mkdir -p "$p" 2>/dev/null || cover /mnt "$p" || exit 9; done
  mount --bind "$1" /opt/glm53/kit && mount --bind "$2" /opt/glm53/data || exit 9
  # the share llm-wksp dir (HF snapshots, tt-cache) under its two host-8chip names, when the replica provides <glm53>/../llm-wksp
  if [ -d "$0/../llm-wksp" ]; then for p in /opt/glm53/data /opt/glm53/share/llm-wksp; do mkdir -p "$p" 2>/dev/null && mount --bind "$0/../llm-wksp" "$p"; done; fi
  shift 2; exec "$@"' "$R_GLM53" "$R_WS" "$R_DATA" "$@"
