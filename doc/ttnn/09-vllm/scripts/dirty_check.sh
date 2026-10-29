#!/usr/bin/env bash
# Stage 9's dirty-box marker (work-s9/NEEDS_RESET.all; review rounds 19-22: a failed mesh open closed "cleanly" --
# UMD's "Closing devices in cluster completed" -- and the launcher re-filed the request for a chip that needed a reset).
#   dirty_check.sh scan <server.log> [<kernel_window.txt>]
#       -> "" (clean) or "<reasons>; chips=<tt-lock chips|all>" (dirty)
#   dirty_check.sh cleared <marker> <journal>
#       -> rc 0 when tt-lock's reset.journal shows a successful reset (rc=0) STARTED after the marker's time whose
#          chips cover the marker's chips
#   dirty_check.sh table
#       -> this host's chip table now ("<dev> <bdf> <root port> <chip>" per chip); serve_window.sh saves it as
#          <run>/tt_chips.txt when it takes the mesh, and `scan` uses that copy (a reset / re-enumeration between the
#          run and the scan cannot shift the numbers)
# Dirty (the reset policy's triggers: open failures, all-1s reads, Completion Timeouts, unclean deaths; plus hangs):
#  - server.log: a tt-metal / UMD device-failure line (FW init / open failure, firmware startup error, device timeout /
#    hang, NOC hang, all-1s read, SIGBUS device access, MMIO / memory-barrier / DMA timeout, DRAM / ETH training,
#    watcher stop);
#  - kernel_window.txt: on a Tenstorrent chip or its root port, a Completion Timeout, an uncorrected AER (native or GHES
#    firmware-first records), an all-1s read, a KMD failure (ARC timeout / message failure, NOC hung, power state,
#    power cable fault), link down / surprise down; corrected errors and other devices' errors (the NIC and its
#    switch) stay log-only; KMD firmware-log error records have no fixed text and stay log-only;
#  - the server reached the device open ("Opening user mode device driver") and did not close it orderly: ALL chips,
#    whatever else was named; a server that never reached the device open (a tt-lock refusal, a port clash, a Python
#    error) is not dirty.
# CHIP NUMBERS = tt-lock chips = TT_VISIBLE_DEVICES integers = the rank of the chip's BDF among the host's sorted
# Tenstorrent BDFs (UMD pci_device.cpp). /dev/tenstorrent/<N> is a DIFFERENT order (here /dev/5 = 0000:02:00.0 =
# chip 1 = card0; card2 = /dev/0,1); doc/hardware.md §2 predates the 10-16 card swap, the live sysfs table rules.
# Naming per failure line:
#  - UMD "PCIe ID N" / "device ID N" / "Firmware startup error on device N" / "Failed to open device N" /
#    "/dev/tenstorrent/N" = /dev N -> its BDF -> its rank;
#  - tt-metal "Device N" = logical id N -> the N-th /dev number in the run's own "Opening local chip ids/PCIe ids:
#    {..}/[d0, d1, ...]" line -> its BDF -> its rank (no map line -> all);
#  - a line naming no chip, any other "device N", anything unmapped, or a chip table without GLM53_DIRTY_NCHIPS (8)
#    rows -> chips=all; a "Timeout detected" preamble counts only when no other failure line exists.
# Test hook: GLM53_DIRTY_TT_TABLE="dev=bdf=rootport ..." (else <run>/tt_chips.txt, else the live sysfs).
# Marker line: "<UTC>Z stage9 <run> rc=<rc>: <reason>; chips=<ids|all>" (an older marker without chips= means all; all =
# the chips of that run's saved table). The numbers are the run's: `cleared` refuses whenever the run's saved table has
# another BDF set than the live box (a renumbering, e.g. 2026-10-21 17:45Z): such a marker clears only on a coordinator
# ruling, by hand -- tt-lock's new chip numbers are not the run's. The check runs when `cleared` is asked (the
# launcher's poll), against that moment's box, and only when <run>/tt_chips.txt exists with content: a marker whose run
# has no saved table (runs before round 22, an empty table, a moved / pruned run dir) skips it.
set -u
SIG='Try resetting the board|failed to initialize FW|Firmware startup error|Failed to open device|Timeout detected'
SIG+='|device timeout|potential hang|is unrecoverable|waiting for physical cores|the board should be reset'
SIG+='|should reset the board|appears hung| is hung|SIGBUS signal detected|Device access failed|MMIO per-op timeout'
SIG+='|Memory barrier timed out|DMA timeout|training.* timed out|training failed|Watcher detected'
SIG+='|watcher.*(stopped|assert)|all-1s|all ones|Completion Timeout'
DEVN='((PCIe ID|PCIe device ID|device ID|Firmware startup error on device|Failed to open device) '
DEVN+='|/dev/tenstorrent/)[0-9]+'
KERR='CmpltTO|Completion Timeout|Uncorrect|0xffffffff|all-1s|all ones|Timeout waiting for ARC|NOC is likely hung'
KERR+='|Failed to send ARC message|Failed to set initial power state|power cable fault|[Ll]ink [Dd]own|[Ss]urprise'
ALL=0,1,2,3,4,5,6,7

tt_table() {  # "dev bdf rootport chip" per Tenstorrent chip; chip = rank of the BDF in sorted order
  {
    if [ -n "${GLM53_DIRTY_TT_TABLE:-}" ]; then tr ' ' '\n' <<<"$GLM53_DIRTY_TT_TABLE" | tr '=' ' '
    elif [ -n "${1:-}" ] && [ -e "$1" ]; then awk 'NF == 4 {print $1, $2, $3}' "$1"  # the run's saved table (an
                                                       # empty one = a failed save: 0 rows -> all, never sysfs)
    else
      local d p
      for d in /sys/class/tenstorrent/tenstorrent!*; do
        [ -e "$d/device" ] || continue; p=$(readlink -f "$d/device")
        echo "${d##*!} $(basename "$p") $(basename "$(dirname "$p")")"
      done
    fi
  } | awk 'NF == 3' | sort -k2,2 | awk '{print $1, $2, $3, NR - 1}'
}

name_line() {  # one failure line -> chip numbers (one per line) or "all"
  local l=$1 rest p d c devs logs
  rest=$(sed -E "s#$DEVN##g; s#\bDevice [0-9]+##g" <<<"$l")  # '#': DEVN contains '/'
  grep -qiE "device [0-9]+" <<<"$rest" && { echo all; return; }  # another "device N": its numbering is unknown
  devs=$(grep -oE "$DEVN" <<<"$l" | grep -oE "[0-9]+$")
  logs=$(grep -oE "\bDevice [0-9]+" <<<"$l" | grep -oE "[0-9]+$")
  [ -n "$devs$logs" ] || { echo all; return; }
  for p in $devs; do
    c=$(awk -v p="$p" '$1 == p {print $4}' <<<"$TT"); [ -n "$c" ] && echo "$c" || { echo all; return; }
  done
  for d in $logs; do  # logical d -> the run's d-th /dev number -> its chip
    p=${MAPDEV[$d]:-}; [ -n "$p" ] || { echo all; return; }
    c=$(awk -v p="$p" '$1 == p {print $4}' <<<"$TT"); [ -n "$c" ] && echo "$c" || { echo all; return; }
  done
}

scan() {
  local log=$1 kw=${2:-} sig named l chips="" reasons=() k opened=0 closed=0 rows
  TT=$(tt_table "$(dirname "$log")/tt_chips.txt")
  rows=$(grep -c . <<<"$TT")
  IFS=, read -ra MAPDEV <<<"$(grep -oE "Opening local chip ids/PCIe ids: \{[0-9, ]+\}/\[[0-9, ]+\]" "$log" 2>/dev/null |
    tail -1 | sed -E 's/.*\/\[([0-9, ]+)\]/\1/' | tr -d ' ')"
  # opened / closed in order: the LAST device open must be followed by an orderly close (review round 23)
  local lo lc; lo=$(grep -n "Opening user mode device driver" "$log" 2>/dev/null | tail -1 | cut -d: -f1)
  lc=$(grep -n "Closing devices in cluster completed" "$log" 2>/dev/null | tail -1 | cut -d: -f1)
  [ -n "$lo" ] && opened=1; [ -n "$lc" ] && { [ -z "$lo" ] || [ "$lc" -gt "$lo" ]; } && closed=1
  sig=$(grep -E "$SIG" "$log" 2>/dev/null | head -40)
  if [ -n "$sig" ]; then
    named=$(grep -vE "^[^|]*\| *error *\| *Metal *\| *Timeout detected" <<<"$sig")  # the context-only preamble
    [ -n "$named" ] || named=$sig
    while IFS= read -r l; do chips="$chips $(name_line "$l" | tr '\n' ' ')"; done <<<"$named"
    reasons+=("device failure: $(head -1 <<<"$named" | grep -oE "($SIG).{0,80}" | head -1 | tr -d '\r')")
  fi
  if [ -n "$kw" ] && [ -s "$kw" ]; then
    k=$(awk -v tt="$TT" -v kerr="$KERR" '
      BEGIN { n = split(tt, rows, "\n")
              for (i = 1; i <= n; i++) { split(rows[i], f, " "); chip[f[2]] = f[4]; chip[f[3]] = f[4] } }
      /Configured ACS/ { next }
      /Hardware error from APEI/ { sev = ""; next }
      /\[Hardware Error\]: +event severity:/ { sev = $NF; next }
      /\[Hardware Error\]: +device_id: [0-9a-f]{4}:/ {
        match($0, /[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]/); b = substr($0, RSTART, RLENGTH)
        if (sev != "" && sev != "corrected" && (b in chip)) print chip[b] "\tGHES " sev " error on " b
        next }
      { hit = ""; for (b in chip) if (index($0, b)) { hit = b; break }
        if ($0 !~ kerr) next
        i = index($0, "kernel: "); txt = substr($0, i ? i + 8 : 1, 100)
        if (hit != "") print chip[hit] "\t" txt
        else if ($0 ~ /tenstorrent [0-9a-f]{4}:/) print "all\t" txt }' "$kw")  # a chip missing from the table
    if [ -n "$k" ]; then
      chips="$chips $(cut -f1 <<<"$k" | tr '\n' ' ')"; reasons+=("kernel: $(head -1 <<<"$k" | cut -f2)")
    fi
  fi
  # an unclean death after the device open: every chip the process held, whatever a failure line named (review 22)
  if [ $opened = 1 ] && [ $closed = 0 ]; then chips="$chips all"; reasons+=("no orderly mesh close"); fi
  [ ${#reasons[@]} = 0 ] && return 0  # clean (also: never reached the device open)
  [ "$rows" = "${GLM53_DIRTY_NCHIPS:-8}" ] || chips="$chips all"  # an incomplete chip table: no number is safe
  chips=$(tr ' ' '\n' <<<"$chips" | grep -v '^$' | sort -u)
  grep -qx all <<<"$chips" && chips=all || chips=$(sort -n <<<"$chips" | paste -sd,)
  local out="" r; for r in "${reasons[@]:0:3}"; do out="${out:+$out; }$r"; done
  echo "$out; chips=$chips"
}

cleared() {
  local marker=$1 journal=$2 t need
  t=$(head -1 "$marker" | grep -oE "^[0-9T:-]+Z") || return 1
  # program 2026-10-28T11:02:13Z (a stale work-s9-blk2 marker refused k3srvo after the base tool's reset): (1) a marker that
  # records boot=<id> pid=<n> clears when the box rebooted since (boot id changed) and that pid is gone; (2) on the Galaxy
  # (GLM53_DIRTY_GROUP=blk<N>, the block's tt-lock group) a done rc=0 reset of THAT group or 'all' after the marker clears it --
  # the run saw 8 chips of a 32-chip box, so the host-8chip BDF-set / chip-number rule below cannot apply there
  local mb mp; mb=$(head -1 "$marker" | grep -oE "boot=[0-9a-f-]+" | cut -d= -f2); mp=$(head -1 "$marker" | grep -oE "pid=[0-9]+" | cut -d= -f2)
  if [ -n "$mb" ] && [ "$mb" != "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)" ] && { [ -z "$mp" ] || [ ! -e /proc/$mp ]; }; then return 0; fi
  if [ -n "${GLM53_DIRTY_GROUP:-}" ]; then
    awk -v t="$t" -v g="$GLM53_DIRTY_GROUP" '$2 == "reset" && $4 == "done" && $5 == "rc=0" && $1 >= t && ($3 == g || $3 == "all") { ok = 1 } END { exit ok ? 0 : 1 }' "$journal"
    return
  fi
  need=$(head -1 "$marker" | grep -oE "chips=[0-9,a-z]+" | tail -1 | cut -d= -f2)
  local run tbl; run=$(head -1 "$marker" | awk '{print $3}')
  tbl=${GLM53_DIRTY_RUNS:-/opt/glm53/work-s9/runs}/$run/tt_chips.txt
  # the marker's chip numbers are the run's (BDF ranks of the chips present then): they name the same chips as
  # tt-lock's only while the box has the same chips (the same BDF set); after a renumbering (a chip lost from the bus,
  # 2026-10-21 17:45Z) never auto-clear: a coordinator ruling clears it by hand (review round 25)
  if [ -s "$tbl" ] && [ "$(awk 'NF == 4 {print $2}' "$tbl" | sort)" != "$(tt_table | awk '{print $2}' | sort)" ]; then
    return 1
  fi
  if [ -z "$need" ] || [ "$need" = all ]; then  # all = every chip of THAT run's saved table (else the 8-chip set)
    need=""; [ -s "$tbl" ] && need=$(awk 'NF == 4 {print $4}' "$tbl" | paste -sd,)
    [ -n "$need" ] || need=$ALL  # no usable rows: all 8, never an empty (vacuously cleared) set
  fi
  # journal: "<ts> reset <group> chips=<c> by ..." then "<ts> reset <group> done rc=<rc>"; ISO-8601 UTC sorts as text
  awk -v t="$t" -v need="$need" '
    $2 == "reset" && $4 ~ /^chips=/ { start[$3] = $1; chips[$3] = substr($4, 7); next }
    $2 == "reset" && $4 == "done" && ($3 in start) {
      # 2026-10-28: the reset counts by its DONE time (tt-lock reset needs every device node closed, so a reset that ends after the
      # marker ran after the engine closed; the 06:23:50Z reset began 2 s before serve_window wrote its marker and was not counted)
      if ($5 == "rc=0" && $1 >= t) { n = split(chips[$3], c, ","); for (i = 1; i <= n; i++) done_[c[i]] = 1 }
      delete start[$3] }
    END { n = split(need, w, ","); for (i = 1; i <= n; i++) if (!(w[i] in done_)) exit 1; exit 0 }' "$journal"
}

case "${1:-}" in
  table) tt_table ;;
  scan) scan "$2" "${3:-}" ;;
  cleared) cleared "$2" "$3" ;;
  *) echo "usage: dirty_check.sh table | scan <server.log> [<kernel_window.txt>] | cleared <marker> <journal>" >&2
     exit 2 ;;
esac
