#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dv3ops rcc-nodes -- survey EVERY RCC login node, not just the configured one.
#
# Why this exists (31 Aug 2026): midway3-login3's sshd died while login1/2/4
# stayed up. `rcc-connect --status` probes only the one host it is configured
# for -- and it defaults to login3 -- so the failure read as "RCC is down" when
# three of four nodes were fine. That cost 2.5 h of idle billing on two boxes,
# because the detached drain loops lived on login3 and died with it.
#
# Reads the three states apart, which is the whole point:
#   OPEN      TCP :22 accepts        -> node up, sshd up
#   REFUSED   TCP :22 refused        -> node up, sshd DOWN  (login3's failure)
#   FILTERED  TCP :22 times out      -> node or network down
# and separately whether a ControlMaster socket for it is LIVE.
#
# `Permission denied` from a probe is NOT a fault: RCC requires 2FA, so there
# is no unattended key by design and every RCC command rides the user's shared
# socket. It means sshd is healthy.
#
# Usage: dv3ops rcc-nodes [--user USER] [--timeout SECS]
# ---------------------------------------------------------------------------
set -uo pipefail

USER_AT="${DV3_RCC_USER:-rickybao}"
TMO="${DV3_RCC_PROBE_TIMEOUT:-8}"
NODES="${DV3_RCC_NODES:-midway3-login1 midway3-login2 midway3-login3 midway3-login4}"
CTL_DIR="$HOME/.ssh/cm"

while [ $# -gt 0 ]; do
  case "$1" in
    --user)    USER_AT="${2:?--user needs a value}"; shift 2 ;;
    --timeout) TMO="${2:?--timeout needs a value}"; shift 2 ;;
    *) echo "usage: dv3ops rcc-nodes [--user USER] [--timeout SECS]" >&2; exit 2 ;;
  esac
done

printf '%-16s %-16s %-9s %-8s %s\n' NODE IP PORT22 SOCKET NOTE
up=0; total=0
for n in $NODES; do
  total=$((total+1))
  host="$n.rcc.uchicago.edu"
  ip=$(getent hosts "$host" 2>/dev/null | awk '{print $1}' | head -1)
  if [ -z "$ip" ]; then
    printf '%-16s %-16s %-9s %-8s %s\n' "$n" "-" "NXDOMAIN" "-" "DNS failure"
    continue
  fi
  # connect-only: never read, an ssh banner is short and a read blocks.
  timeout "$TMO" bash -c "exec 3<>/dev/tcp/$ip/22" 2>/dev/null
  case $? in
    0)   state=OPEN ;;
    124) state=FILTERED ;;
    *)   state=REFUSED ;;
  esac
  sock="-"
  if [ -S "$CTL_DIR/$USER_AT@$host:22" ]; then
    if ssh -O check -o "ControlPath=$CTL_DIR/%r@%h:%p" "$USER_AT@$host" >/dev/null 2>&1
    then sock=LIVE; else sock=STALE; fi
  fi
  note=""
  case "$state" in
    OPEN)     note="node up, sshd up"; up=$((up+1)) ;;
    REFUSED)  note="node up, SSHD DOWN -- processes here are gone" ;;
    FILTERED) note="node or network down" ;;
  esac
  [ "$sock" = LIVE ] && note="$note; shared socket usable"
  [ "$sock" = STALE ] && note="$note; STALE socket -- rcc-connect --close first"
  printf '%-16s %-16s %-9s %-8s %s\n' "$n" "$ip" "$state" "$sock" "$note"
done

echo
if [ "$up" -eq 0 ]; then
  echo "ALL $total login nodes unreachable -- this one really is an RCC outage."
else
  echo "$up/$total login nodes up. /home and /scratch are SHARED, so a dead node"
  echo "loses only PROCESSES, never data. Reconnect elsewhere with:"
  echo "  dv3ops rcc-connect --host $USER_AT@<one of the OPEN nodes>.rcc.uchicago.edu"
fi
