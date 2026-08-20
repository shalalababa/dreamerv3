#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Open (or reuse) one shared SSH connection to RCC.  (2026-08-15)
#
#   dv3ops rcc-connect            # open if needed, else report the live one
#   dv3ops rcc-connect --status   # report only, never authenticate
#   dv3ops rcc-connect --close    # drop the shared connection
#
# Run this once when you open your laptop. Everything afterwards that talks to
# RCC -- pull-local, the wave verbs, ad-hoc ssh -- rides the same socket and
# never asks for a password or a 2FA code again until it expires.
#
# Why a master socket rather than an agent or a key: RCC requires interactive
# 2FA, so there is no unattended key to install. ControlMaster pays that cost
# ONCE and multiplexes every later session over the same TCP connection. It is
# already the house convention -- pull_local.sh uses this exact ControlPath, so
# opening it here makes that script password-free too.
#
# Idempotent on purpose: running it twice must not open a second connection or
# re-prompt, because "did I already start it?" is exactly the question you have
# when you sit down.
# ---------------------------------------------------------------------------
set -uo pipefail

RCC_HOST="${DV3_RCC_HOST:-rickybao@midway3-login3.rcc.uchicago.edu}"
PERSIST="${DV3_RCC_PERSIST:-24h}"
CTL_DIR="$HOME/.ssh/cm"
CTL="$CTL_DIR/%r@%h:%p"

MODE=open
while [ $# -gt 0 ]; do
  case "$1" in
    --status) MODE=status; shift ;;
    --close)  MODE=close;  shift ;;
    --host)   RCC_HOST="${2:?--host needs a value}"; shift 2 ;;
    *) echo "usage: dv3ops rcc-connect [--status|--close] [--host user@host]" >&2
       exit 2 ;;
  esac
done

mkdir -p "$CTL_DIR"
chmod 700 "$CTL_DIR" 2>/dev/null || true

alive () { ssh -O check -o "ControlPath=$CTL" "$RCC_HOST" >/dev/null 2>&1; }

# A dead master routinely leaves its socket FILE behind -- WSL sleeping, the
# network changing, a server-side timeout, or the master being SIGKILLed. The
# process may even linger while its connection is gone. `ssh -O check` then
# fails (Connection refused) while the file is still on disk, and ssh REFUSES
# to create a new master over an existing socket path: it prints
#   ControlSocket ... already exists, disabling multiplexing
# then connects WITHOUT a master, backgrounds via -f, and exits 0. The caller
# sees rc=0 and no connection, which reads like a credentials problem and is
# not one. So clear the corpse before trying to open.
sock_path () {
  local u h
  case "$RCC_HOST" in
    *@*) u="${RCC_HOST%%@*}"; h="${RCC_HOST#*@}" ;;
    *)   u="${USER:-$(id -un)}"; h="$RCC_HOST" ;;
  esac
  ls -1 "$CTL_DIR/$u@$h:"* 2>/dev/null   # glob the port: non-22 sockets count
}

reap_stale () {
  local s
  while IFS= read -r s; do
    [ -n "$s" ] || continue
    echo "rcc: stale control socket ($s) -- removing"
    rm -f "$s"
  done < <(sock_path)
  # A master whose socket is dead holds nothing worth keeping.
  if pkill -f "ssh -MNf.*ControlPersist.*$RCC_HOST" 2>/dev/null; then
    echo "rcc: reaped a defunct master process"
    sleep 1
  fi
  return 0
}

case "$MODE" in
  status)
    if alive; then echo "rcc: CONNECTED  $RCC_HOST"; exit 0
    else echo "rcc: not connected ($RCC_HOST)"; exit 1; fi
    ;;
  close)
    if alive; then
      ssh -O exit -o "ControlPath=$CTL" "$RCC_HOST" >/dev/null 2>&1
      echo "rcc: closed $RCC_HOST"
    elif [ -n "$(sock_path)" ]; then
      # Not alive, but a socket file remains -- this is exactly the state that
      # makes a later `rcc-connect` fail with rc=0, so --close must clear it
      # rather than report "nothing to close".
      reap_stale
      echo "rcc: cleared a stale socket (there was no live connection)"
    else
      echo "rcc: nothing to close"
    fi
    exit 0
    ;;
esac

if alive; then
  echo "rcc: already connected to $RCC_HOST -- nothing to do"
  exit 0
fi

reap_stale   # otherwise ssh declines to multiplex and exits 0
echo "rcc: opening a shared connection to $RCC_HOST (persists ${PERSIST})"
echo "     you will be asked for your password + 2FA once"
# -M master, -N no command, -f background AFTER authenticating (so the prompt
# still reaches your terminal). No BatchMode: 2FA is interactive by nature.
ssh -MNf \
    -o ControlMaster=yes \
    -o "ControlPath=$CTL" \
    -o "ControlPersist=$PERSIST" \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=10 \
    "$RCC_HOST"
rc=$?

if [ "$rc" -ne 0 ] || ! alive; then
  echo "rcc: FAILED to open a shared connection (ssh rc=$rc)" >&2
  echo "     try plain 'ssh $RCC_HOST' first -- if that works, the problem is" >&2
  echo "     the socket path, not your credentials: $CTL_DIR" >&2
  if [ "$rc" -eq 0 ]; then
    echo "     rc=0 with no connection means ssh declined to multiplex over an" >&2
    echo "     existing socket file. Clear it with: dv3ops rcc-connect --close" >&2
  fi
  exit 1
fi

echo "rcc: CONNECTED. Verifying it actually carries commands..."
# A socket that exists is not a socket that works. Prove it end to end.
who="$(ssh -o BatchMode=yes -o ControlMaster=no -o "ControlPath=$CTL" \
        "$RCC_HOST" 'hostname' 2>/dev/null | tail -1)"
if [ -n "$who" ]; then
  echo "rcc: verified -> $who   (expires after ${PERSIST} idle)"
else
  echo "rcc: socket opened but a test command returned nothing" >&2
  exit 1
fi
