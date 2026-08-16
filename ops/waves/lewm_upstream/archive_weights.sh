#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Archive the LeWM upstream ViT weights off a rented instance. (2026-08-16)
#
#   archive_weights.sh <instance>
#
# WHY THIS EXISTS. `lewm_train` writes its checkpoint wherever
# stable-pretraining puts it -- /root/.cache/stable-pretraining/runs/<date>/
# <time>/<hash>/checkpoints/last.ckpt -- and the wave links to it:
#
#   $RUNROOT/lewm_finger_s0_seed1/lewm_weights.ckpt
#     -> /root/.cache/stable-pretraining/runs/.../last.ckpt
#
# Every result pull copied that SYMLINK, not the 206 MB behind it, so all 16
# upstream LeWMs reached RCC as links into a container path that does not
# exist there. On RCC they stat as EACCES (/root is unreadable), which is how
# this was found: the wave checker died on one of them (see wavecheck.py's
# unreadable-path handling). The distills and grafts survived because they are
# real files under $RUNROOT; the ViT weights they came from existed in exactly
# one place, on a rented box, and instances 6/8/9/10 have already been
# destroyed. Nothing about the LeWM emb leg (P-J1-emb) can be re-run without
# these.
#
# So: copy the REFERENT (-L), not the link, and land it at the path the wave
# already declares. After this, lewm_weights.ckpt is a regular file on RCC and
# the link into /root is gone.
#
# A file, not an inline ssh string: nested quoting mangles the rsync option
# array (§0.9).
# ---------------------------------------------------------------------------
set -uo pipefail

N="${1:?usage: archive_weights.sh <instance>}"

DV3OPS_ROOT="${DV3OPS_ROOT:-$HOME/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"

REMOTE_ROOT="$(dv3_ssh_dv3 "$N" 'printf %s "$RUNROOT"')"
[ -n "$REMOTE_ROOT" ] || { echo "ERROR: cannot read RUNROOT on instance $N" >&2; exit 1; }

# Which weights does the instance actually still hold? Ask it, rather than
# assuming the 16-cell expansion is intact: a link whose target is gone must be
# reported as lost, not silently skipped.
echo "== weights present on instance $N =="
present="$(dv3_ssh_dv3 "$N" '
  for d in "$RUNROOT"/lewm_finger_s[01]_seed[1-8]; do
    [ -d "$d" ] || continue
    l="$d/lewm_weights.ckpt"
    if [ -f "$l" ]; then
      printf "%s %s\n" "$(basename "$d")" "$(stat -Lc %s "$l" 2>/dev/null || echo 0)"
    else
      printf "%s MISSING\n" "$(basename "$d")"
    fi
  done')"
printf '%s\n' "$present"

lost="$(printf '%s\n' "$present" | awk '$2=="MISSING"{print $1}')"
[ -z "$lost" ] || {
  echo
  echo "WARNING: no readable weights for:"; printf '  %s\n' $lost
  echo "         (these cannot be archived from instance $N)"
}

runs="$(printf '%s\n' "$present" | awk '$2!="MISSING" && $2+0 > 0 {print $1}')"
[ -n "$runs" ] || { echo "ERROR: instance $N holds no LeWM weights" >&2; exit 1; }

src=()
for r in $runs; do src+=("root@$(dv3_ip "$N"):$REMOTE_ROOT/$r"); done

echo
echo "[lewm-weights] instance $N -> $RUNROOT/  ($(printf '%s\n' $runs | grep -c .) runs, dereferenced)"
# -L: transfer what the link POINTS AT. This is the whole point of the script;
# without it we would faithfully re-copy the same dangling link again.
rsync -az -L --info=stats2 \
  -e "ssh -p $(dv3_port "$N") ${DV3_SSH_OPTS[*]}" \
  "${src[@]}" "$RUNROOT/"
rc=$?
[ "$rc" -eq 0 ] || { echo "[lewm-weights] rsync FAILED rc=$rc" >&2; exit "$rc"; }

echo
echo "== verify: real files, matching size =="
fail=0
for r in $runs; do
  want="$(printf '%s\n' "$present" | awk -v k="$r" '$1==k{print $2}')"
  got=0; [ -f "$RUNROOT/$r/lewm_weights.ckpt" ] && got="$(stat -Lc %s "$RUNROOT/$r/lewm_weights.ckpt")"
  link=no; [ -L "$RUNROOT/$r/lewm_weights.ckpt" ] && link=yes
  if [ "$got" = "$want" ] && [ "$link" = no ]; then
    echo "  OK   $r  $got bytes"
  else
    echo "  FAIL $r  got=$got want=$want still_a_symlink=$link"; fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "[lewm-weights] VERIFY FAILED -- do not destroy instance $N" >&2; exit 1; }
echo "[lewm-weights] archived and verified"
