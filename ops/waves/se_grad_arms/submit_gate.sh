#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pre-submit gate for the SE gradient wave. Run this BEFORE submit.sh.
#
# It exists for one reason: the flat arm's `--distractor.scale 0.35891393`
# reaches the producer through an environment variable, and an environment
# variable a script does not read is NOT an error. If uncfield_se.sbatch still
# hardcodes `--distractor.scale 1.0`, then DISTRACTOR_SCALE is silently
# ignored, the four flat runs train as four more ungated2 runs under flat
# run_ids, and nothing says a word for 7.5 h each. The registered identity
# gate would refuse the read afterwards, so the science is safe -- but 30
# GPU-hours are not. This turns that into a refusal before submission.
#
#   usage: submit_gate.sh <instance>
# ---------------------------------------------------------------------------
set -uo pipefail
N="${1:?usage: submit_gate.sh <instance>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"

fail=0
say () { printf '  %-6s %s\n' "$1" "$2"; [ "$1" = FAIL ] && fail=1; return 0; }

echo "== producer supports every override the three arms need =="
remote_sb="$(dv3_ssh_dv3 "$N" 'cat "$REPO/scripts/uncfield_se.sbatch"' 2>/dev/null)"
[ -n "$remote_sb" ] || { echo "FAIL: cannot read uncfield_se.sbatch on instance $N" >&2; exit 1; }

if grep -qE '[-]-distractor\.scale +"?\$\{DISTRACTOR_SCALE' <<<"$remote_sb"; then
  say PASS "--distractor.scale reads DISTRACTOR_SCALE"
else
  say FAIL "--distractor.scale does NOT read DISTRACTOR_SCALE -- the flat arm would train at scale 1.0"
fi
if grep -qE '[-]-distractor\.scale +1\.0( |\\|$)' <<<"$remote_sb"; then
  say FAIL "a hardcoded '--distractor.scale 1.0' is still present"
else
  say PASS "no hardcoded --distractor.scale literal"
fi
for v in DISTRACTOR_MOD_KEY DISTRACTOR_MOD_INDEX DISTRACTOR_MOD_LO DISTRACTOR_MOD_HI; do
  grep -q "$v" <<<"$remote_sb" && say PASS "$v wired" || say FAIL "$v NOT wired"
done

echo "== wrapper supports amplitude modulation =="
if dv3_ssh_dv3 "$N" 'grep -q "_mod_key" "$REPO/embodied/envs/distractor.py"' 2>/dev/null; then
  say PASS "distractor.py carries the mod extension"
else
  say FAIL "distractor.py has no mod extension -- the hetero arm would be inert"
fi

echo "== the frozen code on the instance is the frozen code here =="
lh="$(cd "$DV3OPS_ROOT" && git rev-parse --short HEAD)"
rh="$(dv3_ssh_dv3 "$N" 'cat "$RUNROOT/../dreamerv3/scripts/ops/VERSION" 2>/dev/null || true' 2>/dev/null | head -1)"
say INFO "local HEAD $lh / instance helpers ${rh:-unknown}"
if (cd "$DV3OPS_ROOT" && ! git diff --quiet -- scripts/uncfield_se.sbatch embodied/envs/distractor.py); then
  say FAIL "uncfield_se.sbatch / distractor.py are DIRTY locally -- freeze-commit first, then re-push, so the runs are the registered code"
else
  say PASS "producer files are committed"
fi

echo "== no clobber: none of the 12 run dirs exist yet =="
existing="$(dv3_ssh_dv3 "$N" 'cd "$RUNROOT" 2>/dev/null && ls -d se_het_s2[89] se_het_s3[01] se_flat_s3[6-9] se_ungate2_s3[2-5] 2>/dev/null' 2>/dev/null)"
if [ -z "$existing" ]; then say PASS "runroot is clean of these 12 ids"
else say FAIL "already present: $(tr '\n' ' ' <<<"$existing")"; fi

echo "== disk headroom =="
# 4.6 G per completed run, measured on RCC. A full round of 4 needs ~19 G;
# the wave needs an evacuation after round 1 regardless (12 x 4.6 = 55 G
# against a 50 G overlay). This only refuses the case where round 1 itself
# cannot fit.
avail="$(dv3_ssh_dv3 "$N" "df -BG --output=avail /workspace | tail -1 | tr -dc '0-9'" 2>/dev/null)"
if [ -n "$avail" ] && [ "$avail" -ge 25 ]; then
  say PASS "${avail}G free (round of 4 needs ~19G)"
else
  say FAIL "${avail:-?}G free -- a round of 4 needs ~19G"
fi
say INFO "12 runs x 4.6G = 55G > 50G overlay: EVACUATE round 1 to RCC during round 2"

echo
if [ "$fail" -eq 0 ]; then echo "GATE PASS -- safe to run generated/submit.sh $N"; exit 0; fi
echo "GATE FAIL -- do NOT submit" >&2; exit 1
