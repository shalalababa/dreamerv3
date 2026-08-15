#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dv3ops selfcheck (P0) -- runs anywhere, touches no instance and no cluster.
#
# Verifies the properties that would otherwise only fail on a paid instance at
# 3am: syntax, the onstart size cap, helper/shim contract, the queue-semantics
# port, and live behaviour of the pure-bash pieces against a fake RUNROOT.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
pass=0; fail=0
ok   () { printf '  \033[32mPASS\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad  () { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
chk  () { if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

echo "dv3ops selfcheck -- $ROOT"
echo
echo "== syntax =="
for f in scripts/ops/instance_helpers.sh scripts/ops/watchdog.sh \
         scripts/ops/lib/common.sh scripts/ops/dv3ops \
         scripts/ops/migrate_instance.sh scripts/ops/selfcheck.sh \
         template_onstart_script.sh; do
  chk "bash -n $f" "bash -n '$ROOT/$f'"
done

echo
echo "== onstart cap =="
SZ=$(wc -c < "$ROOT/template_onstart_script.sh")
if [ "$SZ" -lt 16384 ]; then
  ok "template $SZ bytes < 16384 (headroom $((16384-SZ)))"
else
  bad "template $SZ bytes EXCEEDS the Vast onstart cap of 16384"
fi
[ "$SZ" -lt 8192 ] && ok "template is comfortably small (was 16249 in v1)" \
                   || bad "template still large: the point of P0 was to shrink it"

echo
echo "== shim contract =="
chk "template installs a shim, not the library" \
    "grep -q 'scripts/ops/instance_helpers.sh' '$ROOT/template_onstart_script.sh'"
chk "template no longer inlines dv3_queue_supervisor" \
    "! grep -q 'dv3_queue_supervisor ()' '$ROOT/template_onstart_script.sh'"
chk "template warns when helpers are missing" \
    "grep -q 'helpers not installed yet' '$ROOT/template_onstart_script.sh'"
chk "template exposes notification config" \
    "grep -q 'DV3_NTFY_TOPIC' '$ROOT/template_onstart_script.sh'"

echo
echo "== queue semantics ported intact =="
for fn in dv3_init dv3_qc dv3_alive dv3_lane_ok dv3_clear_stale_locked dv3_status \
          dv3_tail dv3_validate_args dv3_snapshot_cmds dv3_task_name dv3_wait_gpu_idle \
          dv3_claim_next dv3_advance dv3_finish_queue dv3_abort_queue dv3_run_cmd \
          dv3_queue_cmds_list dv3_add_cmd_tasks dv3_queue_or_add dv3_remove_tasks \
          dv3_queue_supervisor dv3_cancel_queue dv3_cancel_lane; do
  chk "helper present: $fn" "grep -q '^${fn} ()' '$ROOT/scripts/ops/instance_helpers.sh'"
done
echo "  -- P0 additions --"
for fn in dv3_event dv3_notify dv3_jesc dv3_version dv3_status_json dv3_watch \
          dv3_sample_util dv3_util_summary dv3_tail_task; do
  chk "helper present: $fn" "grep -q '^${fn} ()' '$ROOT/scripts/ops/instance_helpers.sh'"
done

echo
echo "== fixes that P0 promised =="
chk "add_cmd_tasks distinguishes empty-lane (exit 3)" \
    "grep -q 'no active queue lane \$gpu\"; exit 3' '$ROOT/scripts/ops/instance_helpers.sh'"
chk "queue_or_add refuses to fall through on a real error" \
    "grep -q 'NOT the empty-lane case' '$ROOT/scripts/ops/instance_helpers.sh'"
chk "wait_gpu_idle has a ceiling" \
    "grep -q 'DV3_GPU_WAIT_CEILING_SECONDS' '$ROOT/scripts/ops/instance_helpers.sh'"
chk "run_cmd emits a failure event" \
    "grep -q 'dv3_event task_fail' '$ROOT/scripts/ops/instance_helpers.sh'"
chk "supervisor emits abort with blast radius" \
    "grep -q 'stranded' '$ROOT/scripts/ops/instance_helpers.sh'"
chk "watchdog cannot fork-bomb on source" \
    "grep -q 'BASH_SOURCE' '$ROOT/scripts/ops/watchdog.sh' && grep -q 'MUST NOT run on source' '$ROOT/scripts/ops/watchdog.sh'"
# P0 watchdog only reports. It must not call anything that mutates a queue --
# checking for the word "requeue" would match the prose in its own alerts.
chk "watchdog has no auto-mitigation in P0" \
    "! grep -qE 'dv3_(queue_cmds_list|add_cmd_tasks|queue_or_add|cancel_queue|remove_tasks)' '$ROOT/scripts/ops/watchdog.sh'"
chk "ssh helper pre-quotes arguments (§0.9)" \
    "grep -q \"printf '%q '\" '$ROOT/scripts/ops/lib/common.sh'"
chk "repo sync keeps the canonical whole-repo shape" \
    "grep -q 'root@\$(dv3_ip \"\$n\"):/workspace/dreamerv3/' '$ROOT/scripts/ops/lib/common.sh'"

echo
echo "== live behaviour (fake RUNROOT, no GPU) =="
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
# DV3_HELPERS_FILE normally points at the instance shim (/root/...), which the
# supervisor re-sources when it respawns. Locally that path does not exist, so
# point it at the repo copy -- on an instance the shim resolves to the same file.
export REPO="$ROOT" RUNROOT="$TMP/runroot" DV3_ENV_FILE="$TMP/env" \
       DV3_HELPERS_FILE="$ROOT/scripts/ops/instance_helpers.sh"
cat > "$TMP/env" <<EOF
export REPO='$ROOT'
export RUNROOT='$TMP/runroot'
export DV3_HELPERS_FILE='$ROOT/scripts/ops/instance_helpers.sh'
EOF
mkdir -p "$RUNROOT/_cloud_logs" "$RUNROOT/_queue_control"
# shellcheck disable=SC1090
source "$ROOT/scripts/ops/instance_helpers.sh"

chk "dv3_init succeeds" "dv3_init"
chk "dv3_lane_ok accepts 0"    "dv3_lane_ok 0"
chk "dv3_lane_ok accepts none" "dv3_lane_ok none"
chk "dv3_lane_ok rejects junk" "! dv3_lane_ok 'rm -rf'"

[ "$(dv3_task_name 7 'DV3_TASK_NAME=my_task foo')" = "my_task" ] \
  && ok "task_name reads DV3_TASK_NAME" || bad "task_name reads DV3_TASK_NAME"
[ "$(dv3_task_name 7 'plain command')" = "cmd_0007" ] \
  && ok "task_name falls back to index" || bad "task_name falls back to index"

printf 'a\r\n\n# comment\nb\n' > "$TMP/in.cmds"
dv3_snapshot_cmds "$TMP/in.cmds" "$TMP/out.cmds" >/dev/null 2>&1
[ "$(wc -l < "$TMP/out.cmds")" = "2" ] \
  && ok "snapshot strips CRLF, blanks, comments" || bad "snapshot strips CRLF, blanks, comments"

esc="$(dv3_jesc 'he said "hi"
and \ that')"
esc_ok=1
case "$esc" in *'\"hi\"'*) ;; *) esc_ok=0 ;; esac   # quotes escaped
case "$esc" in *'\\'*)      ;; *) esc_ok=0 ;; esac   # backslash escaped
case "$esc" in *'\n'*)      ;; *) esc_ok=0 ;; esac   # newline folded
case "$esc" in *'
'*) esc_ok=0 ;; esac                                 # no literal newline left
[ "$esc_ok" = 1 ] && ok "json escaping handles quote/backslash/newline" \
                  || bad "json escaping handles quote/backslash/newline (got: $esc)"

# events.jsonl must be valid JSON per line, since the digest parses it.
DV3_NTFY_TOPIC="" dv3_event test_kind low "title \"quoted\"" "line1
line2" '"extra":1'
if command -v python3 >/dev/null 2>&1; then
  if python3 -c "
import json,sys
for l in open('$RUNROOT/_cloud_logs/events.jsonl'):
    d=json.loads(l)
    assert d['kind']=='test_kind' and d['extra']==1 and '\n' in d['message']
" 2>/dev/null; then ok "event log emits parseable JSON"; else bad "event log emits parseable JSON"; fi
else
  ok "event log written (python3 absent; JSON not validated)"
fi

printf '%s\n' "1700000000,40,5900,16384" "1700000060,80,6100,16384" > "$TMP/u"
case "$(dv3_util_summary "$TMP/u")" in
  *"util_peak=80%"*"vram_peak=6100MiB"*) ok "util summary computes peaks" ;;
  *) bad "util summary computes peaks" ;;
esac

# The regression that motivated the rc fix: appending to a lane with no active
# queue (exit 3, a normal condition queue_or_add handles) must be
# distinguishable from a genuine failure such as lock contention (exit 1).
rc=0; dv3_add_cmd_tasks "$TMP/in.cmds" 0 >/dev/null 2>&1 || rc=$?
[ "$rc" = "3" ] && ok "add_cmd_tasks returns 3 on empty lane" \
                || bad "add_cmd_tasks returns 3 on empty lane (got $rc)"
echo
echo "== end-to-end queue, real supervisor, CPU lane =="
# lane "none" needs no GPU: validate_args skips nvidia-smi and wait_gpu_idle
# returns immediately, so the full claim/advance/abort path is exercisable here.
export DV3_QUEUE_IDLE_GRACE_SECONDS=2 DV3_NTFY_TOPIC=""
SHARDLOG="$RUNROOT/_cloud_logs/worker_shardnone.out"

printf 'DV3_TASK_NAME=sc_hold sleep 8\n' > "$TMP/hold.cmds"
if dv3_queue_cmds_list "$TMP/hold.cmds" none >/dev/null 2>&1; then
  ok "supervisor launches on lane none"
else
  bad "supervisor launches on lane none"
fi

# Now that a queue IS active, the two failure modes become distinguishable --
# which is the whole point of the exit-3 change. With no active queue the lane
# check short-circuits first, so this can only be tested here.
rc=0; dv3_add_cmd_tasks "$TMP/nonexistent.cmds" none >/dev/null 2>&1 || rc=$?
[ "$rc" != 3 ] && [ "$rc" != 0 ] \
  && ok "add to ACTIVE lane with bad file != empty-lane code (rc=$rc)" \
  || bad "add to active lane with bad file returned rc=$rc (expected a real error)"

rc=0; dv3_add_cmd_tasks "$TMP/in.cmds" none >/dev/null 2>&1 || rc=$?
[ "$rc" = 0 ] && ok "add to active lane succeeds" || bad "add to active lane succeeds (rc=$rc)"

json="$(dv3_status_json 2>/dev/null)"
if command -v python3 >/dev/null 2>&1; then
  python3 -c "
import json,sys
d=json.loads('''$json''')
assert d['lanes'], 'no lanes reported'
l=d['lanes'][0]
assert l['lane']=='none' and l['alive'] is True and l['total']>=3
" 2>/dev/null && ok "status --json reports the live lane" || bad "status --json reports the live lane"
else
  ok "status --json produced output (python3 absent; not validated)"
fi

# Cancel KILLs the supervisor process group rather than asking the loop to
# notice `cancelled`, so it usually never reaches its own QUEUE_CANCELLED line.
# The contract to assert is therefore the state, not the log: lane released and
# supervisor gone.
spid="$(cat "$RUNROOT/_queue_control/lane_none/$(cat "$RUNROOT/_queue_control/lane_none/active_queue" 2>/dev/null)/supervisor.pid" 2>/dev/null || true)"
dv3_cancel_queue none >/dev/null 2>&1
sleep 3
if [ ! -f "$RUNROOT/_queue_control/lane_none/active_queue" ] && ! dv3_alive "$spid"; then
  ok "cancel releases the lane and kills the supervisor"
else
  bad "cancel releases the lane and kills the supervisor (pid=$spid)"
fi

# The P0 acceptance criterion: a deliberate failure must abort the lane AND
# emit an event carrying the blast radius. ntfy is unset here, so this proves
# the event path independently of the network.
: > "$RUNROOT/_cloud_logs/events.jsonl"
printf 'DV3_TASK_NAME=sc_ok true\nDV3_TASK_NAME=sc_boom false\nDV3_TASK_NAME=sc_never true\n' > "$TMP/boom.cmds"
dv3_queue_cmds_list "$TMP/boom.cmds" none >/dev/null 2>&1
for _ in $(seq 1 30); do grep -q 'QUEUE_ABORTED' "$SHARDLOG" 2>/dev/null && break; sleep 1; done

grep -q 'QUEUE_ABORT ' "$SHARDLOG" 2>/dev/null \
  && ok "deliberate failure aborts the lane" || bad "deliberate failure aborts the lane"
[ -f "$RUNROOT/_cloud_logs/sc_ok.out" ] && [ ! -f "$RUNROOT/_cloud_logs/sc_never.out" ] \
  && ok "abort stops before the next task" || bad "abort stops before the next task"
grep -q '"kind":"task_fail"' "$RUNROOT/_cloud_logs/events.jsonl" 2>/dev/null \
  && ok "task failure emits an event" || bad "task failure emits an event"
if grep -q '"kind":"queue_abort"' "$RUNROOT/_cloud_logs/events.jsonl" 2>/dev/null &&
   grep -q '"stranded":1' "$RUNROOT/_cloud_logs/events.jsonl" 2>/dev/null; then
  ok "abort event carries blast radius (1 stranded)"
else
  bad "abort event carries blast radius"
fi
dv3_cancel_queue none >/dev/null 2>&1 || true

echo
echo "== dv3ops CLI =="
chk "dv3ops help runs" "bash '$ROOT/scripts/ops/dv3ops' help"
# NOTE: capture then grep. Piping a failing command into grep under `pipefail`
# fails the pipeline even when grep matched -- which is what made this check
# report a false failure on its first run.
# Use a verb that is genuinely unbuilt; `gen` exists now and asks for --wave.
gen_out="$(bash "$ROOT/scripts/ops/dv3ops" refresh 2>&1 || true)"
case "$gen_out" in *P2*) ok "unbuilt verbs name their phase" ;;
                   *) bad "unbuilt verbs name their phase (got: $gen_out)" ;; esac
wave_out="$(bash "$ROOT/scripts/ops/dv3ops" gen 2>&1 || true)"
case "$wave_out" in *--wave*) ok "wave verbs demand --wave" ;;
                    *) bad "wave verbs demand --wave (got: $wave_out)" ;; esac
chk "unknown verb fails" "! bash '$ROOT/scripts/ops/dv3ops' bogusverb"

echo
echo "== P1: spec -> artifacts =="
export RUNROOT_SAVE="$RUNROOT"
WD="$TMP/wave"; mkdir -p "$WD"
cat > "$WD/spec.yaml" <<'SPEC'
wave_id: sc_wave
kinds:
  solo: {per_gpu: 1}
  duo:  {per_gpu: 2, pairing: homogeneous}
stages:
  - name: fit
    kind: duo
    expand: {arm: [a, b], seed: [1, 2]}
    run_id: "sc_fit_${arm}_seed${seed}"
    cmd: 'echo fit ${arm} ${seed} > "$RUNROOT/${run_id}/out"'
    inputs: ["donor_${arm}/ckpt", "donor_${arm}/config.yaml"]
    done_when:
      - exists: out
      - jsonl_min_lines: {file: scores.jsonl, n: 10}
  - name: probe
    kind: solo
    needs: fit
    expand: {arm: [a, b]}
    run_id: "sc_probe_${arm}"
    cmd: 'echo probe ${arm}'
    done_when: [{exists: PROBE_DONE}]
bundle: {prefix: sc, include_stages: [fit], cloud_logs: ["sc_*.out"]}
SPEC
if python3 "$ROOT/scripts/ops/wavegen.py" "$WD" --lanes 0,1 --quiet 2>"$TMP/gen.err"; then
  ok "wavegen runs"
else
  bad "wavegen runs ($(head -2 "$TMP/gen.err"))"
fi
G="$WD/generated"
grep -q 'sc_fit_a_seed1' "$G/RUN_IDS.txt" 2>/dev/null \
  && ok "RUN_IDS lists expanded ids" || bad "RUN_IDS lists expanded ids"
grep -q 'TOTAL RUNS: 6' "$G/RUN_IDS.txt" 2>/dev/null \
  && ok "expansion count correct (4 fit + 2 probe)" || bad "expansion count correct"
# per_gpu 2 must collapse 4 runs into 2 queued tasks across 2 lanes
[ "$(cat "$G"/lane_fit_*.cmds 2>/dev/null | grep -c '^DV3_TASK_NAME=')" = "2" ] \
  && ok "per_gpu=2 yields 2 queued tasks for 4 runs" || bad "per_gpu=2 pairing"
grep -q 'wait $p1; r1=\$\?' "$G"/lane_fit_0.cmds 2>/dev/null \
  || grep -q 'wait \$p1' "$G"/lane_fit_0.cmds 2>/dev/null \
  && ok "pair barrier waits on both members" || bad "pair barrier waits on both members"
# Files vs directories: both include forms must be present (the config.yaml bug)
grep -qE -- "--include=/donor_a/config\.yaml( |$|\\\\)" "$G/push_donors.sh" 2>/dev/null \
  && ok "donor include covers plain files" || bad "donor include covers plain files"
grep -q -- "--include='/donor_a/ckpt/\*\*\*'" "$G/push_donors.sh" 2>/dev/null \
  && ok "donor include covers directory contents" || bad "donor include covers directory contents"
# $RUNROOT must survive generation unexpanded: RCC and instance paths differ.
grep -q '\$RUNROOT' "$G"/lane_fit_0.cmds 2>/dev/null \
  && ok "\$RUNROOT left for the remote shell" || bad "\$RUNROOT was expanded at generation time"

echo
echo "== P1: checker =="
FR="$TMP/fakerun"; mkdir -p "$FR"
for a in a b; do for s in 1 2; do
  d="$FR/sc_fit_${a}_seed${s}"; mkdir -p "$d"; echo x > "$d/out"
  seq 1 50 > "$d/scores.jsonl"
done; done
seq 1 40 > "$FR/sc_fit_b_seed2/scores.jsonl"     # silently short: passes n>=10
rm -rf "$FR/sc_fit_a_seed2"                       # never ran
rep="$(python3 "$ROOT/scripts/ops/wavecheck.py" "$WD" --runroot "$FR" 2>&1)"
case "$rep" in *"MISSING sc_fit_a_seed2"*) ok "checker reports a missing run" ;;
               *) bad "checker reports a missing run" ;; esac
case "$rep" in *"OUTLIER"*|*"outlier"*) ok "checker flags realized-training outlier" ;;
               *) bad "checker flags realized-training outlier" ;; esac
case "$rep" in *"sc_fit_b_seed2  n_scores=40 vs modal 50"*) ok "outlier names the run and both values" ;;
               *) bad "outlier names the run and both values" ;; esac
case "$rep" in *"PARTIAL sc_probe_a"*) ok "unstarted downstream stage is PARTIAL/MISSING" ;;
               *"MISSING sc_probe_a"*) ok "unstarted downstream stage is PARTIAL/MISSING" ;;
               *) bad "unstarted downstream stage reported" ;; esac
pend="$(python3 "$ROOT/scripts/ops/wavecheck.py" "$WD" --runroot "$FR" --pending 2>/dev/null)"
case "$pend" in *sc_fit_a_seed2*) ok "--pending lists the incomplete run" ;;
                *) bad "--pending lists the incomplete run" ;; esac
case "$pend" in *sc_fit_a_seed1*) bad "--pending wrongly lists a complete run" ;;
                *) ok "--pending excludes complete runs" ;; esac

echo
echo "== P1: spec validation rejects bad specs =="
mkbad () { mkdir -p "$TMP/bad"; printf '%s\n' "$1" > "$TMP/bad/spec.yaml"; }
mkbad 'wave_id: b
stages: [{name: s, run_id: "r_${nope}", cmd: "x"}]'
chk "unknown \${var} rejected" "! python3 '$ROOT/scripts/ops/wavegen.py' '$TMP/bad' --quiet"
mkbad 'wave_id: b
stages: [{name: s, run_id: "r", cmd: "x", target: mars}]'
chk "bad target rejected" "! python3 '$ROOT/scripts/ops/wavegen.py' '$TMP/bad' --quiet"
mkbad 'wave_id: b
kinds: {k: {per_gpu: 2, pairing: mixed}}
stages: [{name: s, kind: k, run_id: "r", cmd: "x"}]'
chk "heterogeneous co-residency rejected" "! python3 '$ROOT/scripts/ops/wavegen.py' '$TMP/bad' --quiet"
mkbad 'wave_id: b
stages: [{name: s, run_id: "same", cmd: "x", expand: {i: [1, 2]}}]'
chk "duplicate run_ids rejected" "! python3 '$ROOT/scripts/ops/wavegen.py' '$TMP/bad' --quiet"
mkbad 'wave_id: b
stages: [{name: s, run_id: "r", cmd: "x", done_when: [{bogus_pred: 1}]}]'
chk "unknown predicate rejected" "! python3 '$ROOT/scripts/ops/wavegen.py' '$TMP/bad' --quiet"

echo
echo "== P1: RCC Slurm backend =="
mkdir -p "$TMP/rcc"
cat > "$TMP/rcc/spec.yaml" <<'SPEC'
wave_id: sc_rcc
stages:
  - name: fit
    target: rcc-slurm
    expand: {seed: [1, 2, 3]}
    run_id: "sc_rcc_seed${seed}"
    sbatch:
      script: scripts/axis1.sbatch
      time: "12:00:00"
      export: {RUN_ID: "${run_id}", SEED: "${seed}", TASK: dmc_finger_turn_hard}
    done_when: [{exists: ADAPT_DONE}]
SPEC
chk "rcc-slurm stage generates submit_rcc.sh" \
    "python3 '$ROOT/scripts/ops/wavegen.py' '$TMP/rcc' --quiet --out '$TMP/rccgen' \
     && test -f '$TMP/rccgen/submit_rcc.sh'"
grep -q -- '--export=ALL,REPO=' "$TMP/rccgen/submit_rcc.sh" 2>/dev/null \
  && ok "sbatch uses the canonical --export=ALL,... form" || bad "sbatch export form"
grep -q 'MAX_JOBS:-12' "$TMP/rccgen/submit_rcc.sh" 2>/dev/null \
  && ok "12-job cap respected" || bad "12-job cap respected"
# The --export=ALL module-leakage segfault: refuse rather than inherit.
out="$(LOADEDMODULES=cuda/12 bash "$TMP/rccgen/submit_rcc.sh" --dry-run 2>&1 || true)"
case "$out" in *"REFUSING TO SUBMIT"*) ok "refuses to submit from a module-loaded shell" ;;
               *) bad "module-leakage guard did not fire" ;; esac
# A slurm target with no sbatch block would be expanded, checked and bundled
# but never submitted -- silently doing nothing. That must be a spec error.
mkbad2 () { mkdir -p "$TMP/bad2"; printf '%s\n' "$1" > "$TMP/bad2/spec.yaml"; }
mkbad2 'wave_id: b
stages: [{name: s, target: rcc-slurm, run_id: "r", cmd: "x"}]'
chk "rcc-slurm without an sbatch block is rejected" \
    "! python3 '$ROOT/scripts/ops/wavegen.py' '$TMP/bad2' --quiet"

echo
echo "== P1: generated scripts locate the repo =="
# These previously counted "../" levels from generated/, which is wrong at the
# default depth AND changes again under --out. Every generated script that runs
# on RCC must resolve the root explicitly.
# Point at committed generated output, not a dir produced later in this file:
# a check that silently skips is worse than no check.
for f in submit.sh push_donors.sh pull_results.sh bundle.sh; do
  g="$ROOT/ops/waves/lewm_upstream/generated/$f"
  if [ ! -f "$g" ]; then bad "$f missing from committed generated output"; continue; fi
  grep -q 'DV3OPS_ROOT' "$g" \
    && ok "$f resolves DV3OPS_ROOT explicitly" || bad "$f counts ../ levels"
done
# preflight_remote.sh is the exception: it runs ON an instance, where the repo
# is $REPO from the instance env, not a path known at generation time.
! grep -q 'DV3OPS_ROOT' "$ROOT/ops/waves/lewm_upstream/generated/preflight_remote.sh" \
  && ok "preflight_remote.sh correctly uses the instance's \$REPO" \
  || bad "preflight_remote.sh should not reference DV3OPS_ROOT"
grep -q 'DV3OPS_ROOT' "$TMP/rccgen/submit_rcc.sh" 2>/dev/null \
  && ok "submit_rcc.sh resolves DV3OPS_ROOT explicitly" || bad "submit_rcc.sh root"

echo
echo "== P1: reference waves reproduce real hand-written waves =="
for w in turn_easy_poscontrol tm2_bridge_fit2 lewm_graft lewm_upstream; do
  chk "reference wave loads: $w" \
      "python3 '$ROOT/scripts/ops/wavegen.py' '$ROOT/ops/waves/$w' --quiet --out '$TMP/ref_$w'"
done
grep -q 'TOTAL RUNS: 21' "$TMP/ref_turn_easy_poscontrol/RUN_IDS.txt" 2>/dev/null \
  && ok "turn_easy = 21 runs (16 adapt + 5 floor, as submitted by hand)" \
  || bad "turn_easy run count"
grep -q 'TOTAL RUNS: 48' "$TMP/ref_tm2_bridge_fit2/RUN_IDS.txt" 2>/dev/null \
  && ok "tm2_bridge = 48 runs (3 arms x 2 sides x 8 seeds)" || bad "tm2_bridge run count"
[ "$(cat "$TMP/ref_tm2_bridge_fit2"/lane_fit_*.cmds | grep -c '^DV3_TASK_NAME=')" = "24" ] \
  && ok "tm2_bridge = 24 pairs" || bad "tm2_bridge pairing count"
grep -q 'TOTAL RUNS: 32' "$TMP/ref_lewm_graft/RUN_IDS.txt" 2>/dev/null \
  && ok "lewm_graft = 32 runs (16 graft + 16 probe)" || bad "lewm_graft run count"
# The graft's fit half lives in a sibling dir the same task wrote; an adapt can
# complete on top of a world model that stopped short.
grep -q 'ckpt_step_min' "$TMP/ref_lewm_graft/runs.json" 2>/dev/null \
  && ok "lewm_graft checks the fit half in its sibling dir" || bad "lewm_graft sibling check"
[ -f "$TMP/ref_lewm_upstream/submit_rcc.sh" ] \
  && ok "lewm_upstream mixes vast and rcc-slurm stages in one wave" \
  || bad "lewm_upstream rcc stage"

echo
echo "== P2: durations, packing, board =="
mkdir -p "$TMP/plogs"
cat > "$TMP/plogs/worker_shard0.out" <<'LOG'
[Wed Aug 13 01:00:00 CDT 2026] START lewm_finger_s0_seed1 gpu=0 log=/x
[Wed Aug 13 01:19:00 CDT 2026] DONE lewm_finger_s0_seed1 log=/x
[Wed Aug 13 02:00:00 CDT 2026] START adapt_ax1jppxq1ms0_finger_seed1_ckpt500000 gpu=0 log=/x
[Wed Aug 13 09:00:00 CDT 2026] DONE adapt_ax1jppxq1ms0_finger_seed1_ckpt500000 log=/x elapsed=25200s
[Wed Aug 13 09:10:00 CDT 2026] START lewm_finger_s1_seed1 gpu=0 log=/x
[Wed Aug 13 09:30:00 CDT 2026] FAIL lewm_finger_s1_seed1 rc=1 log=/x elapsed=1200s
LOG
python3 "$ROOT/scripts/ops/durations.py" scan "$TMP/plogs" --gpu 5060Ti --out "$TMP/d.json" >/dev/null 2>&1
tbl="$(python3 "$ROOT/scripts/ops/durations.py" show --out "$TMP/d.json" 2>/dev/null)"
# Legacy logs have no elapsed=; the duration must come from the timestamps.
case "$tbl" in *"lewm_train|5060Ti"*"19m"*) ok "legacy log (no elapsed=) timed from timestamps" ;;
               *) bad "legacy log timing" ;; esac
case "$tbl" in *"graft_fit_adapt|5060Ti"*"7.0h"*) ok "P0 log elapsed= parsed" ;;
               *) bad "P0 log elapsed= parsed" ;; esac
# A failed run's runtime predicts nothing about a successful one. The log has
# one successful lewm_train (19m) and one FAILED lewm_train (20m), so a correct
# aggregate has n=1; n=2 would mean the failure was folded in.
n_train="$(printf '%s\n' "$tbl" | awk '$1 ~ /^lewm_train\|/ {print $2}')"
[ "$n_train" = "1" ] && ok "failed runs excluded from aggregates (n=1, not 2)" \
                     || bad "failed runs excluded from aggregates (got n=$n_train)"
# run_id -> spec kind, so families are the same names the specs use.
case "$tbl" in *"lewm_train"*) ok "families resolve to spec kinds" ;;
               *) bad "families resolve to spec kinds" ;; esac
chk "durations survives a corrupt db" \
    "printf 'not json' > '$TMP/bad.json'; python3 '$ROOT/scripts/ops/durations.py' show --out '$TMP/bad.json'"

pk="$(python3 "$ROOT/scripts/ops/packing.py" --wave lewm_upstream --lanes 0,1,2,3 \
      --gpu 5060Ti --durations "$TMP/d.json" 2>&1)"
case "$pk" in *"16 run(s) in 8 unit(s)"*) ok "packer treats a pair as one unit" ;;
              *) bad "packer pair unit" ;; esac
case "$pk" in *"spread 0s"*) ok "LPT balances lanes evenly on equal units" ;;
              *) bad "LPT balance" ;; esac
case "$pk" in *"23m"*) ok "co-location penalty applied to pairs (19m -> 23m)" ;;
              *) bad "co-location penalty" ;; esac

bd="$(printf '%s\n' \
  '{"instance":"1","state":{"gpus":[{"index":0,"name":"x","util":98,"mem_used":1,"mem_total":2}],"lanes":[{"lane":"0","alive":true,"next":3,"total":8,"pending":6,"running_task":"lewm_finger_s0_seed3","aborted":false}]}}' \
  '{"instance":"2","state":{"gpus":[],"lanes":[{"lane":"1","alive":false,"next":5,"total":9,"pending":5,"running_task":"z","aborted":false}]}}' \
  '{"instance":"3","state":{"gpus":[],"lanes":[]}}' \
  | python3 "$ROOT/scripts/ops/board.py" --gpu 5060Ti --durations "$TMP/d.json" 2>&1)"
case "$bd" in *"DEAD supervisors: 2/lane1"*) ok "board flags a dead supervisor" ;;
              *) bad "board flags a dead supervisor" ;; esac
case "$bd" in *"IDLE and billing: instance(s) 3"*) ok "board flags an idle billing instance" ;;
              *) bad "board flags idle instance" ;; esac
case "$bd" in *"(+1.7h)"*) ok "board projects drain time from measurements" ;;
              *) bad "board drain projection" ;; esac
case "$bd" in *"??"*) ok "board marks unmeasured kinds rather than guessing" ;;
              *) bad "board unmeasured marker" ;; esac

echo
echo "--------------------------------------------------------------"
printf 'PASS %d   FAIL %d\n' "$pass" "$fail"
[ "$fail" -eq 0 ] && echo "SELFCHECK PASS" || echo "SELFCHECK FAIL"
exit "$([ "$fail" -eq 0 ] && echo 0 || echo 1)"
