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
# ssh reads stdin; an ssh inside a heredoc-fed script eats the rest of that
# script and every later command silently never runs.
chk "bare-command ssh uses -n so it cannot eat its parent script" \
    "grep -q 'ssh -n -p' '$ROOT/scripts/ops/lib/common.sh'"
# push-repo must replace the shim, not just ship helpers into the repo: an
# instance that booted (or rebooted) under the v1 template still has the whole
# v1 library at /root/dreamer_instance_helpers.sh, and nothing would point it
# at the new one -- the watchdog works while every shell reports
# "dv3_version: command not found".
chk "push-repo installs the helper shim" \
    "grep -q 'shim installed' '$ROOT/scripts/ops/lib/common.sh'"
chk "push-repo backs up inline v1 helpers before replacing them" \
    "grep -q 'dreamer_instance_helpers.v1.bak' '$ROOT/scripts/ops/lib/common.sh'"
chk "ssh helper pre-quotes arguments (§0.9)" \
    "grep -q \"printf '%q '\" '$ROOT/scripts/ops/lib/common.sh'"
# dv3_ssh_dv3 pipes its own wrapper into ssh, so a remote `bash -s` reads an
# already-consumed stdin and silently executes nothing. preflight did exactly
# that: it printed no checks, and an empty report is indistinguishable from a
# clean pass (2026-08-15). Scripts must be passed as TEXT.
chk "no verb feeds a remote 'bash -s' on stdin" \
    "! grep -q 'dv3_ssh_dv3 \"\$n\" \"bash -s\"' '$ROOT/scripts/ops/dv3ops'"
chk "preflight refuses to treat silence as success" \
    "grep -q 'preflight produced no checks' '$ROOT/scripts/ops/dv3ops'"
# --append-verify only APPENDS to files shorter on the destination and skips
# same-size files entirely. On a code sync that means an edited file of
# unchanged length never propagates -- which is why VERSION kept reporting the
# previous commit after a successful push (2026-08-15).
chk "repo sync does NOT use --append-verify" \
    "! grep -q 'DV3_CODE_RSYNC_OPTS=(.*append-verify' '$ROOT/scripts/ops/lib/common.sh'"
chk "repo sync uses --checksum" \
    "grep -q 'DV3_CODE_RSYNC_OPTS=(-az --checksum' '$ROOT/scripts/ops/lib/common.sh'"
chk "push-repo uses the CODE option set, not the payload one" \
    "grep -q 'rsync \"\${DV3_CODE_RSYNC_OPTS\[@\]}\" --delete' '$ROOT/scripts/ops/lib/common.sh'"
chk "payload sync KEEPS --append-verify (resumable, immutable)" \
    "grep -q 'DV3_RSYNC_OPTS=(-az --partial --append-verify' '$ROOT/scripts/ops/lib/common.sh'"
chk "generated submit.sh checksums the command files" \
    "grep -q 'rsync -az --checksum' '$ROOT/scripts/ops/wavegen.py'"
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
# Every phase is built, so there is no longer an unbuilt verb to test. What
# still matters is that a verb blocked by a MISSING DEPENDENCY says how to fix
# it rather than dying on a bare "command not found".
dep_out="$(bash "$ROOT/scripts/ops/dv3ops" refresh 2>&1 || true)"
case "$dep_out" in
  *"pip install vastai"*) ok "a missing dependency names its own fix" ;;
  *IDX*|*"VAST ID"*)      ok "vastai present; refresh ran" ;;
  *) bad "missing-dependency error is not actionable (got: $dep_out)" ;;
esac
wave_out="$(bash "$ROOT/scripts/ops/dv3ops" gen 2>&1 || true)"
case "$wave_out" in *--wave*) ok "wave verbs demand --wave" ;;
                    *) bad "wave verbs demand --wave (got: $wave_out)" ;; esac
chk "unknown verb fails" "! bash '$ROOT/scripts/ops/dv3ops' bogusverb"

echo
echo "== push-repo isolates failures per instance =="
# dv3_push_repo calls die(), and die() exits the SCRIPT. A single unreachable
# instance therefore aborted the loop and silently skipped every one after it,
# so the fleet looked pushed because the command "finished".
chk "push-repo runs each instance in a subshell" \
    "grep -q '( dv3_push_repo' '$ROOT/scripts/ops/dv3ops'"
chk "push-repo reports which instances failed" \
    "grep -q 'still run old code' '$ROOT/scripts/ops/dv3ops'"
chk "push-repo verifies the stamp landed afterwards" \
    "grep -q 'bash \"\$0\" versions' '$ROOT/scripts/ops/dv3ops'"
loopres="$(bash -c '
die () { echo "ERROR: $*" >&2; exit 1; }
push_one () { [ "$1" = 2 ] && die "unreachable"; echo "pushed $1"; }
ok=(); bad=()
for n in 1 2 3 4; do if ( push_one "$n" ); then ok+=("$n"); else bad+=("$n"); fi; done
echo "ok=${ok[*]} bad=${bad[*]}"' 2>/dev/null)"
case "$loopres" in *"ok=1 3 4 bad=2"*) ok "one failure no longer skips later instances" ;;
                   *) bad "loop isolation (got: $loopres)" ;; esac

echo
echo "== migration resumes what it cancelled =="
# A migration that cancels lanes and then only writes instructions can complete
# "successfully" while leaving the instance idle and the wave dead. That
# happened on inst1, 2026-08-15.
chk "migrate requeues the captured lanes" \
    "grep -q 'dv3_queue_cmds_list /tmp/' '$ROOT/scripts/ops/migrate_instance.sh'"
chk "migrate resume is opt-OUT, not opt-in" \
    "grep -q 'SKIP_RESUME:-0' '$ROOT/scripts/ops/migrate_instance.sh'"
chk "migrate still snapshots realized counters before cancelling" \
    "grep -q 'realized_before.json' '$ROOT/scripts/ops/migrate_instance.sh'"
# The resume set must include the INTERRUPTED task, not just the untouched
# queue behind it.
mkdir -p "$TMP/mig"
cat > "$TMP/mig/lanes.txt" <<'LANES'
===LANE 0 queue_abc next=3 total=4 running=3
===TASKS
DV3_TASK_NAME=t1 c1
DV3_TASK_NAME=t2 c2
DV3_TASK_NAME=t3 c3
DV3_TASK_NAME=t4 c4
===END
LANES
awk '
    /^===LANE /   { lane=$2; nx=0; for(i=1;i<=NF;i++) if ($i ~ /^next=/) { split($i,a,"="); nx=a[2] } ; n=0; next }
    /^===TASKS$/  { intask=1; n=0; next }
    /^===END$/    { intask=0; next }
    intask        { n++; if (n >= nx && $0 !~ /^#removed# /) print > ("'"$TMP"'/mig/resume_lane_" lane ".cmds") }
' "$TMP/mig/lanes.txt"
[ "$(wc -l < "$TMP/mig/resume_lane_0.cmds" 2>/dev/null)" = "2" ] \
  && ok "resume set includes the interrupted task and the rest" \
  || bad "resume set wrong size"
grep -q '^DV3_TASK_NAME=t3 ' "$TMP/mig/resume_lane_0.cmds" 2>/dev/null \
  && ok "the interrupted task is re-queued, not skipped" \
  || bad "interrupted task missing from resume set"

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
    # Loose files in a SHARED folder -- the real probe/E4 shape (lewm_graft,
    # sigma_ladder). Exercises output-parent creation and logdir != run_id.
    logdir: "sc_probe_out"
    outputs: ["sc_probe_out/${arm}.json"]
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
# A pull runs against instances that are still COMPUTING, so rsync exit 24
# ("some files vanished") is routine. Under set -e it aborted the whole pull
# after the first live instance and silently skipped every later one
# (2026-08-15: `pull 1 2 3 7` stopped after 2). 23 must still fail.
grep -q 'rc" -eq 24' "$G/pull_results.sh" 2>/dev/null \
  && ok "pull tolerates rsync 24 (files rotated mid-copy)" \
  || bad "pull treats rsync 24 as fatal"
grep -q 'PULL INCOMPLETE for instance' "$G/pull_results.sh" 2>/dev/null \
  && ok "pull reports which instances it could not finish" \
  || bad "pull can skip instances silently"
# $RUNROOT must survive generation unexpanded: RCC and instance paths differ.
grep -q '\$RUNROOT' "$G"/lane_fit_0.cmds 2>/dev/null \
  && ok "\$RUNROOT left for the remote shell" || bad "\$RUNROOT was expanded at generation time"

# Every generated queue line must PARSE as bash. `DV3_TASK_NAME=x ( ... )` is a
# syntax error -- an assignment prefix may precede a simple command, never a
# subshell -- and bash rejects it before running anything, so the task fails in
# 0s with rc=2 and no output of its own. That shipped in the per_gpu=1 path and
# killed all 16 ridge tasks on instance 7 (2026-08-15). Parsing is the check
# that catches it; reading the line does not.
syntax_bad=0
for f in "$G"/lane_*.cmds; do
  bash -n "$f" 2>/dev/null || { syntax_bad=1; echo "    unparseable: $(basename "$f")"; }
done
[ "$syntax_bad" -eq 0 ] \
  && ok "every generated queue line parses as bash" \
  || bad "a generated queue line is not valid bash"
# ...and specifically for the single-run (per_gpu=1) shape, which is the one
# that regressed. sc_probe is per_gpu 1.
bash -n "$G"/lane_probe_*.cmds 2>/dev/null \
  && ok "single-run queue line parses (per_gpu=1 shape)" \
  || bad "single-run queue line does not parse"

# A stage whose outputs are loose files in a shared folder must have that
# folder created first. Otherwise the run does its entire GPU pass and dies on
# the final open() -- the most expensive possible place to fail. sc_probe
# writes into a shared folder; the fit stage writes run dirs and must NOT have
# its run dir pre-created (that can look like a resume to a driver).
grep -q 'mkdir -p "$RUNROOT/sc_probe_out"' "$G"/lane_probe_*.cmds 2>/dev/null \
  && ok "shared-folder output dir is created before the run" \
  || bad "shared-folder output dir is not created (probe would fail at write)"
grep -q 'mkdir -p "$RUNROOT/sc_fit_a_seed1"' "$G"/lane_fit_0.cmds 2>/dev/null \
  && bad "run dir pre-created (can read as a resume)" \
  || ok "per-run output dirs are left to the driver"

# submit.sh addresses TWO machines with different runroots. The wave dir must
# land where the instance's queue will look for it, so the path is resolved by
# asking the instance -- not by expanding this machine's $RUNROOT (which rsyncs
# to a nonexistent path) and not by leaving $REMOTE for the instance (which is
# unset there, so `set -u` kills every queue line after the rsync "succeeded").
grep -q 'REMOTE_ROOT="\$(dv3_ssh_dv3 "\$N"' "$G/submit.sh" 2>/dev/null \
  && ok "submit.sh resolves the instance's own RUNROOT" \
  || bad "submit.sh does not ask the instance for its RUNROOT"
grep -q "dv3_queue_or_add \"\\\$REMOTE/" "$G/submit.sh" 2>/dev/null \
  && bad "submit.sh leaves \$REMOTE for the instance to expand (unset there)" \
  || ok "queue lines expand \$REMOTE locally"
grep -qE 'dv3_ssh_dv3 "\$N" "DV3_ABORT_ON_FAIL=[01] dv3_queue_or_add' "$G/submit.sh" 2>/dev/null \
  && ok "queue line quoting is locally-expanding" || bad "queue line quoting"

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

# ckpt_step must not read a training step out of a bare timestamp. Adapt runs
# write `<timestamp>` checkpoints with no step suffix, and a trailing-digit
# match turns the microsecond field into a six-digit pseudo-step -- which the
# outlier check then compares across runs and flags as disagreement it invented
# (2026-08-15, sigma-ladder: "ckpt_step=669262 vs modal 649186" on two complete
# runs). Fits, which DO carry `-<step>`, must still parse.
CS="$TMP/ckptstep"; mkdir -p "$CS"
mk_ckpt () { mkdir -p "$CS/$1/ckpt/$2"; touch "$CS/$1/ckpt/$2/done"; }
mk_ckpt bare_ts   20260815T041132F669262
mk_ckpt with_step 20260815T035522F111998-000000500000
mk_ckpt bare_num  000000450000
step_of () { python3 -c "
import sys; sys.path.insert(0, '$ROOT/scripts/ops')
from pathlib import Path
import wavecheck; print(wavecheck.ckpt_step(Path('$CS/$1')))"; }
[ "$(step_of bare_ts)" = "0" ] \
  && ok "ckpt_step ignores a bare timestamp (no step in the name)" \
  || bad "ckpt_step read a step out of a timestamp: got $(step_of bare_ts)"
[ "$(step_of with_step)" = "500000" ] \
  && ok "ckpt_step parses <timestamp>-<step>" \
  || bad "ckpt_step parses <timestamp>-<step>: got $(step_of with_step)"
[ "$(step_of bare_num)" = "450000" ] \
  && ok "ckpt_step still parses a plain step-named ckpt" \
  || bad "ckpt_step still parses a plain step-named ckpt: got $(step_of bare_num)"

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
# THE REAL PIPELINE SHAPE. dv3_status --json ends with a newline, so a producer
# that interleaves printf with it splits every record across two lines. The
# renderer was only ever tested on hand-written single-line input, so it looked
# fine while dropping every instance in production.
split="$(printf '{"instance":"1","state":{"gpus":[],"lanes":[]}\n}\n{"instance":"2","state":{"gpus":[],"lanes":[{"lane":"0","alive":true,"next":2,"total":5,"pending":4,"running_task":"x"}]}\n}\n' \
  | python3 "$ROOT/scripts/ops/board.py" --durations /dev/null 2>&1)"
case "$split" in *"IDLE and billing: instance(s) 1"*) ok "board parses records split across lines" ;;
                 *) bad "board drops newline-split records (got: $split)" ;; esac
case "$split" in *"2      0     RUNNING"*) ok "board renders a lane from split input" ;;
                 *) bad "board lane from split input" ;; esac
# And the producer must not emit that shape in the first place.
chk "board-raw captures before printing (one record per line)" \
    "grep -q 'st=\"\$(dv3_ssh_dv3' '$ROOT/scripts/ops/dv3ops'"
chk "board-raw no longer interleaves printf with the ssh call" \
    "! grep -q \"printf '{\\\"instance\\\":\\\"%s\\\",\\\"state\\\":' \" '$ROOT/scripts/ops/dv3ops'"

echo
echo "== P3: failure classification =="
chk "signature table parses" "python3 '$ROOT/scripts/ops/triage.py' signatures"
class_of () { printf '%s' "$1" > "$TMP/f.log"
  python3 "$ROOT/scripts/ops/triage.py" classify "$TMP/f.log" 2>/dev/null | awk '{print $1}'; }
[ "$(class_of 'RESOURCE_EXHAUSTED: Out of memory allocating 2147483648 bytes')" = oom ] \
  && ok "classifies jax OOM" || bad "classifies jax OOM"
[ "$(class_of 'OSError: [Errno 28] No space left on device')" = disk_full ] \
  && ok "classifies a full disk" || bad "classifies a full disk"
[ "$(class_of 'ERROR: no done ckpt under /r/lewm_distill_finger_s0_seed1/ckpt')" = missing_donor ] \
  && ok "classifies a missing donor" || bad "classifies a missing donor"
[ "$(class_of 'no manifest.json beside /r/axis1_finger/pxq1m/side0')" = missing_replay_manifest ] \
  && ok "classifies the replay parent-manifest trap" || bad "classifies replay manifest trap"
# The rule the whole night design rests on: never invent a cause.
[ "$(class_of 'some entirely novel explosion nobody has seen before')" = unknown ] \
  && ok "refuses to guess on an unmatched failure" || bad "guessed on an unmatched failure"
# Instance-side bash classifier must agree with the RCC-side python one.
bcls="$(dv3_classify 'RESOURCE_EXHAUSTED: Out of memory allocating 1 bytes' | cut -f1)"
[ "$bcls" = oom ] && ok "instance-side bash classifier agrees with python" \
                  || bad "bash/python classifiers disagree (bash said $bcls)"

echo
echo "== P3: digest =="
NOWTS="$(date +%s)"
python3 - "$NOWTS" > "$TMP/night.jsonl" <<'NIGHT'
import json, sys
now = int(sys.argv[1])
def ev(dt, host, kind, title, **kw):
    return json.dumps({"ts": now-dt, "host": host, "kind": kind, "title": title,
                       "priority": "high", "message": "", **kw})
print("\n".join([
  ev(7*3600, "inst4", "task_fail", "FAIL", **{"class": "oom", "retry": "other"}),
  ev(7*3600, "inst4", "queue_abort", "lane 2 ABORTED", lane="2", stranded=12),
  ev(5*3600, "inst1", "supervisor_restarted", "restarted", lane="0"),
  ev(4*3600, "inst4", "instance_idle", "idle", idle_s=7320, gpus=4),
  ev(3*3600, "inst4", "instance_idle", "idle", idle_s=9000, gpus=4),
]))
NIGHT
dg="$(python3 "$ROOT/scripts/ops/digest.py" --since 16 < "$TMP/night.jsonl" 2>&1)"
case "$dg" in *"NEEDS YOU"*) ok "digest separates decisions from noise" ;;
              *) bad "digest NEEDS YOU section" ;; esac
case "$dg" in *"instance_idle (x2"*) ok "digest collapses repeat reminders" ;;
              *) bad "digest collapses repeats" ;; esac
case "$dg" in *"SELF-HEALED"*) ok "digest reports what fixed itself" ;;
              *) bad "digest self-healed section" ;; esac
case "$dg" in *"GPU-hours burned"*) ok "digest reports the idle-hours KPI" ;;
              *) bad "digest KPI" ;; esac
case "$dg" in *"lower bound"*) ok "digest states the KPI is a lower bound" ;;
              *) bad "digest KPI caveat" ;; esac
empty="$(printf '' | python3 "$ROOT/scripts/ops/digest.py" 2>&1)"
case "$empty" in *"Silence is only good news"*) ok "empty digest questions the silence" ;;
                 *) bad "empty digest should not read as all-clear" ;; esac

echo
echo "== P3: night auto-mitigation is narrow =="
chk "auto-restart is off by default" \
    "grep -q 'DV3_NIGHT_AUTO:-0' '$ROOT/scripts/ops/watchdog.sh'"
chk "auto-restart refuses while the child still runs" \
    "grep -q 'task still running' '$ROOT/scripts/ops/watchdog.sh'"
chk "auto-restart refuses an aborted or cancelled queue" \
    "grep -q 'aborted' '$ROOT/scripts/ops/watchdog.sh' && grep -q 'cancelled' '$ROOT/scripts/ops/watchdog.sh'"
chk "auto-restart gives up after repeated attempts" \
    "grep -q 'tries.*-ge 2' '$ROOT/scripts/ops/watchdog.sh'"
chk "watchdog still mutates no queue contents" \
    "! grep -qE 'dv3_(add_cmd_tasks|queue_or_add|cancel_queue|remove_tasks)' '$ROOT/scripts/ops/watchdog.sh'"

echo
echo "== P4: archive coverage =="
AR="$TMP/arroot"; mkdir -p "$AR/run_archived" "$AR/run_bare" "$AR/_cloud_logs"
head -c 3000 /dev/urandom > "$AR/run_archived/w"; head -c 6000 /dev/urandom > "$AR/run_bare/w"
mkdir -p "$TMP/mans"; printf 'abc123  runroot/run_archived/w\n' > "$TMP/mans/bundle_x.sha256"
ar="$(python3 "$ROOT/scripts/ops/archive_report.py" --runroot "$AR" --manifests "$TMP/mans" 2>&1)"
case "$ar" in *"NO ARCHIVE"*run_bare*) ok "unarchived dirs are listed first" ;;
              *) bad "archive report NO ARCHIVE" ;; esac
case "$ar" in *ARCHIVED*run_archived*bundle_x*) ok "archived dirs cite their manifest" ;;
              *) bad "archive report ARCHIVED" ;; esac
case "$ar" in *"COMMITTED manifest"*) ok "report states a transfer is not evidence" ;;
              *) bad "archive report evidence rule" ;; esac
case "$ar" in *"delete"*"may"*) bad "report still frames things as delete-safety" ;;
              *) ok "report is coverage-framed, not delete-framed" ;; esac

echo
echo "== refresh: sticky instance indices =="
cat > "$TMP/v1.json" <<'J'
[{"id":9911,"ssh_host":"1.2.3.4","ssh_port":41771,"gpu_name":"RTX 5060 Ti","num_gpus":4,"actual_status":"running"},
 {"id":9922,"ssh_host":"5.6.7.8","ssh_port":13315,"gpu_name":"RTX 5060 Ti","num_gpus":2,"actual_status":"running"}]
J
cat > "$TMP/v2.json" <<'J'
[{"id":9922,"ssh_host":"5.6.7.8","ssh_port":13315,"gpu_name":"RTX 5060 Ti","num_gpus":2,"actual_status":"running"},
 {"id":9933,"ssh_host":"9.9.9.9","ssh_port":22222,"gpu_name":"RTX 5090","num_gpus":1,"actual_status":"running"}]
J
mkdir -p "$TMP/rst"
python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rst" --from-json "$TMP/v1.json" >/dev/null 2>&1
python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rst" --from-json "$TMP/v2.json" >/dev/null 2>&1
# Index 2 must survive a refresh: it is what you type and what alerts say.
grep -q '^export IP2=5.6.7.8$' "$TMP/rst/instances.env" \
  && ok "an instance keeps its index across refreshes" \
  || bad "index churned across refreshes"
# A destroyed instance must not hand its number to a stranger -- and not just
# for one cycle. The first implementation dropped retired ids from the file, so
# the index freed up on the NEXT refresh and the following rental inherited it.
grep -q '^export IP1=' "$TMP/rst/instances.env" \
  && bad "destroyed instance's index was reused" \
  || ok "destroyed instance's index stays reserved (one cycle)"
# Separate state dir: this sequence rewrites instances.env, and the assertions
# above (and below) still read the one built from v1/v2.
mkdir -p "$TMP/rstc"
python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rstc" --from-json "$TMP/v1.json" >/dev/null 2>&1
python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rstc" --from-json "$TMP/v2.json" >/dev/null 2>&1
python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rstc" --from-json "$TMP/v2.json" >/dev/null 2>&1
cat > "$TMP/v4.json" <<'J'
[{"id":9922,"public_ipaddr":"5.6.7.8","ports":{"22/tcp":[{"HostPort":"13315"}]},"gpu_name":"G","num_gpus":2},
 {"id":9944,"public_ipaddr":"4.4.4.4","ports":{"22/tcp":[{"HostPort":"14444"}]},"gpu_name":"G","num_gpus":1}]
J
r4="$(python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rstc" --from-json "$TMP/v4.json" 2>&1)"
case "$r4" in *"IP1"*|*" 1  "*9944*) bad "retired index recycled after two cycles" ;;
              *) ok "retired index survives repeated refreshes" ;; esac
case "$r4" in *"retired indices held"*) ok "retired indices are reported" ;;
              *) bad "retired indices not reported" ;; esac
fg="$(python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rstc" --from-json "$TMP/v4.json" --forget 9922 2>&1 || true)"
case "$fg" in *REFUSED*) ok "refuses to forget a still-rented instance" ;;
              *) bad "should refuse to forget a live instance" ;; esac
grep -q '^export IP3=9.9.9.9$' "$TMP/rst/instances.env" \
  && ok "a new instance takes the next free index" || bad "new instance indexing"
# Real field names, confirmed against a live instance 2026-08-14. The proxy
# host (ssh_host) is present too: host and port must be taken as a PAIR, or a
# proxy hostname gets glued to a direct port and the address silently does not
# answer -- which looks like a network fault, not a config bug.
cat > "$TMP/v_real.json" <<'J'
[{"id":9911,"public_ipaddr":"113.177.120.190","ssh_host":"ssh3.vast.ai","ssh_port":17836,
  "ports":{"22/tcp":[{"HostIp":"0.0.0.0","HostPort":"13124"},{"HostIp":"::","HostPort":"13124"}],
           "8080/tcp":[{"HostIp":"0.0.0.0","HostPort":"13339"}]},
  "gpu_name":"RTX 5060 Ti","num_gpus":4,"actual_status":"running",
  "search":{"totalHour":0.3526851851851852}}]
J
mkdir -p "$TMP/rst2"
rout="$(python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rst2" --from-json "$TMP/v_real.json" 2>&1)"
# The direct port lives in ports["22/tcp"], NOT in ssh_port -- ssh_port belongs
# to the proxy host. Gluing public_ipaddr to ssh_port yields a valid-looking
# address nothing listens on.
case "$rout" in *"113.177.120.190:13124"*) ok "resolves the DIRECT ssh port from ports.22/tcp" ;;
                *) bad "direct port resolution (got: $rout)" ;; esac
case "$rout" in *":17836"*) bad "glued the proxy port onto the direct IP" ;;
                *) ok "never pairs the direct IP with the proxy port" ;; esac
case "$rout" in *ssh3.vast.ai*) bad "picked the proxy host over the direct IP" ;;
                *) ok "prefers the direct IP over the proxy host" ;; esac
case "$rout" in *"direct(public_ipaddr+ports.22/tcp)"*) ok "reports which route it used" ;;
                *) bad "route not reported" ;; esac
case "$rout" in *0.353*) ok "captures real \$/hr for the dollar threshold" ;;
                *) bad "cost not captured" ;; esac
# A proxy-only instance (no ports block) must still resolve, via the proxy.
printf '[{"id":42,"ssh_host":"ssh3.vast.ai","ssh_port":17836,"gpu_name":"X","num_gpus":1}]\n' \
  > "$TMP/v_proxy.json"
pout="$(python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rst2" --from-json "$TMP/v_proxy.json" 2>&1)"
case "$pout" in *"proxy(ssh_host+ssh_port)"*) ok "proxy-only instance falls back to the proxy route" ;;
                *) bad "proxy fallback (got: $pout)" ;; esac
# Two machines each running refresh produce two numberings that both look
# right. Warn before inventing a second one.
mkdir -p "$TMP/fresh"
g="$(python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/fresh" --from-json "$TMP/v1.json" --dry-run 2>&1)"
case "$g" in *"no index history here"*) ok "warns when a second machine would renumber a live fleet" ;;
             *) bad "no second-machine warning" ;; esac
chk "push-to sends the authority file, not just the derived env" \
    "grep -q 'instances.json\", str(state / \"instances.env' '$ROOT/scripts/ops/refresh.py' \
     || grep -q 'instances.json + instances.env' '$ROOT/scripts/ops/refresh.py'"

# Single-writer: two machines each running refresh produce two numberings that
# both look right. The file records its owner and refuses a second one.
mkdir -p "$TMP/auth"
python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/auth" --from-json "$TMP/v_real.json" >/dev/null 2>&1
python3 - "$TMP/auth/instances.json" <<'PYX'
import json, sys, pathlib
f = pathlib.Path(sys.argv[1]); d = json.loads(f.read_text())
d["authority_host"] = "some-other-machine"; f.write_text(json.dumps(d))
PYX
a1="$(python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/auth" --from-json "$TMP/v_real.json" 2>&1 || true)"
case "$a1" in *REFUSING*) ok "refresh refuses to write an inventory owned elsewhere" ;;
              *) bad "second writer not blocked" ;; esac
a2="$(python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/auth" --from-json "$TMP/v_real.json" --take-authority 2>&1 || true)"
case "$a2" in *"authority moved"*) ok "ownership can be taken deliberately" ;;
              *) bad "--take-authority did not transfer ownership" ;; esac

# Vast has changed its JSON shape before; an unusable payload must name the keys.
printf '[{"weird":1,"other":2}]\n' > "$TMP/v3.json"
out="$(python3 "$ROOT/scripts/ops/refresh.py" --state "$TMP/rst" --from-json "$TMP/v3.json" 2>&1 || true)"
case "$out" in *"Keys seen"*) ok "unrecognised vast JSON reports the keys it saw" ;;
               *) bad "unrecognised vast JSON should name the keys" ;; esac

echo
echo "== destroy: the archive gate =="
# Destroying a rented instance deletes every run dir on it. The gate is the
# whole point of the verb existing, so assert its shape rather than its prose.
D="$ROOT/scripts/ops/destroy_instance.sh"
chk "destroy is report-only without --yes" \
    "grep -q 'REPORT ONLY -- nothing destroyed' '$D'"
chk "destroy refuses on unarchived data" \
    "grep -q 'refusing to destroy: unarchived data' '$D'"
chk "destroy refuses while work is in flight" \
    "grep -q 'refusing to destroy an instance with work in flight' '$D'"
chk "destroy requires typing the vast id" \
    "grep -q 'Type the vast id' '$D'"
chk "unsafe override needs a typed acknowledgement" \
    "grep -q 'DESTROY UNARCHIVED' '$D'"
chk "destroy records the instance BEFORE calling vastai" \
    "awk '/destroyed.jsonl/{r=NR} /vastai destroy instance/{d=NR} END{exit !(r && d && r<d)}' '$D'"
chk "destroy retires the index afterwards" \
    "grep -q 'refresh.py' '$D'"
# Every bare ssh in destroy must use -n. Without it the inventory call eats the
# script's stdin and the confirmation prompt reads an empty stream, aborting a
# gate-passing destroy with "got ''" (2026-08-15) -- or swallowing the user's
# keystrokes on a terminal.
chk "destroy's RCC ssh cannot eat the confirmation prompt" \
    "grep -q 'RCC_SSH=(-n ' '$D'"
# There is deliberately no create verb: picking an offer is a price judgement.
chk "no create/rent verb exists" \
    "! grep -qE '^  (create|rent|launch)\)' '$ROOT/scripts/ops/dv3ops'"

# The inventory is the evidence the gate runs on. Counters, not names.
INVP="$ROOT/scripts/ops/runroot_inventory.py"
IR="$TMP/invroot"; mkdir -p "$IR/keep_me/ckpt/20260815T035522F111998-000000500000" \
                            "$IR/_cloud_logs" "$IR/_queue_control"
touch "$IR/keep_me/ckpt/20260815T035522F111998-000000500000/done"
seq 1 42 > "$IR/keep_me/scores.jsonl"
touch "$IR/keep_me/ADAPT_DONE"
inv="$(python3 "$INVP" "$IR" 2>/dev/null)"
python3 - "$inv" <<'PY' && ok "inventory reports counters and skips scaffolding" \
                        || bad "inventory output wrong"
import json, sys
d = json.loads(sys.argv[1])["entries"]
assert "keep_me" in d, d
assert "_cloud_logs" not in d and "_queue_control" not in d, d
e = d["keep_me"]
assert e["ckpt_step"] == 500000, e
assert e["n_scores"] == 42, e
assert "ADAPT_DONE" in e["markers"], e
assert e["n_files"] >= 3 and e["bytes"] > 0, e
PY

echo
echo "== every advertised verb dispatches =="
missing=""
for v in push-repo versions status board-raw sh watch events migrate selfcheck \
         gen preflight submit check requeue pull bundle pull-local \
         durations refresh board pack triage digest archive-report \
         rcc-connect destroy; do
  # rcc-connect with no args would try to OPEN an ssh connection; a selfcheck
  # must never reach the network or prompt for 2FA. --status only reports.
  vargs=""; [ "$v" = rcc-connect ] && vargs="--status"
  o="$(bash "$ROOT/scripts/ops/dv3ops" "$v" $vargs 2>&1 </dev/null | head -1)"
  case "$o" in *"not built yet"*|*"unknown verb"*) missing="$missing $v" ;; esac
done
[ -z "$missing" ] && ok "all 26 verbs dispatch" || bad "verbs not dispatching:$missing"
# `help` must not advertise anything that does not exist.
bash "$ROOT/scripts/ops/dv3ops" help 2>&1 | grep -q "not built yet" \
  && bad "help still lists unbuilt verbs" || ok "help lists no unbuilt verbs"

echo
echo "--------------------------------------------------------------"
printf 'PASS %d   FAIL %d\n' "$pass" "$fail"
[ "$fail" -eq 0 ] && echo "SELFCHECK PASS" || echo "SELFCHECK FAIL"
exit "$([ "$fail" -eq 0 ] && echo 0 || echo 1)"
