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
gen_out="$(bash "$ROOT/scripts/ops/dv3ops" gen 2>&1 || true)"
case "$gen_out" in *P1*) ok "unbuilt verbs name their phase" ;;
                   *) bad "unbuilt verbs name their phase (got: $gen_out)" ;; esac
chk "unknown verb fails" "! bash '$ROOT/scripts/ops/dv3ops' bogusverb"

echo
echo "--------------------------------------------------------------"
printf 'PASS %d   FAIL %d\n' "$pass" "$fail"
[ "$fail" -eq 0 ] && echo "SELFCHECK PASS" || echo "SELFCHECK FAIL"
exit "$([ "$fail" -eq 0 ] && echo 0 || echo 1)"
