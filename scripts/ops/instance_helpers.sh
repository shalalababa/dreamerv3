# ---------------------------------------------------------------------------
# Dreamer Vast instance helpers  (dv3ops P0, 2026-08-14)
#
# This file used to live inside temp_files/../template_onstart_script.sh as a
# heredoc, bounded by Vast's 16384-char onstart cap. It now lives in the repo
# and is delivered by `dv3ops push-repo`; /root/dreamer_instance_helpers.sh is
# a ~10-line shim that sources this file.
#
# CORE QUEUE SEMANTICS ARE UNCHANGED from the v1 template (2026-07-26):
# claim/advance/abort, per-lane flock, command snapshots, "#removed#" marking,
# abort-on-first-failure default, lane == GPU, pair barrier for co-location.
# Do not "simplify" them; they are the tested part.
#
# ADDED IN P0 (previously priced out by the char cap):
#   dv3_event / dv3_notify   structured event log + ntfy push
#   dv3_status --json        machine-readable state for board + watchdog
#   dv3_wait_gpu_idle        ceiling + escalation (was: silent infinite wait)
#   util sampler             per-task VRAM/util sidecar -> co-location capacity
#   dv3_version              helper provenance stamp vs repo HEAD
#   dv3_queue_or_add         distinguishes "no active queue" from "lock busy"
#   dv3_watch                the nvidia-smi + shard-log loop, no longer pasted
#
# NOTE: sourced by interactive shells. Never `set -e` here.
# ---------------------------------------------------------------------------

DV3_ENV_FILE="${DV3_ENV_FILE:-/root/.dreamer_vast_env}"
DV3_HELPERS_FILE="${DV3_HELPERS_FILE:-/root/dreamer_instance_helpers.sh}"

dv3_init () {
  source "$DV3_ENV_FILE" >/dev/null 2>&1 || true
  if [ -z "${REPO:-}" ] || [ -z "${RUNROOT:-}" ]; then
    echo "ERROR: REPO/RUNROOT unset; missing $DV3_ENV_FILE?" >&2
    return 1
  fi
  cd "$REPO" >/dev/null 2>&1 || { echo "ERROR: cannot cd to REPO=$REPO" >&2; return 1; }
  mkdir -p "$RUNROOT/_cloud_logs" "$RUNROOT/_queue_control"
  chmod 700 "$RUNROOT/_queue_control" 2>/dev/null || true
}

dv3_qc () {
  local gpu="${1:?gpu lane}" lane
  dv3_init || return 1
  lane="$RUNROOT/_queue_control/lane_${gpu}"
  mkdir -p "$lane"
  printf '%s\n' "$lane"
}

dv3_alive () { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

dv3_lane_ok () {
  [[ "${1:-}" =~ ^([0-9]+|none)$ ]] || { echo "ERROR: bad lane: ${1:-}" >&2; return 2; }
}

# --- provenance -------------------------------------------------------------
# push-repo writes $REPO/scripts/ops/VERSION (the .git dir is excluded from the
# rsync, so the instance cannot derive its own SHA).

dv3_version () {
  local f="${REPO:-/workspace/dreamerv3}/scripts/ops/VERSION"
  if [ -r "$f" ]; then cat "$f"; else echo "UNKNOWN (no VERSION stamp; push-repo predates dv3ops)"; fi
}

# --- events + notification --------------------------------------------------
# Every transition is emitted at its source (reliable) and appended to
# events.jsonl (the digest's input; ntfy retention is not guaranteed).
# Notification failure must never break a queue: everything is best-effort.

dv3_jesc () {
  printf '%s' "${1:-}" |
    sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' |
    tr -d '\r' |
    sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/\\n/g'
}

dv3_notify () {
  # args: priority title message [tags]
  local pri="${1:-default}" title="${2:-dv3}" msg="${3:-}" tags="${4:-}"
  [ -n "${DV3_NTFY_TOPIC:-}" ] || return 0
  command -v curl >/dev/null 2>&1 || return 0
  local url="${DV3_NTFY_URL:-https://ntfy.sh}/${DV3_NTFY_TOPIC}"
  local host="${DV3_INSTANCE_LABEL:-$(hostname 2>/dev/null || echo instance)}"
  local -a hdr=(-H "Title: [$host] $title" -H "Priority: $pri")
  [ -n "$tags" ] && hdr+=(-H "Tags: $tags")
  curl -fsS -m 10 "${hdr[@]}" --data-binary "$msg" "$url" >/dev/null 2>&1 || true
}

dv3_event () {
  # args: kind priority title message  [extra_json_fields]
  local kind="${1:?kind}" pri="${2:-default}" title="${3:-}" msg="${4:-}" extra="${5:-}"
  local host="${DV3_INSTANCE_LABEL:-$(hostname 2>/dev/null || echo instance)}"
  local log="${RUNROOT:-/workspace/dreamerv3_runs}/_cloud_logs/events.jsonl"
  mkdir -p "$(dirname "$log")" 2>/dev/null || true
  printf '{"ts":%s,"iso":"%s","host":"%s","kind":"%s","priority":"%s","title":"%s","message":"%s"%s}\n' \
    "$(date +%s)" "$(date -Is 2>/dev/null || date)" \
    "$(dv3_jesc "$host")" "$(dv3_jesc "$kind")" "$(dv3_jesc "$pri")" \
    "$(dv3_jesc "$title")" "$(dv3_jesc "$msg")" \
    "${extra:+,$extra}" >> "$log" 2>/dev/null || true
  dv3_notify "$pri" "$title" "$msg" "${DV3_NTFY_TAGS:-}"
}

# --- stale queue handling ---------------------------------------------------

dv3_clear_stale_locked () {
  local qc="$1" active qdir pid cpid
  [ -f "$qc/active_queue" ] || return 0
  active="$(cat "$qc/active_queue" 2>/dev/null || true)"
  qdir="$qc/$active"
  pid="$(cat "$qdir/supervisor.pid" 2>/dev/null || true)"
  if [ -n "$active" ] && [ -d "$qdir" ] && dv3_alive "$pid"; then
    return 0
  fi
  cpid="$(cat "$qdir/child.pid" 2>/dev/null || true)"
  dv3_alive "$cpid" && echo "WARNING: stale ${active:-?} child pid=$cpid still running"
  echo "[$(date)] clearing stale queue: ${active:-unknown}"
  rm -f "$qc/active_queue"
}

# --- status -----------------------------------------------------------------

dv3_status () {
  [ "${1:-}" = "--json" ] && { dv3_status_json; return $?; }
  dv3_init || return 1
  local lane gpu active qdir nxt total pid dead found=0
  echo "== host =="; hostname
  echo "helpers: $(dv3_version)"
  echo; echo "== gpu =="
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv 2>/dev/null || nvidia-smi || true
  echo; echo "== procs =="
  ps -eo pid,etime,args |
    grep -E 'dv3_queue_supervisor|axis1\.sbatch|dreamerv3/main.py|python -m' |
    grep -v grep || true
  echo; echo "== queues =="
  for lane in "$RUNROOT/_queue_control"/lane_*; do
    [ -f "$lane/active_queue" ] || continue
    found=1
    gpu="${lane##*/lane_}"
    active="$(cat "$lane/active_queue" 2>/dev/null || true)"
    qdir="$lane/$active"
    nxt="$(cat "$qdir/next_index" 2>/dev/null || echo 1)"
    total="$(wc -l < "$qdir/tasks.txt" 2>/dev/null || echo 0)"
    pid="$(cat "$qdir/supervisor.pid" 2>/dev/null || true)"
    dead=""; dv3_alive "$pid" || dead=" (DEAD)"
    echo "-- lane $gpu --"
    echo "id=$active shard=$(cat "$qdir/shard" 2>/dev/null) next=$nxt total=$total supervisor=${pid:-none}$dead"
    echo "abort=$(cat "$qdir/abort_on_fail" 2>/dev/null || echo 1) grace=$(cat "$qdir/idle_grace_seconds" 2>/dev/null || echo 300)"
    awk -v nxt="$nxt" '{
      s = (NR < nxt ? "done" : (NR == nxt ? "next" : "queued"))
      printf "%4d %-6s %s\n", NR, s, $0
    }' "$qdir/tasks.txt" 2>/dev/null || true
    echo
  done
  [ "$found" -eq 1 ] || echo "none"
}

dv3_status_json () {
  dv3_init || return 1
  local lane gpu active qdir nxt total run pid alive first=1
  printf '{"ts":%s,"host":"%s","helpers":"%s","gpus":[' \
    "$(date +%s)" "$(dv3_jesc "$(hostname 2>/dev/null)")" "$(dv3_jesc "$(dv3_version)")"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' 'NR>1{printf ","} {printf "{\"index\":%s,\"name\":\"%s\",\"util\":%s,\"mem_used\":%s,\"mem_total\":%s}", $1,$2,$3,$4,$5}'
  printf '],"lanes":['
  for lane in "$RUNROOT/_queue_control"/lane_*; do
    [ -d "$lane" ] || continue
    [ -f "$lane/active_queue" ] || continue
    gpu="${lane##*/lane_}"
    active="$(cat "$lane/active_queue" 2>/dev/null || true)"
    qdir="$lane/$active"
    nxt="$(cat "$qdir/next_index" 2>/dev/null || echo 1)"
    total="$(wc -l < "$qdir/tasks.txt" 2>/dev/null || echo 0)"
    run="$(cat "$qdir/running_index" 2>/dev/null || echo 0)"
    pid="$(cat "$qdir/supervisor.pid" 2>/dev/null || true)"
    if dv3_alive "$pid"; then alive=true; else alive=false; fi
    [ "$first" -eq 1 ] || printf ','
    first=0
    printf '{"lane":"%s","queue":"%s","shard":"%s","next":%s,"total":%s,"running":%s,"supervisor":%s,"alive":%s,"abort_on_fail":%s,"pending":%s,"running_task":"%s","aborted":%s,"finished":%s}' \
      "$gpu" "$(dv3_jesc "$active")" "$(cat "$qdir/shard" 2>/dev/null || echo "$gpu")" \
      "$nxt" "$total" "$run" "${pid:-0}" "$alive" \
      "$(cat "$qdir/abort_on_fail" 2>/dev/null || echo 1)" \
      "$(( total - nxt + 1 < 0 ? 0 : total - nxt + 1 ))" \
      "$(dv3_jesc "$(dv3_task_name "$run" "$(cat "$qdir/running_task" 2>/dev/null || echo '')" 2>/dev/null)")" \
      "$([ -f "$qdir/aborted" ] && echo true || echo false)" \
      "$([ -f "$qdir/finished" ] && echo true || echo false)"
  done
  printf ']}\n'
}

dv3_tail () {
  dv3_init || return 1
  tail -n "${2:-40}" "$RUNROOT/_cloud_logs/worker_shard${1:-0}.out" 2>/dev/null || true
}

dv3_tail_task () {
  dv3_init || return 1
  tail -n "${2:-60}" "$RUNROOT/_cloud_logs/${1:?task name}.out" 2>/dev/null || true
}

# The monitoring loop that used to be pasted by hand into every tmux pane.
dv3_watch () {
  local secs="${1:-15}"
  watch -n "$secs" "
    nvidia-smi
    echo
    tail -n 3 ${RUNROOT:-/workspace/dreamerv3_runs}/_cloud_logs/worker_shard*.out 2>/dev/null
    echo
  "
}

# --- validation / snapshots -------------------------------------------------

dv3_validate_args () {
  local gpu="${1:?gpu}" shard="${2:?shard}" abort="${3:-1}" grace="${4:-300}"
  [[ "$gpu" =~ ^[0-9]+$ || "$gpu" = "none" ]] || { echo "ERROR: gpu must be integer or none"; return 2; }
  [[ "$shard" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "ERROR: bad shard"; return 2; }
  [[ "$abort" =~ ^[01]$ ]] || { echo "ERROR: bad abort flag"; return 2; }
  [[ "$grace" =~ ^[0-9]+$ ]] || { echo "ERROR: bad grace"; return 2; }
  if [ "$gpu" != "none" ]; then
    command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi missing"; return 2; }
    nvidia-smi --id="$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1 ||
      { echo "ERROR: unavailable gpu index: $gpu"; return 2; }
  fi
}

dv3_snapshot_cmds () {
  local input="${1:?command file}" output="${2:?snapshot file}" count
  [ -f "$input" ] || { echo "ERROR: no such file: $input"; return 2; }
  awk '{ sub(/\r$/, "") } NF && $0 !~ /^[[:space:]]*#/ { print }' "$input" > "$output"
  [ -s "$output" ] || { echo "ERROR: no commands in $input"; return 2; }
  count="$(wc -l < "$output")"
  echo "validated $count commands -> $output"
}

dv3_task_name () {
  local idx="${1:?index}" cmd="${2:-}" name=""
  if [[ "$cmd" =~ (^|[[:space:]])DV3_TASK_NAME=([^[:space:]]+) ]]; then
    name="${BASH_REMATCH[2]}"
  elif [[ "$cmd" =~ (^|[[:space:]])RUN_ID=([^[:space:]]+) ]]; then
    name="${BASH_REMATCH[2]}"
  fi
  name="$(printf '%s' "$name" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_*//;s/_*$//')"
  [ -n "$name" ] || name="$(printf 'cmd_%04d' "$idx")"
  printf '%s\n' "$name"
}

# --- GPU idle wait ----------------------------------------------------------
# v1 waited forever behind a leaked compute process, logging once per 30 min.
# That presented as "lane active, nothing progressing" -- a silent stall, the
# failure class this whole system exists to eliminate. Now: escalate at the
# ceiling and keep escalating, but never auto-kill somebody else's process.

dv3_wait_gpu_idle () {
  local gpu="${1:-0}" pids last=0 now start elapsed
  local ceiling="${DV3_GPU_WAIT_CEILING_SECONDS:-7200}"
  local renotify="${DV3_GPU_WAIT_RENOTIFY_SECONDS:-7200}"
  local escalated=0 last_escalation=0
  [ "$gpu" = "none" ] && return 0
  start="$(date +%s)"
  while :; do
    pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' \t\r' | awk 'NF' | paste -sd, -)"
    [ -z "$pids" ] && break
    now="$(date +%s)"
    elapsed=$(( now - start ))
    if [ $((now - last)) -ge 1800 ]; then
      echo "[$(date)] WAIT gpu${gpu}: $pids (${elapsed}s)"
      last="$now"
    fi
    if [ "$elapsed" -ge "$ceiling" ] && [ $((now - last_escalation)) -ge "$renotify" ]; then
      escalated=1
      last_escalation="$now"
      dv3_event gpu_stall high "GPU ${gpu} stalled ${elapsed}s" \
        "Lane $gpu cannot start its next task: GPU $gpu still held by pid(s) $pids after $((elapsed/60))m. Likely a leaked process from a killed task. Queue is intact and waiting; no automatic action taken." \
        "\"gpu\":\"$gpu\",\"held_by\":\"$(dv3_jesc "$pids")\",\"waited_s\":$elapsed"
    fi
    sleep 60
  done
  if [ "$escalated" -eq 1 ]; then
    dv3_event gpu_stall_cleared default "GPU ${gpu} free again" \
      "The stall on GPU $gpu cleared after $(( ($(date +%s) - start) / 60 ))m; the lane is proceeding."
  fi
}

# --- utilization sampler ----------------------------------------------------
# Answers "can this kind co-reside?" from measurement instead of luck. With a
# pair barrier the sample is the pair's combined footprint, which is exactly
# the quantity the packer needs.

dv3_sample_util () {
  local gpu="$1" out="$2" interval="${DV3_UTIL_SAMPLE_SECONDS:-60}"
  [ "$gpu" = "none" ] && return 0
  : > "$out"
  while :; do
    nvidia-smi --id="$gpu" --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null |
      awk -v t="$(date +%s)" -F', *' '{print t","$1","$2","$3}' >> "$out"
    sleep "$interval"
  done
}

dv3_util_summary () {
  local f="$1"
  [ -s "$f" ] || return 0
  awk -F, '
    { n++; us+=$2; if ($2>up) up=$2; ms+=$3; if ($3>mp) mp=$3; tot=$4 }
    END { if (n) printf "util_mean=%.0f%% util_peak=%d%% vram_mean=%dMiB vram_peak=%dMiB vram_total=%dMiB samples=%d\n",
                        us/n, up, ms/n, mp, tot, n }
  ' "$f"
}

# --- queue plumbing (unchanged core) ---------------------------------------

dv3_claim_next () {
  local qc="$1" qdir="$2" idx cmd
  (
    flock -w 30 9 || exit 1
    idx="$(cat "$qdir/next_index" 2>/dev/null || echo 1)"
    while :; do
      cmd="$(sed -n "${idx}p" "$qdir/tasks.txt" 2>/dev/null || true)"
      case "$cmd" in '#removed#'*) idx=$((idx + 1));; *) break;; esac
    done
    printf '%s\n' "$idx" > "$qdir/next_index"
    [ -n "$cmd" ] || exit 0
    printf '%s\n' "$idx" > "$qdir/running_index"
    printf '%s\n' "$cmd" > "$qdir/running_task"
    printf '%s\n' "$cmd"
  ) 9>"$qc/queue.lock"
}

dv3_advance () {
  local qc="$1" qdir="$2" idx="$3" cur
  (
    flock -w 30 9 || exit 1
    cur="$(cat "$qdir/next_index" 2>/dev/null || echo 1)"
    [ "$cur" = "$idx" ] || { echo "ERROR: index changed $idx->$cur"; exit 2; }
    printf '%s\n' "$((idx + 1))" > "$qdir/next_index"
    rm -f "$qdir/running_index" "$qdir/running_task" "$qdir/child.pid"
  ) 9>"$qc/queue.lock"
}

dv3_finish_queue () {
  local qc="$1" qdir="$2" qid="$3" active nxt total
  (
    flock -w 30 9 || exit 1
    nxt="$(cat "$qdir/next_index" 2>/dev/null || echo 1)"
    total="$(wc -l < "$qdir/tasks.txt" 2>/dev/null || echo 0)"
    [ "$nxt" -le "$total" ] && exit 3
    active="$(cat "$qc/active_queue" 2>/dev/null || true)"
    [ "$active" = "$qid" ] && rm -f "$qc/active_queue"
    touch "$qdir/finished"
  ) 9>"$qc/queue.lock"
}

dv3_abort_queue () {
  local qc="$1" qdir="$2" qid="$3" active
  (
    flock -w 30 9 || exit 1
    active="$(cat "$qc/active_queue" 2>/dev/null || true)"
    [ "$active" = "$qid" ] && rm -f "$qc/active_queue"
    rm -f "$qdir/running_index" "$qdir/running_task"
    touch "$qdir/aborted"
  ) 9>"$qc/queue.lock"
}

dv3_run_cmd () {
  local idx="${1:?index}" cmd="${2:?command}" gpu="${3:?gpu}" qdir="${4:?queue dir}"
  local name log child rc vis util sampler started elapsed summary
  dv3_init || return 1
  name="$(dv3_task_name "$idx" "$cmd")"
  log="$RUNROOT/_cloud_logs/${name}.out"
  util="$RUNROOT/_cloud_logs/${name}.util"
  printf '\n===== [%s] START %s gpu=%s =====\n' "$(date)" "$name" "$gpu" >> "$log"
  echo "[$(date)] START $name gpu=$gpu log=$log"
  started="$(date +%s)"

  sampler=""
  if [ "$gpu" != "none" ] && [ "${DV3_UTIL_SAMPLE:-1}" = "1" ]; then
    dv3_sample_util "$gpu" "$util" >/dev/null 2>&1 &
    sampler="$!"
  fi

  vis="$gpu"; [ "$gpu" = "none" ] && vis=""
  CUDA_VISIBLE_DEVICES="$vis" setsid bash -c "source '$DV3_ENV_FILE' >/dev/null 2>&1 || true; $cmd" >> "$log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" > "$qdir/child.pid"
  wait "$child"; rc="$?"
  rm -f "$qdir/child.pid"

  [ -n "$sampler" ] && { kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; }
  elapsed=$(( $(date +%s) - started ))
  summary="$(dv3_util_summary "$util")"
  printf '===== [%s] END %s rc=%s elapsed=%ss %s =====\n' \
    "$(date)" "$name" "$rc" "$elapsed" "$summary" >> "$log"

  if [ "$rc" -eq 0 ]; then
    echo "[$(date)] DONE $name log=$log elapsed=${elapsed}s ${summary}"
  else
    echo "[$(date)] FAIL $name rc=$rc log=$log elapsed=${elapsed}s ${summary}"
    dv3_event task_fail default "task FAIL: $name" \
      "$(printf 'rc=%s lane=%s after %sm\n%s\n\n--- last 12 log lines ---\n%s' \
          "$rc" "$gpu" "$((elapsed/60))" "${summary:-no util samples}" \
          "$(tail -n 12 "$log" 2>/dev/null)")" \
      "\"task\":\"$(dv3_jesc "$name")\",\"rc\":$rc,\"lane\":\"$gpu\",\"elapsed_s\":$elapsed"
  fi
  return "$rc"
}

dv3_queue_cmds_list () {
  local task_file="${1:?command file}" gpu="${2:-0}" shard="${3:-}"
  local abort="${DV3_ABORT_ON_FAIL:-1}" grace="${DV3_QUEUE_IDLE_GRACE_SECONDS:-300}"
  local qc qdir qid pid
  [ -n "$shard" ] || shard="$gpu"
  dv3_init || return 1
  dv3_validate_args "$gpu" "$shard" "$abort" "$grace" || return $?
  qc="$(dv3_qc "$gpu")" || return 1
  (
    flock -n 9 || { echo "ERROR: lane $gpu lock busy"; exit 1; }
    dv3_clear_stale_locked "$qc"
    [ ! -f "$qc/active_queue" ] || { echo "ERROR: lane $gpu already active: $(cat "$qc/active_queue")"; exit 1; }
    qdir="$(mktemp -d "$qc/queue_XXXXXXXX")" || exit 1
    qid="${qdir##*/}"
    if ! dv3_snapshot_cmds "$task_file" "$qdir/tasks.txt"; then rm -rf "$qdir"; exit 1; fi
    printf '%s\n' "$gpu" > "$qdir/gpu"
    printf '%s\n' "$shard" > "$qdir/shard"
    printf '%s\n' "$abort" > "$qdir/abort_on_fail"
    printf '%s\n' "$grace" > "$qdir/idle_grace_seconds"
    printf '1\n' > "$qdir/next_index"
    printf '%s\n' "$qid" > "$qc/active_queue"
    setsid bash -c "source '$DV3_ENV_FILE' >/dev/null 2>&1 || true; source '$DV3_HELPERS_FILE'; dv3_queue_supervisor '$gpu' '$qid'" \
      >> "$RUNROOT/_cloud_logs/worker_shard${shard}.out" 2>&1 </dev/null 9>&- &
    pid="$!"
    printf '%s\n' "$pid" > "$qdir/supervisor.pid"
    for _ in {1..15}; do [ -f "$qdir/supervisor_started" ] && break; dv3_alive "$pid" || break; sleep 1; done
    if [ ! -f "$qdir/supervisor_started" ] || ! dv3_alive "$pid"; then
      echo "ERROR: supervisor init failed lane $gpu"
      kill -TERM -- "-$pid" 2>/dev/null || true
      rm -f "$qc/active_queue"
      exit 1
    fi
    echo "launched queue id=$qid lane=$gpu shard=$shard pid=$pid"
  ) 9>"$qc/queue.lock"
}

# Exit codes are load-bearing for dv3_queue_or_add:
#   3 = no active queue on this lane (the append target does not exist)
#   1 = a real error (lock busy, bad command file, ...)
# v1 returned 1 for both, so queue_or_add treated lock contention as
# "nothing to append to" and printed a reassuring message over a real failure.
dv3_add_cmd_tasks () {
  local task_file="${1:?command file}" gpu="${2:?gpu lane}" qc active qdir tmp count shard
  dv3_init || return 1
  dv3_lane_ok "$gpu" || return $?
  qc="$(dv3_qc "$gpu")" || return 1
  (
    flock -w 30 9 || { echo "ERROR: lane $gpu lock busy"; exit 1; }
    dv3_clear_stale_locked "$qc"
    [ -f "$qc/active_queue" ] || { echo "ERROR: no active queue lane $gpu"; exit 3; }
    active="$(cat "$qc/active_queue")"
    qdir="$qc/$active"
    tmp="$(mktemp "$qdir/add_XXXXXXXX.txt")" || exit 1
    if ! dv3_snapshot_cmds "$task_file" "$tmp"; then rm -f "$tmp"; exit 1; fi
    cat "$tmp" >> "$qdir/tasks.txt"
    count="$(wc -l < "$tmp")"
    rm -f "$tmp"
    shard="$(cat "$qdir/shard" 2>/dev/null || echo "$gpu")"
    echo "[$(date)] QUEUE_ADD id=$active lane=$gpu added=$count file=$task_file" >> "$RUNROOT/_cloud_logs/worker_shard${shard}.out"
    echo "added $count commands to $active lane $gpu"
  ) 9>"$qc/queue.lock"
}

dv3_queue_or_add () {
  local task_file="${1:?command file}" gpu="${2:?gpu lane}" shard="${3:-}" rc
  dv3_add_cmd_tasks "$task_file" "$gpu"
  rc="$?"
  case "$rc" in
    0) return 0 ;;
    3) echo "queue_or_add: no active queue on lane $gpu; starting a new one"
       dv3_queue_cmds_list "$task_file" "$gpu" "$shard" ;;
    *) echo "ERROR: queue_or_add aborted on lane $gpu (append failed rc=$rc, NOT the empty-lane case)" >&2
       echo "       nothing was queued; re-check the lane with dv3_status before retrying" >&2
       return "$rc" ;;
  esac
}

dv3_remove_tasks () {
  local gpu="${1:?gpu lane}" qc qdir nxt run total i
  shift
  [ $# -gt 0 ] || { echo "usage: dv3_remove_tasks gpu idx..."; return 2; }
  dv3_lane_ok "$gpu" || return $?
  qc="$(dv3_qc "$gpu")" || return 1
  (
    flock -w 30 9 || { echo "ERROR: lane $gpu lock busy"; exit 1; }
    [ -f "$qc/active_queue" ] || { echo "ERROR: no active queue lane $gpu"; exit 1; }
    qdir="$qc/$(cat "$qc/active_queue")"
    nxt="$(cat "$qdir/next_index" 2>/dev/null || echo 1)"
    run="$(cat "$qdir/running_index" 2>/dev/null || echo 0)"
    total="$(wc -l < "$qdir/tasks.txt")"
    for i in "$@"; do
      [[ "$i" =~ ^[0-9]+$ ]] && [ "$i" -ge "$nxt" ] && [ "$i" -le "$total" ] ||
        { echo "SKIP $i: not a pending index"; continue; }
      [ "$i" = "$run" ] && { echo "SKIP $i: running; use dv3_cancel_queue"; continue; }
      sed -i "${i}s/^/#removed# /" "$qdir/tasks.txt"
      echo "removed task $i"
    done
  ) 9>"$qc/queue.lock"
}

dv3_queue_supervisor () {
  local gpu="${1:?gpu lane}" qid="${2:?queue id}" qc qdir shard abort grace cmd idx rc frc failures=0 idle_start="" now result=DONE
  local total pending
  qc="$(dv3_qc "$gpu")" || exit 1
  qdir="$qc/$qid"
  shard="$(cat "$qdir/shard" 2>/dev/null || echo "$gpu")"
  abort="$(cat "$qdir/abort_on_fail" 2>/dev/null || echo 1)"
  grace="$(cat "$qdir/idle_grace_seconds" 2>/dev/null || echo 300)"
  dv3_validate_args "$gpu" "$shard" "$abort" "$grace" || exit 2
  touch "$qdir/supervisor_started"
  total="$(wc -l < "$qdir/tasks.txt")"
  echo "[$(date)] QUEUE_START id=$qid lane=$gpu shard=$shard tasks=$total"
  while :; do
    [ -f "$qdir/cancelled" ] && { result=CANCELLED; break; }
    if ! cmd="$(dv3_claim_next "$qc" "$qdir")"; then
      echo "[$(date)] QUEUE_LOCK_FAIL id=$qid claiming"
      sleep 30
      continue
    fi
    if [ -z "$cmd" ]; then
      now="$(date +%s)"
      if [ -z "$idle_start" ]; then idle_start="$now"; echo "[$(date)] QUEUE_IDLE id=$qid grace=${grace}s"; fi
      if [ $((now - idle_start)) -ge "$grace" ]; then
        if dv3_finish_queue "$qc" "$qdir" "$qid"; then
          result=DONE
          break
        else
          frc="$?"
          if [ "$frc" -eq 3 ]; then
            echo "[$(date)] QUEUE_RESUME id=$qid late adds found"
            idle_start=""
            continue
          fi
          echo "[$(date)] QUEUE_LOCK_FAIL id=$qid finishing"
        fi
      fi
      sleep 30
      continue
    fi
    idle_start=""
    idx="$(cat "$qdir/running_index")"
    dv3_wait_gpu_idle "$gpu"
    if dv3_run_cmd "$idx" "$cmd" "$gpu" "$qdir"; then rc=0; else rc="$?"; fi
    if [ "$rc" -ne 0 ]; then
      failures=$((failures + 1))
      if [ "$abort" = "1" ]; then
        echo "[$(date)] QUEUE_ABORT id=$qid idx=$idx rc=$rc"
        total="$(wc -l < "$qdir/tasks.txt" 2>/dev/null || echo 0)"
        pending=$(( total - idx ))
        dv3_abort_queue "$qc" "$qdir" "$qid" || true
        dv3_event queue_abort high "lane $gpu ABORTED — $pending stranded" \
          "$(printf 'Task %s/%s failed (rc=%s) on lane %s; abort_on_fail=1.\nStranded: %s task(s), GPU %s now idle.\nTask: %s\n\n--- last 12 log lines ---\n%s' \
              "$idx" "$total" "$rc" "$gpu" "$pending" "$gpu" \
              "$(dv3_task_name "$idx" "$cmd")" \
              "$(tail -n 12 "$RUNROOT/_cloud_logs/$(dv3_task_name "$idx" "$cmd").out" 2>/dev/null)")" \
          "\"lane\":\"$gpu\",\"queue\":\"$(dv3_jesc "$qid")\",\"index\":$idx,\"total\":$total,\"stranded\":$pending,\"rc\":$rc"
        result=ABORTED
        break
      fi
    fi
    while :; do
      if dv3_advance "$qc" "$qdir" "$idx"; then
        break
      else
        rc="$?"
      fi
      [ "$rc" -eq 2 ] && { dv3_abort_queue "$qc" "$qdir" "$qid" || true; result=ABORTED; break; }
      echo "[$(date)] QUEUE_LOCK_FAIL id=$qid advancing"
      sleep 30
    done
    [ "$result" = ABORTED ] && break
  done
  echo "[$(date)] QUEUE_$result id=$qid lane=$gpu failures=$failures"
  case "$result" in
    DONE)
      dv3_event queue_done default "lane $gpu drained" \
        "Queue $qid finished on lane $gpu: $total task(s), $failures failure(s). GPU $gpu is now idle and billing." \
        "\"lane\":\"$gpu\",\"queue\":\"$(dv3_jesc "$qid")\",\"failures\":$failures,\"total\":$total" ;;
    # Reached only if the loop notices `cancelled` between tasks. The usual
    # path is dv3_cancel_lane KILLing this process group mid-task, which never
    # gets here -- acceptable, because cancel is user-initiated and needs no
    # alert. Do not rely on this event to detect cancellation.
    CANCELLED)
      dv3_event queue_cancelled low "lane $gpu cancelled" \
        "Queue $qid on lane $gpu was cancelled after $failures failure(s)." \
        "\"lane\":\"$gpu\",\"queue\":\"$(dv3_jesc "$qid")\"" ;;
  esac
}

dv3_cancel_queue () {
  local arg="${1:?gpu lane or all}" lane any=0
  dv3_init || return 1
  if [ "$arg" = "all" ]; then
    for lane in "$RUNROOT/_queue_control"/lane_*; do
      [ -f "$lane/active_queue" ] || continue
      any=1
      dv3_cancel_lane "${lane##*/lane_}"
    done
    [ "$any" -eq 1 ] || echo "no active queue on any lane"
    return 0
  fi
  dv3_lane_ok "$arg" || return $?
  dv3_cancel_lane "$arg"
}

dv3_cancel_lane () {
  local gpu="${1:?gpu lane}" qc active qdir spid cpid
  qc="$(dv3_qc "$gpu")" || return 1
  (
    flock -w 30 9 || { echo "ERROR: lane $gpu lock busy"; exit 1; }
    dv3_clear_stale_locked "$qc"
    [ -f "$qc/active_queue" ] || { echo "no active queue lane $gpu"; exit 0; }
    active="$(cat "$qc/active_queue")"
    qdir="$qc/$active"
    touch "$qdir/cancelled"
    rm -f "$qc/active_queue"
    spid="$(cat "$qdir/supervisor.pid" 2>/dev/null || true)"
    cpid="$(cat "$qdir/child.pid" 2>/dev/null || true)"
    [ -n "$cpid" ] && kill -TERM -- "-$cpid" 2>/dev/null || true
    [ -n "$spid" ] && kill -TERM -- "-$spid" 2>/dev/null || true
    sleep 2
    [ -n "$cpid" ] && kill -KILL -- "-$cpid" 2>/dev/null || true
    [ -n "$spid" ] && kill -KILL -- "-$spid" 2>/dev/null || true
    echo "[$(date)] cancelled $active lane $gpu"
  ) 9>"$qc/queue.lock"
}
