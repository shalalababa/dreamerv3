#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a /workspace/logs/startup.log) 2>&1
echo "[$(date)] startup begin"

mkdir -p /workspace/dreamerv3 /workspace/dreamerv3_runs/_cloud_logs \
  /workspace/dreamerv3_runs/_queue_control /root/.ssh
chmod 700 /workspace/dreamerv3_runs/_queue_control 2>/dev/null || true

[ -x /venv/main/bin/python ] || { echo "ERROR: /venv/main/bin/python missing"; exit 1; }

chown root:root /root /root/.ssh
chmod 700 /root /root/.ssh
k=/root/.ssh/authorized_keys
[ -f "$k" ] && { chown root:root "$k"; chmod 600 "$k"; } || true

cat > /root/.dreamer_vast_env <<'EOF'
export REPO=/workspace/dreamerv3
export RUNROOT=/workspace/dreamerv3_runs
export CONDA_ENV=/venv/main
export MANIFEST=$RUNROOT/runs.csv

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export PATH=$CONDA_ENV/bin:/opt/instance-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

source "$CONDA_ENV/bin/activate" >/dev/null 2>&1 || true
EOF

cat > /root/dreamer_instance_helpers.sh <<'EOF'
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

dv3_status () {
  dv3_init || return 1
  local lane gpu active qdir nxt total pid dead found=0
  echo "== host =="; hostname
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

dv3_tail () {
  dv3_init || return 1
  tail -n "${2:-40}" "$RUNROOT/_cloud_logs/worker_shard${1:-0}.out" 2>/dev/null || true
}

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
  local idx="${1:?index}" cmd="${2:?command}" name=""
  if [[ "$cmd" =~ (^|[[:space:]])DV3_TASK_NAME=([^[:space:]]+) ]]; then
    name="${BASH_REMATCH[2]}"
  elif [[ "$cmd" =~ (^|[[:space:]])RUN_ID=([^[:space:]]+) ]]; then
    name="${BASH_REMATCH[2]}"
  fi
  name="$(printf '%s' "$name" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_*//;s/_*$//')"
  [ -n "$name" ] || name="$(printf 'cmd_%04d' "$idx")"
  printf '%s\n' "$name"
}

dv3_wait_gpu_idle () {
  local gpu="${1:-0}" pids last=0 now
  [ "$gpu" = "none" ] && return 0
  while :; do
    pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' \t\r' | awk 'NF' | paste -sd, -)"
    [ -z "$pids" ] && break
    now="$(date +%s)"
    if [ $((now - last)) -ge 1800 ]; then
      echo "[$(date)] WAIT gpu${gpu}: $pids"
      last="$now"
    fi
    sleep 60
  done
}

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
  local name log child rc vis
  dv3_init || return 1
  name="$(dv3_task_name "$idx" "$cmd")"
  log="$RUNROOT/_cloud_logs/${name}.out"
  printf '\n===== [%s] START %s gpu=%s =====\n' "$(date)" "$name" "$gpu" >> "$log"
  echo "[$(date)] START $name gpu=$gpu log=$log"
  vis="$gpu"; [ "$gpu" = "none" ] && vis=""
  CUDA_VISIBLE_DEVICES="$vis" setsid bash -c "source '$DV3_ENV_FILE' >/dev/null 2>&1 || true; $cmd" >> "$log" 2>&1 &
  child="$!"
  printf '%s\n' "$child" > "$qdir/child.pid"
  wait "$child"; rc="$?"
  rm -f "$qdir/child.pid"
  if [ "$rc" -eq 0 ]; then
    echo "[$(date)] DONE $name log=$log"
  else
    echo "[$(date)] FAIL $name rc=$rc log=$log"
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

dv3_add_cmd_tasks () {
  local task_file="${1:?command file}" gpu="${2:?gpu lane}" qc active qdir tmp count shard
  dv3_init || return 1
  dv3_lane_ok "$gpu" || return $?
  qc="$(dv3_qc "$gpu")" || return 1
  (
    flock -w 30 9 || { echo "ERROR: lane $gpu lock busy"; exit 1; }
    dv3_clear_stale_locked "$qc"
    [ -f "$qc/active_queue" ] || { echo "ERROR: no active queue lane $gpu"; exit 1; }
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
  dv3_add_cmd_tasks "${1:?command file}" "${2:?gpu lane}" || {
    echo "queue_or_add: nothing to append to; starting a new queue on lane $2"
    dv3_queue_cmds_list "$1" "$2" "${3:-}"
  }
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
  qc="$(dv3_qc "$gpu")" || exit 1
  qdir="$qc/$qid"
  shard="$(cat "$qdir/shard" 2>/dev/null || echo "$gpu")"
  abort="$(cat "$qdir/abort_on_fail" 2>/dev/null || echo 1)"
  grace="$(cat "$qdir/idle_grace_seconds" 2>/dev/null || echo 300)"
  dv3_validate_args "$gpu" "$shard" "$abort" "$grace" || exit 2
  touch "$qdir/supervisor_started"
  echo "[$(date)] QUEUE_START id=$qid lane=$gpu shard=$shard tasks=$(wc -l < "$qdir/tasks.txt")"
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
        dv3_abort_queue "$qc" "$qdir" "$qid" || true
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
EOF

chmod 644 /root/dreamer_instance_helpers.sh /root/.dreamer_vast_env

grep -q dreamer_vast_env /root/.bashrc 2>/dev/null ||
  printf 'source /root/.dreamer_vast_env\nsource /root/dreamer_instance_helpers.sh\n' >> /root/.bashrc

echo "[$(date)] startup done"
