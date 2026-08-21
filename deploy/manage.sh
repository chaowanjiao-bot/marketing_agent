#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$DEPLOY_DIR/.." && pwd)
ENV_FILE=${ENV_FILE:-$DEPLOY_DIR/production.env}
if [[ ! -f "$ENV_FILE" ]]; then echo "Missing $ENV_FILE. Copy production.env.example first." >&2; exit 2; fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
APP_PYTHON=${APP_PYTHON:-python3}; API_HOST=${API_HOST:-127.0.0.1}; API_PORT=${API_PORT:-8000}
RUNTIME_DIR=${RUNTIME_DIR:-${TASK_ROOT%/tasks}}; PID_DIR=${PID_DIR:-$RUNTIME_DIR/pids}; LOG_DIR=${LOG_DIR:-$RUNTIME_DIR/logs}
API_PID_FILE=$PID_DIR/api.pid; WORKER_PID_FILE=$PID_DIR/worker.pid
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
prepare(){ [[ -x "$APP_PYTHON" ]] || { echo "APP_PYTHON is not executable: $APP_PYTHON" >&2; exit 2; }; mkdir -p "$TASK_ROOT" "$MODEL_OUTPUT_DIR" "$PID_DIR" "$LOG_DIR"; }
running(){ local file=$1 pattern=$2 pid; [[ -f "$file" ]] || return 1; pid=$(<"$file"); [[ "$pid" =~ ^[0-9]+$ ]] || return 1; kill -0 "$pid" 2>/dev/null || return 1; tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q "$pattern"; }
run_api(){ prepare; exec "$APP_PYTHON" -m uvicorn marketing_agent.api:app --host "$API_HOST" --port "$API_PORT"; }
run_worker(){ prepare; exec "$APP_PYTHON" -m marketing_agent.worker; }
status(){ if running "$API_PID_FILE" "uvicorn marketing_agent.api:app"; then echo "api: running pid=$(<"$API_PID_FILE")"; else echo "api: stopped"; fi; if running "$WORKER_PID_FILE" "marketing_agent.worker"; then echo "worker: running pid=$(<"$WORKER_PID_FILE")"; else echo "worker: stopped"; fi; curl -fsS --max-time 3 "http://$API_HOST:$API_PORT/health" 2>/dev/null || true; echo; }
start(){ prepare; if ! running "$API_PID_FILE" "uvicorn marketing_agent.api:app"; then nohup "$0" run-api >>"$LOG_DIR/api.log" 2>&1 & echo $! >"$API_PID_FILE"; fi; if ! running "$WORKER_PID_FILE" "marketing_agent.worker"; then nohup "$0" run-worker >>"$LOG_DIR/worker.log" 2>&1 & echo $! >"$WORKER_PID_FILE"; fi; for _ in $(seq 1 60); do if curl -fsS --max-time 2 "http://$API_HOST:$API_PORT/health" >/dev/null 2>&1; then echo "Marketing Agent is ready at http://$API_HOST:$API_PORT/app"; status; return; fi; sleep 1; done; echo "API did not become healthy; inspect $LOG_DIR/api.log" >&2; exit 1; }
stop_one(){ local file=$1 pattern=$2 pid; if ! running "$file" "$pattern"; then rm -f "$file"; return; fi; pid=$(<"$file"); kill -TERM "$pid"; for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid"; fi; rm -f "$file"; }
stop(){ stop_one "$WORKER_PID_FILE" "marketing_agent.worker"; stop_one "$API_PID_FILE" "uvicorn marketing_agent.api:app"; echo "Marketing Agent stopped"; }
case "${1:-}" in start) start;; stop) stop;; restart) stop; start;; status) status;; run-api) run_api;; run-worker) run_worker;; *) echo "Usage: $0 {start|stop|restart|status|run-api|run-worker}" >&2; exit 2;; esac
