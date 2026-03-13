#!/bin/bash
# =============================================================================
# DSL Debug Environment — Vast.ai Pod Management
# =============================================================================
#
# One-command workflow for renting GPUs and training.
#
# Prerequisites:
#   1. pip install vastai
#   2. vastai set api-key YOUR_KEY
#   3. Upload your SSH key at https://cloud.vast.ai/cli/
#      (or: vastai set ssh-key "$(cat ~/.ssh/id_rsa.pub)")
#
# Quick start:
#   bash scripts/vast.sh create          # rent cheapest 2xA100
#   bash scripts/vast.sh sync            # upload code + data
#   bash scripts/vast.sh train sft       # SFT training
#   bash scripts/vast.sh train grpo      # GRPO RL-only
#   bash scripts/vast.sh train sft-rl    # GRPO from SFT checkpoint
#   bash scripts/vast.sh logs            # watch progress
#   bash scripts/vast.sh destroy         # clean up

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VASTAI="${VASTAI:-vastai}"

# GPU preferences (override with env vars)
GPU_NAME="${GPU_NAME:-}"  # empty = any GPU, or e.g. A100_SXM4, H100_SXM
GPU_COUNT="${GPU_COUNT:-2}"
GPU_RAM_MIN="${GPU_RAM_MIN:-79}"
DISK_SIZE="${DISK_SIZE:-500}"
MAX_PRICE="${MAX_PRICE:-}"  # empty = no cap, just pick cheapest
CUDA_MIN="${CUDA_MIN:-12.4}"
IMAGE="${IMAGE:-andrewlngdn/dsl-debug-train}"

STATE_FILE="${SCRIPT_DIR}/.vast_state"
VAST_DENY="${VAST_DENY:-}"  # comma-separated machine IDs to skip

# Load .env
if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | grep -v '^$' | grep -E '^(WANDB_API_KEY|HF_TOKEN|WANDB_PROJECT)=' | xargs) 2>/dev/null || true
fi

# Build search query (price filter is optional)
_search_query() {
    local q="num_gpus=${GPU_COUNT} gpu_ram>=${GPU_RAM_MIN} disk_space>=${DISK_SIZE} inet_down>=200 reliability>=0.95 cuda_max_good>=${CUDA_MIN}"
    [ -n "${GPU_NAME}" ] && q="gpu_name=${GPU_NAME} ${q}"
    [ -n "${MAX_PRICE}" ] && q="${q} dph_total<=${MAX_PRICE}"
    echo "${q}"
}

# ============================================================================
# Helpers
# ============================================================================

_get_instance_id() {
    if [ -f "${STATE_FILE}" ]; then
        cat "${STATE_FILE}"
    else
        ${VASTAI} show instances --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
running = [d for d in data if d.get('actual_status') == 'running']
if running: print(running[0]['id'])
" 2>/dev/null || true
    fi
}

_get_ssh_info() {
    ${VASTAI} show instances --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for d in data:
    if str(d['id']) == '$1':
        # Prefer direct connection (more reliable than proxy relay)
        ports = d.get('ports', {})
        ssh_mapping = ports.get('22/tcp', [{}])
        direct_port = ssh_mapping[0].get('HostPort', '') if ssh_mapping else ''
        public_ip = d.get('public_ipaddr', '')
        if public_ip and direct_port:
            print(f'{public_ip} {direct_port}')
        else:
            print(f\"{d.get('ssh_host', '')} {d.get('ssh_port', '')}\")
        break
" 2>/dev/null
}

_ssh_cmd() {
    local instance_id=$(_get_instance_id)
    [ -z "${instance_id}" ] && echo "ERROR: No instance. Run: vast.sh create" >&2 && exit 1
    local ssh_info=$(_get_ssh_info "${instance_id}")
    local host=$(echo "${ssh_info}" | awk '{print $1}')
    local port=$(echo "${ssh_info}" | awk '{print $2}')
    [ -z "${host}" ] && echo "ERROR: No SSH info for ${instance_id}" >&2 && exit 1
    echo "ssh -o StrictHostKeyChecking=no root@${host} -p ${port}"
}

_run() { local ssh=$(_ssh_cmd); ${ssh} "$1"; }

_wait() {
    local id="$1" max="${2:-1800}"
    local elapsed=0
    local last_status=""
    echo "Waiting for instance ${id}..."
    for _ in $(seq 1 $((max / 5))); do
        # Get full status line from Vast
        local status_line=$(${VASTAI} show instances --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
match = [d for d in data if str(d['id']) == '${id}']
if not match: print('not found'); sys.exit()
d = match[0]
s = d.get('actual_status') or 'waiting'
extra = d.get('status_msg') or ''
if extra and extra != s: print(f'{s} ({extra})')
else: print(s)
" 2>/dev/null || echo "?")
        local status=$(echo "${status_line}" | awk '{print $1}')
        if [ "${status}" = "running" ]; then
            if [ "${last_status}" != "running" ]; then
                echo "  ${status_line}"
                last_status="running"
            fi
            local info=$(_get_ssh_info "${id}")
            local h=$(echo "${info}" | awk '{print $1}') p=$(echo "${info}" | awk '{print $2}')
            if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@${h} -p ${p} "echo ok" 2>/dev/null; then
                echo ""
                echo "  SSH ready! Connect with: bash scripts/vast.sh ssh"
                return 0
            fi
        elif [ "${status_line}" != "${last_status}" ]; then
            echo "  ${status_line}"
            last_status="${status_line}"
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: Timed out after ${max}s"
    return 1
}

# ============================================================================
# Commands
# ============================================================================

_filter_denied() {
    # Filter out VAST_DENY machine IDs from JSON offer list on stdin
    if [ -z "${VAST_DENY}" ]; then cat; return; fi
    python3 -c "
import json, sys, os
deny = set(os.environ.get('VAST_DENY', '').split(','))
data = json.load(sys.stdin)
print(json.dumps([o for o in data if str(o.get('machine_id', '')) not in deny and str(o.get('host_id', '')) not in deny]))
"
}

cmd_search() {
    local gpu_msg="${GPU_NAME:-any GPU}"
    local price_msg="${MAX_PRICE:+, <\$${MAX_PRICE}/hr}"
    echo "Searching for ${GPU_COUNT}x ${gpu_msg} (>=${GPU_RAM_MIN}GB${price_msg})..."
    ${VASTAI} search offers \
        "$(_search_query)" \
        -o 'dph_total' --storage "${DISK_SIZE}" 2>&1 | head -20
}

cmd_create() {
    local offer_id="${1:-}"
    if [ -z "${offer_id}" ]; then
        local gpu_msg="${GPU_NAME:-any GPU}"
        local price_msg="${MAX_PRICE:+, <\$${MAX_PRICE}/hr}"
        echo "Searching for cheapest ${GPU_COUNT}x ${gpu_msg} (>=${GPU_RAM_MIN}GB${price_msg})..."
        local offers_json=$(${VASTAI} search offers \
            "$(_search_query)" \
            -o 'dph_total' --storage "${DISK_SIZE}" --raw 2>/dev/null | _filter_denied)
        offer_id=$(echo "${offers_json}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if not d: sys.exit(1)
o = d[0]
print(o['id'])
" 2>/dev/null) || true
        [ -z "${offer_id}" ] && echo "ERROR: No offers found. Try: MAX_PRICE=4.00 vast.sh create" && exit 1
        # Print offer details
        echo "${offers_json}" | python3 -c "
import json, sys
o = json.load(sys.stdin)[0]
print(f\"  Best offer: \${o.get('dph_total', 0):.2f}/hr, {o.get('num_gpus', '?')}x {o.get('gpu_name', '?')}, {o.get('inet_down', 0):.0f} Mbps down\")
" 2>/dev/null
    fi

    echo "Creating instance from offer ${offer_id}..."
    echo "  Image: ${IMAGE}"
    echo "  Disk:  ${DISK_SIZE}GB"

    local result=$(${VASTAI} create instance "${offer_id}" \
        --image "${IMAGE}" --disk "${DISK_SIZE}" --ssh --direct \
        --label "dsl-debug" \
        --onstart-cmd "chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys 2>/dev/null; apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1" \
        --raw 2>&1)

    local id=$(echo "${result}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('new_contract',d.get('id','')))" 2>/dev/null || true)
    if [ -n "${id}" ]; then
        echo "${id}" > "${STATE_FILE}"
        echo "Instance created: ${id}"
        _wait "${id}"
    else
        echo "ERROR: Failed to create instance"
        echo "${result}"
        exit 1
    fi
}

cmd_sync() {
    local id=$(_get_instance_id)
    local info=$(_get_ssh_info "${id}")
    local host=$(echo "${info}" | awk '{print $1}')
    local port=$(echo "${info}" | awk '{print $2}')

    echo "Syncing to ${id} (${host}:${port})..."

    # Ensure rsync on remote
    ssh -o StrictHostKeyChecking=no root@${host} -p ${port} \
        "mkdir -p /workspace && command -v rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null 2>&1)" 2>/dev/null

    # Code (exclude secrets + build artifacts)
    echo "  Code..."
    rsync -az --delete \
        -e "ssh -o StrictHostKeyChecking=no -p ${port}" \
        --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
        --exclude '.venv' --exclude '.env' --exclude 'checkpoints' \
        "${PROJECT_ROOT}/" root@${host}:/workspace/dsl-debug/

    # Install dsl-debug CLI on pod
    _run "pip install -q /workspace/dsl-debug/ && ln -sf /workspace/dsl-debug/scripts/dsl-debug /usr/local/bin/dsl-debug"

    # Training data
    if [ -d "${PROJECT_ROOT}/data" ]; then
        echo "  Data..."
        _run "mkdir -p /workspace/dsl_debug_data/sft /workspace/dsl_debug_data/rl"
        for sub in sft rl; do
            [ -d "${PROJECT_ROOT}/data/${sub}" ] && \
                rsync -az --progress -e "ssh -o StrictHostKeyChecking=no -p ${port}" \
                    "${PROJECT_ROOT}/data/${sub}/" root@${host}:/workspace/dsl_debug_data/${sub}/
        done
    fi

    # Credentials
    # Send tokens via stdin to avoid leaking in process list
    local ssh=$(_ssh_cmd)
    [ -n "${WANDB_API_KEY:-}" ] && echo "${WANDB_API_KEY}" | ${ssh} "cat > /workspace/.wandb_key" && echo "  WandB key"
    [ -n "${HF_TOKEN:-}" ] && echo "${HF_TOKEN}" | ${ssh} "cat > /workspace/.hf_token" && echo "  HF token"
    echo "  Done!"
}

cmd_setup() {
    echo "Running setup on pod..."
    _run "dsl-debug setup"
}

cmd_train() {
    local mode="${1:-}"
    [ -z "${mode}" ] && echo "Usage: vast.sh train <sft|grpo|sft-rl>" && exit 1

    echo "Starting training: ${mode}"
    _run "cd /workspace && nohup dsl-debug train ${mode} > /dev/null 2>&1 &"
    echo "Training started. Use: vast.sh logs"
}

cmd_ssh() { local s=$(_ssh_cmd); echo "Connecting: ${s}"; ${s}; }
cmd_logs() { _run "LATEST=\$(ls -t /workspace/logs/*/train.log 2>/dev/null | head -1) && echo \"=== \${LATEST} ===\" && tail -f \"\${LATEST}\""; }
cmd_kill() { _run "ray stop --force 2>/dev/null; killall -9 python3 2>/dev/null; sleep 1; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"; }

cmd_status() {
    local id=$(_get_instance_id)
    [ -z "${id}" ] && echo "No instance tracked." && ${VASTAI} show instances 2>/dev/null && return
    ${VASTAI} show instances --raw 2>/dev/null | python3 -c "
import json, sys
for d in json.load(sys.stdin):
    if str(d['id']) == '${id}':
        print(f\"Instance:  {d['id']}\")
        print(f\"Status:    {d.get('actual_status', '?')}\")
        print(f\"GPU:       {d.get('num_gpus', '?')}x {d.get('gpu_name', '?')} ({d.get('gpu_ram', 0)//1024}GB)\")
        print(f\"Cost:      \${d.get('dph_total', 0):.2f}/hr\")
        print(f\"SSH:       ssh -o StrictHostKeyChecking=no root@{d.get('ssh_host', '?')} -p {d.get('ssh_port', '?')}\")
        break
"
    echo ""; _run "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null | \
        awk -F', ' 'BEGIN{printf "%-6s %-10s %-20s\n","GPU","Util","Memory"} {printf "%-6s %-10s %s/%s MiB\n",$1,$2"%",$3,$4}' || true
}

cmd_stop() {
    local id=$(_get_instance_id); [ -z "${id}" ] && echo "Nothing to stop" && return
    ${VASTAI} stop instance "${id}"; echo "Stopped. Restart: vastai start instance ${id}"
}

cmd_destroy() {
    local id=$(_get_instance_id); [ -z "${id}" ] && echo "Nothing to destroy" && return
    echo "WARNING: Destroy ${id}? All data will be deleted."
    read -p "(y/N) " c; [ "${c}" != "y" ] && echo "Cancelled." && return
    ${VASTAI} destroy instance "${id}"; rm -f "${STATE_FILE}"; echo "Destroyed."
}

cmd_sync_logs() {
    local id=$(_get_instance_id) info=$(_get_ssh_info "$(_get_instance_id)")
    local host=$(echo "${info}" | awk '{print $1}') port=$(echo "${info}" | awk '{print $2}')
    mkdir -p "${PROJECT_ROOT}/logs"
    rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -p ${port}" \
        root@${host}:/workspace/logs/ "${PROJECT_ROOT}/logs/"
    echo "Done! ${PROJECT_ROOT}/logs/"
}

# ============================================================================
# Main
# ============================================================================
case "${1:-help}" in
    search)    shift; cmd_search ;;
    create)    shift; cmd_create "$@" ;;
    sync)      cmd_sync ;;
    setup)     cmd_setup ;;
    train)     shift; cmd_train "$@" ;;
    ssh)       cmd_ssh ;;
    logs)      cmd_logs ;;
    status)    cmd_status ;;
    stop)      cmd_stop ;;
    destroy)   cmd_destroy ;;
    kill)      cmd_kill ;;
    sync-logs) cmd_sync_logs ;;
    *)
        echo "vast.sh — Vast.ai pod manager for DSL Debug Environment"
        echo ""
        echo "  search              Find GPU offers"
        echo "  create [OFFER_ID]   Rent instance (auto-pick if no ID)"
        echo "  sync                Upload code + data + credentials"
        echo "  setup               Download model + verify deps"
        echo "  train <mode>        Start training (sft|grpo|sft-rl)"
        echo "  ssh                 Connect to instance"
        echo "  logs                Tail training logs"
        echo "  status              Instance info + GPU utilization"
        echo "  stop                Pause (preserves data)"
        echo "  destroy             Delete instance"
        echo "  kill                Kill training processes"
        echo "  sync-logs           Download logs locally"
        echo ""
        echo "Quick start:"
        echo "  vast.sh create && vast.sh sync && vast.sh setup"
        echo "  vast.sh train sft"
        echo ""
        echo "GPU (env vars): GPU_NAME= (any, or A100_SXM4, H100_SXM) GPU_COUNT=2 MAX_PRICE= (unset=no cap)"
        echo "Deny machines:  VAST_DENY=12345,67890 vast.sh create"
        ;;
esac
