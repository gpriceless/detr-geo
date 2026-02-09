#!/usr/bin/env bash
# monitor_training.sh — Check xView training status on RunPod and write report
#
# Usage: bash scripts/monitor_training.sh
# Cron:  */30 * * * * bash /home/gprice/projects/detr_geo/scripts/monitor_training.sh
#
# Writes status to: docs/xview-training-live.md
# Syncs best checkpoint to: /home/gprice/projects/detr_geo/checkpoints/

set -uo pipefail

# ---------------------------------------------------------------------------
# Pod details
# ---------------------------------------------------------------------------
POD_HOST="104.255.9.187"
POD_PORT="11873"
POD_USER="root"
SSH_KEY="$HOME/.runpod/ssh"
TRAINING_PID=210
TRAINING_LOG="/runpod-volume/training.log"
CHECKPOINT_DIR="/runpod-volume/training_output/xview"

# Local paths
PROJECT_DIR="/home/gprice/projects/detr_geo"
STATUS_FILE="${PROJECT_DIR}/docs/xview-training-live.md"
LOCAL_CKPT_DIR="${PROJECT_DIR}/checkpoints"
LOCK_FILE="/tmp/monitor_training.lock"

# SSH options (tight timeout for spot instance that may be gone)
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -p ${POD_PORT}"
# scp uses -P (uppercase) for port, not -p
SCP_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10 -P ${POD_PORT}"

# ---------------------------------------------------------------------------
# Matrix notification
# ---------------------------------------------------------------------------
MATRIX_CREDS="/home/gprice/projects/matrix_meyhem/config/.credentials"
MATRIX_CHANNEL="#agents-general"

matrix_post() {
    local msg="$1"
    if [ -f "$MATRIX_CREDS" ]; then
        (
            source "$MATRIX_CREDS"
            MATRIX_HOMESERVER="$MATRIX_SERVER" \
            MATRIX_ACCESS_TOKEN="$AGENT_explorer_TOKEN" \
            MATRIX_AGENT_NAME="training-monitor" \
            matrix-post "$MATRIX_CHANNEL" "$msg"
        ) 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
# Prevent overlapping runs
# ---------------------------------------------------------------------------
if [ -f "$LOCK_FILE" ]; then
    LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0) ))
    if [ "$LOCK_AGE" -lt 300 ]; then
        echo "Another monitor run is in progress (lock age: ${LOCK_AGE}s). Exiting."
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

mkdir -p "$(dirname "$STATUS_FILE")"
mkdir -p "$LOCAL_CKPT_DIR"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ---------------------------------------------------------------------------
# Try to connect
# ---------------------------------------------------------------------------
POD_STATUS=$(ssh $SSH_OPTS ${POD_USER}@${POD_HOST} "echo OK" 2>/dev/null)

if [ "$POD_STATUS" != "OK" ]; then
    cat > "$STATUS_FILE" << EOF
# xView Training Status

**Last checked:** ${TIMESTAMP}
**Pod:** bj2sgxnooihy65 (A100 PCIe 80GB spot @ \$0.65/hr)

## STATUS: UNREACHABLE

Pod is not responding. Possible causes:
- Pod was **preempted** (spot instance reclaimed)
- Network issue
- Pod was stopped manually

### Action needed
Check RunPod dashboard or run: \`runpod pod list\`

If preempted, check local checkpoints at: \`checkpoints/\`
EOF
    echo "[${TIMESTAMP}] Pod unreachable" >> "${PROJECT_DIR}/docs/training-monitor.log"

    exit 1
fi

# ---------------------------------------------------------------------------
# Gather status from pod
# ---------------------------------------------------------------------------
REMOTE_DATA=$(ssh $SSH_OPTS ${POD_USER}@${POD_HOST} bash << 'REMOTE_SCRIPT'
echo "=== PROCESS ==="
TRAIN_PID=$(pgrep -f 'python.*train_xview' | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "RUNNING"
    ps -o etime= -p "$TRAIN_PID" 2>/dev/null | tr -d ' '
else
    echo "STOPPED"
    echo "N/A"
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A"

echo "=== LOG_TAIL ==="
tail -3 /runpod-volume/training.log 2>/dev/null || echo "No log found"

echo "=== EPOCH ==="
# Extract current epoch and batch from log format: Epoch: [0]  [1490/1994]
LAST_LINE=$(grep 'Epoch:' /runpod-volume/training.log 2>/dev/null | tail -1)
if [ -n "$LAST_LINE" ]; then
    EP=$(echo "$LAST_LINE" | grep -oP 'Epoch: \[\K[0-9]+')
    BATCH=$(echo "$LAST_LINE" | grep -oP '\]\s+\[\K[0-9]+/[0-9]+')
    echo "${EP}/49 (batch ${BATCH})"
else
    echo "N/A"
fi

echo "=== LOSS ==="
# Extract total loss from log format: loss: 8.1210 (9.0564)
grep -oP 'loss: [0-9]+\.[0-9]+ \([0-9]+\.[0-9]+\)' /runpod-volume/training.log 2>/dev/null | tail -5 || echo "N/A"

echo "=== CHECKPOINTS ==="
ls -lht /runpod-volume/training_output/xview/*.pth 2>/dev/null | head -10 || echo "None"

echo "=== DISK ==="
df -h /runpod-volume 2>/dev/null | tail -1 || echo "N/A"

echo "=== BEST_METRIC ==="
# Try to find best metric from log
grep -i "best\|mAP\|AP50" /runpod-volume/training.log 2>/dev/null | tail -3 || echo "N/A"
REMOTE_SCRIPT
)

# ---------------------------------------------------------------------------
# Parse remote data
# ---------------------------------------------------------------------------
parse_section() {
    echo "$REMOTE_DATA" | sed -n "/=== $1 ===/,/=== /p" | head -n -1 | tail -n +2
}

PROCESS_STATUS=$(parse_section "PROCESS" | head -1)
PROCESS_UPTIME=$(parse_section "PROCESS" | tail -1)
GPU_INFO=$(parse_section "GPU")
LOG_TAIL=$(parse_section "LOG_TAIL")
CURRENT_EPOCH=$(parse_section "EPOCH")
RECENT_LOSS=$(parse_section "LOSS")
CHECKPOINTS=$(parse_section "CHECKPOINTS")
DISK_INFO=$(parse_section "DISK")
BEST_METRIC=$(parse_section "BEST_METRIC")

# Calculate cost estimate
if [ "$PROCESS_UPTIME" != "N/A" ] && [ -n "$PROCESS_UPTIME" ]; then
    # Parse etime format: [[DD-]HH:]MM:SS
    HOURS=$(echo "$PROCESS_UPTIME" | awk -F'[:-]' '{
        n = NF;
        if (n == 4) print $1*24 + $2 + $3/60;
        else if (n == 3) print $1 + $2/60;
        else if (n == 2) print $1/60;
        else print 0;
    }')
    COST=$(echo "$HOURS * 0.65" | bc -l 2>/dev/null | xargs printf "%.2f" 2>/dev/null || echo "N/A")
else
    HOURS="N/A"
    COST="N/A"
fi

# Parse epoch numbers for progress bar
if [ "$CURRENT_EPOCH" != "N/A" ] && [ -n "$CURRENT_EPOCH" ]; then
    CURR_EP=$(echo "$CURRENT_EPOCH" | cut -d'/' -f1)
    TOTAL_EP=$(echo "$CURRENT_EPOCH" | cut -d'/' -f2)
    if [ -n "$TOTAL_EP" ] && [ "$TOTAL_EP" -gt 0 ] 2>/dev/null; then
        PCT=$(( CURR_EP * 100 / TOTAL_EP ))
        FILLED=$(( PCT / 5 ))
        EMPTY=$(( 20 - FILLED ))
        PROGRESS_BAR=$(printf '%0.s█' $(seq 1 $FILLED 2>/dev/null) 2>/dev/null)$(printf '%0.s░' $(seq 1 $EMPTY 2>/dev/null) 2>/dev/null)
    else
        PCT="?"
        PROGRESS_BAR="unknown"
    fi
else
    CURR_EP="?"
    TOTAL_EP="?"
    PCT="?"
    PROGRESS_BAR="not started"
fi

# ---------------------------------------------------------------------------
# Write status file
# ---------------------------------------------------------------------------
cat > "$STATUS_FILE" << EOF
# xView Training Status

**Last checked:** ${TIMESTAMP}
**Pod:** bj2sgxnooihy65 (A100 PCIe 80GB spot @ \$0.65/hr)
**SSH:** root@${POD_HOST} -p ${POD_PORT}

## Status: ${PROCESS_STATUS}

| Metric | Value |
|--------|-------|
| Epoch | ${CURRENT_EPOCH:-N/A} |
| Progress | ${PROGRESS_BAR} ${PCT}% |
| Uptime | ${PROCESS_UPTIME:-N/A} |
| Est. Cost | \$${COST:-N/A} |
| GPU Util | ${GPU_INFO:-N/A} |
| Disk | ${DISK_INFO:-N/A} |

### Recent Loss Values
\`\`\`
${RECENT_LOSS:-No loss data yet}
\`\`\`

### Best Metrics
\`\`\`
${BEST_METRIC:-No metrics yet}
\`\`\`

### Checkpoints on Pod
\`\`\`
${CHECKPOINTS:-None yet}
\`\`\`

### Latest Log Output
\`\`\`
${LOG_TAIL:-No log output}
\`\`\`

---
*Auto-updated every 30 minutes by monitor_training.sh*
*Local checkpoint backup: checkpoints/*
EOF

# ---------------------------------------------------------------------------
# Sync best checkpoint to local (if exists and training running)
# ---------------------------------------------------------------------------
if [ "$PROCESS_STATUS" = "RUNNING" ] || [ "$PROCESS_STATUS" = "STOPPED" ]; then
    # Check if best checkpoint exists on remote
    BEST_EXISTS=$(ssh $SSH_OPTS ${POD_USER}@${POD_HOST} \
        "ls ${CHECKPOINT_DIR}/checkpoint_best_ema.pth > /dev/null 2>&1 && echo YES || echo NO")

    if [ "$BEST_EXISTS" = "YES" ]; then
        # Get remote file size/date
        REMOTE_INFO=$(ssh $SSH_OPTS ${POD_USER}@${POD_HOST} \
            "stat -c '%s %Y' ${CHECKPOINT_DIR}/checkpoint_best_ema.pth 2>/dev/null")
        REMOTE_SIZE=$(echo "$REMOTE_INFO" | awk '{print $1}')
        REMOTE_MTIME=$(echo "$REMOTE_INFO" | awk '{print $2}')

        LOCAL_CKPT="${LOCAL_CKPT_DIR}/checkpoint_best_ema.pth"
        SHOULD_SYNC=false

        if [ ! -f "$LOCAL_CKPT" ]; then
            SHOULD_SYNC=true
        else
            LOCAL_SIZE=$(stat -c '%s' "$LOCAL_CKPT" 2>/dev/null || echo 0)
            LOCAL_MTIME=$(stat -c '%Y' "$LOCAL_CKPT" 2>/dev/null || echo 0)
            if [ "$REMOTE_SIZE" != "$LOCAL_SIZE" ] || [ "$REMOTE_MTIME" -gt "$LOCAL_MTIME" ]; then
                SHOULD_SYNC=true
            fi
        fi

        if [ "$SHOULD_SYNC" = true ]; then
            CKPT_SIZE_HR=$(numfmt --to=iec ${REMOTE_SIZE:-0} 2>/dev/null || echo '?')
            echo "[${TIMESTAMP}] Syncing best checkpoint (${CKPT_SIZE_HR})..."
            scp $SCP_OPTS ${POD_USER}@${POD_HOST}:${CHECKPOINT_DIR}/checkpoint_best_ema.pth \
                "$LOCAL_CKPT" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "[${TIMESTAMP}] Best EMA checkpoint synced" >> "${PROJECT_DIR}/docs/training-monitor.log"
                # Extract latest loss for the message
                LOSS_VAL=$(echo "$RECENT_LOSS" | tail -1 | grep -oP '\([0-9.]+\)' | tr -d '()' || echo "N/A")
                matrix_post "[CHECKPOINT] Best EMA synced locally. Epoch: ${CURRENT_EPOCH:-?}, loss: ${LOSS_VAL:-N/A}, cost so far: \$${COST:-?}."
            else
                echo "[${TIMESTAMP}] Checkpoint sync FAILED" >> "${PROJECT_DIR}/docs/training-monitor.log"
            fi
        fi
    fi

    # Also sync the latest regular checkpoint
    LATEST_CKPT=$(ssh $SSH_OPTS ${POD_USER}@${POD_HOST} \
        "ls -1t ${CHECKPOINT_DIR}/checkpoint_*.pth 2>/dev/null | grep -v best | head -1")
    if [ -n "$LATEST_CKPT" ]; then
        CKPT_NAME=$(basename "$LATEST_CKPT")
        if [ ! -f "${LOCAL_CKPT_DIR}/${CKPT_NAME}" ]; then
            echo "[${TIMESTAMP}] Syncing ${CKPT_NAME}..."
            scp $SCP_OPTS ${POD_USER}@${POD_HOST}:${LATEST_CKPT} \
                "${LOCAL_CKPT_DIR}/${CKPT_NAME}" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "[${TIMESTAMP}] ${CKPT_NAME} synced" >> "${PROJECT_DIR}/docs/training-monitor.log"
                matrix_post "[CHECKPOINT] ${CKPT_NAME} synced locally. Epoch: ${CURRENT_EPOCH:-?}, cost: \$${COST:-?}."
            fi
        fi
    fi

    # Sync small metadata files (class_names.json, training_summary.json, training_config.json, log)
    for REMOTE_FILE in class_names.json training_summary.json training_config.json; do
        LOCAL_FILE="${LOCAL_CKPT_DIR}/${REMOTE_FILE}"
        REMOTE_EXISTS=$(ssh $SSH_OPTS ${POD_USER}@${POD_HOST} \
            "ls ${CHECKPOINT_DIR}/${REMOTE_FILE} > /dev/null 2>&1 && echo YES || echo NO")
        if [ "$REMOTE_EXISTS" = "YES" ]; then
            REMOTE_MD5=$(ssh $SSH_OPTS ${POD_USER}@${POD_HOST} \
                "md5sum ${CHECKPOINT_DIR}/${REMOTE_FILE} 2>/dev/null | awk '{print \$1}'" )
            LOCAL_MD5=$(md5sum "$LOCAL_FILE" 2>/dev/null | awk '{print $1}')
            if [ "$REMOTE_MD5" != "$LOCAL_MD5" ]; then
                scp $SCP_OPTS ${POD_USER}@${POD_HOST}:${CHECKPOINT_DIR}/${REMOTE_FILE} \
                    "$LOCAL_FILE" 2>/dev/null
                if [ $? -eq 0 ]; then
                    echo "[${TIMESTAMP}] ${REMOTE_FILE} synced" >> "${PROJECT_DIR}/docs/training-monitor.log"
                fi
            fi
        fi
    done

    # Notify once if training process stopped but pod is still up (completed or crashed)
    if [ "$PROCESS_STATUS" = "STOPPED" ]; then
        LAST_LOG_LINE=$(tail -1 "${PROJECT_DIR}/docs/training-monitor.log" 2>/dev/null)
        if ! echo "$LAST_LOG_LINE" | grep -q "TRAINING_STOPPED"; then
            echo "[${TIMESTAMP}] TRAINING_STOPPED" >> "${PROJECT_DIR}/docs/training-monitor.log"
            matrix_post "[TRAINING] Process stopped while pod is still up. May have completed or crashed. Last epoch: ${CURRENT_EPOCH:-?}, cost: \$${COST:-?}."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Append to log
# ---------------------------------------------------------------------------
echo "[${TIMESTAMP}] ${PROCESS_STATUS} | Epoch: ${CURRENT_EPOCH:-N/A} | Cost: \$${COST:-N/A}" \
    >> "${PROJECT_DIR}/docs/training-monitor.log"

echo "Status written to ${STATUS_FILE}"
