#!/bin/bash
# Wait for the running stage-1 (LiDAR-only) job to finish, verify it actually
# completed, then launch stage 2 (camera+LiDAR).
#
# Stage 2 is only started when stage 1 reached its final epoch: a crashed or
# killed stage 1 leaves a checkpoint too, and initialising the fusion model from
# a half-trained LiDAR branch would waste the next ~12 hours silently.
#
#   ./chain_stage2.sh              # auto-detect the running stage-1 mpirun
#   ./chain_stage2.sh <pid>        # or watch a specific pid
#
# Run it in its own tmux window; it prints to stdout and to chain_stage2.log.

set -u

STAGE1_DIR="runs/navsim-lidar-only-50x100"
STAGE2_DIR="runs/navsim-fusion-50x100"
EXPECTED_EPOCHS=20
GPUS="0,1,2,3,4,5"
NP=6
INTERVAL=60
MAX_ATTEMPTS=3
LOG="chain_stage2.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

PID="${1:-$(pgrep -f 'mpirun.*voxelnet' | head -1)}"
if [[ -z "$PID" ]]; then
    log "no stage-1 mpirun found; is it still running?"
    exit 1
fi

# Remember what the pid *is*, so a recycled pid cannot keep us waiting forever.
PID_CMD="$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null)"
log "watching stage-1 pid $PID (checking every ${INTERVAL}s)"
while kill -0 "$PID" 2>/dev/null; do
    NOW_CMD="$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null)"
    if [[ "$NOW_CMD" != "$PID_CMD" ]]; then
        log "pid $PID was recycled by another process; treating stage 1 as finished"
        break
    fi
    sleep "$INTERVAL"
done
log "stage-1 pid $PID exited"

# Let the last checkpoint land before inspecting it.
sleep 30

CKPT="$STAGE1_DIR/latest.pth"
if [[ ! -e "$CKPT" ]]; then
    log "ABORT: $CKPT missing -- stage 1 produced no checkpoint"
    exit 1
fi

# The checkpoint's own meta is the authority; the filename can lag a symlink.
EPOCH=$(python - "$CKPT" <<'PY'
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(ck.get("meta", {}).get("epoch", -1)))
except Exception as exc:                                    # noqa: BLE001
    print(f"ERR:{exc}", file=sys.stderr)
    print(-1)
PY
)
log "stage-1 checkpoint reports epoch=$EPOCH (expected $EXPECTED_EPOCHS)"
if [[ "$EPOCH" != "$EXPECTED_EPOCHS" ]]; then
    log "ABORT: stage 1 did not finish; not starting stage 2"
    exit 1
fi

# Only the newest log: older ones belong to earlier, aborted attempts in the
# same run directory and would raise a false alarm every time.
NEWEST_LOG="$(ls -t "$STAGE1_DIR"/*.log 2>/dev/null | head -1)"
if [[ -n "$NEWEST_LOG" ]] && grep -qi "Traceback" "$NEWEST_LOG"; then
    log "WARNING: traceback in $NEWEST_LOG -- inspect before trusting stage 1"
fi

# Stage 2 is ~12 hours; a single crash should not throw all of it away.  Each
# attempt resumes from the newest stage-2 checkpoint if there is one, so a retry
# costs at most the epoch in flight.  --resume_from and --load_from are mutually
# exclusive in mmdet3d (resume wins), and resuming already carries the stage-1
# weights, so only one of them is passed.
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if [[ -e "$STAGE2_DIR/latest.pth" ]]; then
        RESUME_ARGS=(--resume_from "$STAGE2_DIR/latest.pth")
        log "attempt $attempt/$MAX_ATTEMPTS: resuming from $STAGE2_DIR/latest.pth"
    else
        RESUME_ARGS=(--load_from "$STAGE1_DIR/latest.pth")
        log "attempt $attempt/$MAX_ATTEMPTS: starting fresh from stage-1 weights"
    fi

    CUDA_VISIBLE_DEVICES="$GPUS" torchpack dist-run -np "$NP" python tools/train.py \
        configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \
        --model.encoders.camera.backbone.init_cfg.checkpoint \
            pretrained/swint-nuimages-pretrained.pth \
        "${RESUME_ARGS[@]}" \
        --max_epochs 20 \
        --checkpoint_config.max_keep_ckpts -1 \
        --run-dir "$STAGE2_DIR"
    CODE=$?

    if [[ "$CODE" -eq 0 ]]; then
        log "stage 2 finished cleanly on attempt $attempt"
        exit 0
    fi
    log "stage 2 exited with code $CODE on attempt $attempt"

    # An immediate failure is a configuration or OOM problem: retrying just
    # reproduces it.  Only retry if the attempt actually trained for a while.
    if [[ ! -e "$STAGE2_DIR/latest.pth" ]]; then
        log "ABORT: no checkpoint was produced -- this is a startup failure, not a crash"
        exit 1
    fi
    sleep 60
done
log "ABORT: stage 2 still failing after $MAX_ATTEMPTS attempts"
exit 1
