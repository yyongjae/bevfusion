#!/bin/bash
# NAVSIM BEVFusion teacher, 64 m forward / +-32 m lateral, 100 x 100 BEV (0.64 m).
# Same two-stage recipe as the 50 x 100 teacher (ckpts/epoch_20.pth), on 4 GPUs:
#
#   stage 1  lidar/voxelnet_64m.yaml               20 ep, 6/GPU x 4 = 24 per step
#   stage 2  camera+lidar/swint_convfuser_64m.yaml 20 ep, 3/GPU x 4 x 2 accumulation
#            = 24 per step (4/GPU does not fit: spconv's index kernels pin ~12.3 GiB
#            of local memory per process on a 4090, independent of grid and batch)
#
# The old run was 6 GPUs x 4 = 24, 4,304 steps/epoch; both stages keep 24 per step.
# Stage-2 warmup is 1000 micro-iterations = the old 500 steps x 24 samples.
# Evaluation (navtest, 0-64 m) every 5 epochs instead of every epoch.
#
#   setsid nohup ./train_navsim_64m.sh > runs/navsim-64m_chain.log 2>&1 &
#   SMOKE=1 ./train_navsim_64m.sh       # 400-frame / 1-epoch dry run of both stages
#
# Re-running resumes: a stage with a final-epoch checkpoint is skipped, and an
# interrupted stage resumes from its latest.pth.

set -u
cd "$(dirname "$0")" || exit 9

export PATH=/venv/bevfusion/bin:$PATH NCCL_SOCKET_IFNAME=lo OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

S=configs/navsim/det/transfusion/secfpn
EPOCHS=20
EVAL_INTERVAL=5
S1_DIR=runs/navsim-lidar-only-64m
S2_DIR=runs/navsim-fusion-64m
EXTRA=()
if [[ "${SMOKE:-0}" == 1 ]]; then
    EPOCHS=1
    EVAL_INTERVAL=1
    S1_DIR=runs/_setup_64m/smoke4_s1
    S2_DIR=runs/_setup_64m/smoke4_s2
    EXTRA=(--data.train.ann_file runs/_setup_64m/smoke_train_400.pkl
           --data.val.ann_file runs/_setup_64m/smoke_val_64.pkl --log_config.interval 5)
fi
LOG=${S2_DIR%/*}/navsim-64m_chain.events

log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }

ckpt_epoch() {
    [[ -e "$1" ]] || { echo -1; return; }
    python - "$1" <<'PY' 2>/dev/null || echo -1
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ck.get("meta", {}).get("epoch", -1)))
PY
}

# run_stage <name> <dir> <config> <init args...>
run_stage() {
    local name=$1 dir=$2 cfg=$3; shift 3
    local init=("$@") attempt code
    if [[ "$(ckpt_epoch "$dir/latest.pth")" == "$EPOCHS" ]]; then
        log "$name: $dir/latest.pth is already epoch $EPOCHS, skipping"
        return 0
    fi
    for attempt in 1 2 3; do
        local start=()
        if [[ -e "$dir/latest.pth" ]]; then
            start=(--resume_from "$dir/latest.pth")
            log "$name attempt $attempt: resuming from $dir/latest.pth"
        else
            start=("${init[@]}")
            log "$name attempt $attempt: fresh start ${start[*]:-}"
        fi
        torchpack dist-run -np "$NP" python tools/train.py "$cfg" \
            "${start[@]}" "${STAGE_ARGS[@]}" "${EXTRA[@]}" \
            --max_epochs "$EPOCHS" \
            --checkpoint_config.max_keep_ckpts -1 \
            --evaluation.interval "$EVAL_INTERVAL" \
            --run-dir "$dir"
        code=$?
        if [[ "$(ckpt_epoch "$dir/latest.pth")" == "$EPOCHS" ]]; then
            log "$name finished (rc=$code, epoch $EPOCHS)"
            return 0
        fi
        log "$name exited rc=$code before epoch $EPOCHS"
        # No checkpoint at all = a startup/config/OOM problem; retrying repeats it.
        [[ -e "$dir/latest.pth" ]] || { log "ABORT: $name produced no checkpoint"; return 1; }
        sleep 60
    done
    log "ABORT: $name still failing after 3 attempts"
    return 1
}

mkdir -p "${S2_DIR%/*}"
echo $$ > "${S2_DIR%/*}/navsim-64m_chain.pid"
log "chain start: GPUs $CUDA_VISIBLE_DEVICES (np $NP), epochs $EPOCHS, smoke=${SMOKE:-0}"

STAGE_ARGS=(--data.samples_per_gpu 6)
run_stage stage1 "$S1_DIR" "$S/lidar/voxelnet_64m.yaml" || exit 1

STAGE_ARGS=(--data.samples_per_gpu 3 --optimizer_config.cumulative_iters 2
            --lr_config.warmup_iters 1000
            --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth)
run_stage stage2 "$S2_DIR" "$S/camera+lidar/swint_convfuser_64m.yaml" \
    --load_from "$S1_DIR/latest.pth" || exit 1

log "chain done: $S2_DIR/latest.pth"
