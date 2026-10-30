#!/bin/bash
# Teacher BEV cache for the 64 m BEVFusion teacher (100 x 100 BEV, 0.64 m cells),
# same format as teacher_cache/bevfusion/cache_{train,val}_50x100:
#   <dir>/manifest.json + samples/<token[:2]>/<token>.npz (uncompressed, fp16 BEV)
#
#   val   -> /workspace/teacher_cache/bevfusion/cache_val_100x100    (navtest, 12,146)
#   train -> /workspace/teacher_cache/bevfusion/cache_train_100x100  (navtrain, 103,288)
#
#   setsid nohup ./cache_navsim_64m.sh > runs/cache_64m/chain.log 2>&1 &
#
# Re-running resumes (--skip-existing; npz are written to a temp name then renamed).
# post_center_range is widened (PCR) so every frame keeps all 200 proposals, like the
# 50x100 cache (pred_* = [200, ...], index-aligned with raw_*). The config's
# [-37,-37,-10,69,37,10] drops 1-20 low-score out-of-grid proposals in ~10 % of frames.
# The first build (2026-10-27 01:51-02:20 KST) ran without PCR; those 11,725 short frames
# were re-written with PCR afterwards (see manifest "patch").
# After each split, fix_manifest_64m.py rewrites the manifest's to_student_transform:
# cache_teacher_bev.py's text assumes a 100 x 100 teacher already matches PARA-SSR's
# 0-32 m grid, which is not true for the 64 m range.

set -u
cd "$(dirname "$0")" || exit 9

export PATH=/venv/bevfusion/bin:$PATH NCCL_SOCKET_IFNAME=lo OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

CFG=configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser_64m.yaml
CKPT=runs/navsim-fusion-64m/epoch_20.pth
PCR=(--model.heads.object.bbox_coder.post_center_range "[-1000.0,-1000.0,-1000.0,1000.0,1000.0,1000.0]")
OUT=${OUT:-/workspace/teacher_cache/bevfusion}
LOGDIR=runs/cache_64m
mkdir -p "$LOGDIR" "$OUT"
EVENTS=$LOGDIR/chain.events
log() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$EVENTS"; }

for split in ${SPLITS:-val train}; do
    dir=$OUT/cache_${split}_100x100
    for attempt in 1 2 3; do
        log "$split: attempt $attempt on $NP GPUs -> $dir"
        torchpack dist-run -np "$NP" python tools/cache_teacher_bev.py "$CFG" "$CKPT" \
            --cache-dir "$dir" --split "$split" --skip-existing "${PCR[@]}" \
            > "$LOGDIR/${split}_attempt${attempt}.log" 2>&1
        code=$?
        log "$split: exit $code"
        [[ $code == 0 ]] && break
        sleep 30
    done
    if [[ $code != 0 ]]; then
        log "$split: FAILED after 3 attempts"
        exit 1
    fi
    python tools/fix_manifest_64m.py "$dir/manifest.json" && log "$split: manifest fixed"
done
log "ALL DONE"
