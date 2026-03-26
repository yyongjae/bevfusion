#!/bin/bash

THRESHOLD=1000  # MB - 이 이하면 GPU가 비어있다고 판단
INTERVAL=300   # 초 (5분)
NUM_GPUS=4

echo "=== GPU 대기 후 자동 학습 스크립트 ==="
echo "조건: GPU ${NUM_GPUS}장 모두 메모리 사용량 ${THRESHOLD}MB 이하"
echo "체크 간격: $((INTERVAL / 60))분"
echo ""

while true; do
    ALL_FREE=true

    for i in $(seq 0 $((NUM_GPUS - 1))); do
        MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i | tr -d ' ')
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $i: ${MEM_USED}MB 사용 중"

        if [ "$MEM_USED" -gt "$THRESHOLD" ]; then
            ALL_FREE=false
        fi
    done

    if $ALL_FREE; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${NUM_GPUS}장 모두 비어있음! 학습을 시작합니다."
        echo ""

        cd /home/yongjae/e2e/bevfusion

        torchpack dist-run -np 4 python tools/train.py \
            configs/bench2drive/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \
            --model.encoders.camera.backbone.init_cfg.checkpoint /home/yongjae/e2e/bevfusion/tools/pretrained/swint-nuimages-pretrained.pth \
            --resume_from /home/yongjae/e2e/bevfusion/runs/run-326653dc-a4234a4b/epoch_3.pth \
            --run-dir runs/stage

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 학습 완료!"
        exit 0
    fi

    echo "=> GPU 사용 중. ${INTERVAL}초 후 다시 체크합니다..."
    echo ""
    sleep $INTERVAL
done
