# BEVFusion(NAVSIM) 설치 시 docs/NAVSIM.md 와 다르게 한 부분

작업일: 2026-09-08
환경: Ubuntu 24.04 / AMD Ryzen Threadripper PRO 9955WX (16C) / **RTX 5090 x6** / driver 580.173.02
conda env: `bevfusion` (python 3.9.23)

`docs/NAVSIM.md` 1절의 환경은 **RTX A6000(sm_86) + CUDA 11.3** 기준이다.
이 서버는 RTX 5090(**sm_120**, Blackwell)이라 문서 절차를 그대로 따르면
CUDA 커널이 하나도 실행되지 않는다. 아래는 실제로 성공한 절차와 달라진 이유.

---

## 왜 문서 그대로는 불가능한가

| 항목 | 문서 | 이 서버에서의 문제 |
|---|---|---|
| torch | 1.10.2+cu113 | cu113에 sm_120 커널 없음. 설치는 되지만 전 연산 실패 |
| CUDA toolkit | 11.3.1 (conda) | nvcc 11.3은 `compute_120` 자체를 모름 |
| `TORCH_CUDA_ARCH_LIST` | `"8.6"` | A6000용. 5090은 `"12.0"` |
| python | 3.8 | torch가 2.4 이후 3.8 지원 중단 → cu128 wheel은 cp39부터 |
| mmcv-full | 1.4.0 prebuilt | cu128 prebuilt index가 존재하지 않음. 1.4.0은 torch 2.x에서 빌드 불가 |
| `setup.py` | `-gencode` sm_70~86 하드코딩 | sm_120이 조용히 빠짐 (아래 패치 1) |

---

## 요약: 문서와 달라진 점 8가지

| # | 문서 | 실제 | 이유 |
|---|---|---|---|
| 1 | `python=3.8` | **3.9** | cu128 wheel이 cp39부터 |
| 2 | `torch==1.10.2+cu113` / `torchvision==0.11.3` | **`torch==2.8.0+cu128` / `torchvision==0.23.0`** | sm_120 |
| 3 | `mmcv-full==1.4.0` prebuilt wheel | **1.7.2 소스 빌드** (CUDA 12.8) | cu128 wheel 없음. 1.x 중 torch 2.x를 지원하는 건 1.7.2가 마지막 |
| 4 | `mmdet==2.20.0` | **2.28.2** | 2.20은 `mmcv-full<1.5`만 허용 |
| 5 | `conda install cuda-toolkit=11.3.1 gxx_linux-64=9.5.0` | **시스템 `/usr/local/cuda-12.8` + `gcc-13`** | CUDA 12.8은 gcc 14까지 허용. conda 툴체인 불필요 |
| 6 | `pip install "mpi4py==3.1.4"` | **conda-forge로 설치** | pip 빌드가 최신 setuptools와 충돌: `new_compiler() got an unexpected keyword argument 'dry_run'` |
| 7 | `Pillow==8.4.0` | **9.5.0** | 핀 사유인 `Image.LINEAR`은 9.5까지 존재하고, 8.4는 torchvision 0.23과 안 맞음 |
| 8 | (없음) | **버전 재고정 단계 추가** | mmcv 소스 빌드가 numpy 2.0.2 / opencv 5.0.0 / yapf 0.43을 끌어옴 |
| 9 | 800x450 재추출본 전제 | **1920x1080 원본이 기본값** | 이 서버에는 원본만 있다. converter `--image-width` 기본값과 `augment2d.resize` 를 1920 기준으로 옮김 (아래 이미지 해상도 항목) |

---

## 실제 설치 절차 (재현용)

### 1. 환경 생성

```bash
conda create -y -n bevfusion -c conda-forge --override-channels \
    python=3.9 openmpi=4.1 ninja
conda activate bevfusion
```

`ninja`가 없으면 CUDA extension 빌드가 직렬로 돌아 수 배 느려진다.

### 2. PyTorch (cu128)

```bash
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

`torch.cuda.get_arch_list()` 에 `sm_120` 이 있어야 한다.

### 3. 기본 핀

```bash
pip install "setuptools==59.5.0" "wheel==0.37.1" "Pillow==9.5.0" tqdm "numpy==1.23.5"
```

`setuptools==59.5.0` 은 `python setup.py develop` 때문에 필요하며 **torch 설치 후, 빌드 전**에 맞춰 둔다.

### 4. mmcv-full 1.7.2 소스 빌드 (문서 대비 최대 변경점)

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
export TORCH_CUDA_ARCH_LIST="12.0" MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=16

pip install --no-build-isolation --no-binary mmcv-full "mmcv-full==1.7.2"
```

빌드는 정상 완료된다(약 10분). 검증:

```bash
python -c "
import torch; from mmcv.ops import nms
b=torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.],[50.,50.,60.,60.]],device='cuda')
print(nms(b, torch.tensor([0.9,0.8,0.7],device='cuda'), 0.5)[1].tolist())"
```

### 5. 나머지 의존성

```bash
pip install "mmdet==2.28.2" "numba==0.56.4" nuscenes-devkit torchpack \
    "shapely<2.0" tensorboard

# mpi4py는 pip 빌드가 실패하므로 conda-forge로
conda install -y -c conda-forge --override-channels mpi4py=3.1.4
```

### 6. 버전 재고정 — **문서에 없는 필수 단계**

```bash
pip install "numpy==1.23.5" "opencv-python==4.9.0.80" "yapf==0.40.1" "setuptools==59.5.0"
```

4단계 mmcv 소스 빌드가 `numpy 2.0.2`, `opencv-python 5.0.0.93`, `yapf 0.43.0` 을 끌어온다.
`numba==0.56.4` 는 `numpy<1.24` 를 요구하고, opencv 5.0은 numpy 2.x 전제라 그대로 두면 깨진다.

### 7. mmcv 패치 (torch 2.x 호환) — 문서에 없음

mmcv 1.7.2의 DataParallel 헬퍼 2곳이 torch 2.x에서 바뀐 API를 쓴다.

```bash
python tools/patch_mmcv_for_torch2.py
```

- `mmcv/parallel/_functions.py` — `_get_stream(device)` 에 `int`를 넘긴다. torch 2.0+ 는 `torch.device`를 요구
  → `AttributeError: 'int' object has no attribute 'type'`
- `mmcv/parallel/distributed.py` — torch 2.1에서 제거된 `_use_replicated_tensor_module` 참조
  → `AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'`

스크립트는 idempotent다. **mmcv를 재설치하면 반드시 다시 실행**해야 한다.

### 8. 저장소 CUDA extension 빌드

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=$CUDA_HOME/bin:$PATH
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
export TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 MAX_JOBS=16
python setup.py develop
```

`.so` 13개 생성 확인 (feature_decorator, spconv, bev_pool, iou3d, voxel_layer,
roiaware_pool3d, ball_query, knn, paconv, group_points, interpolate,
furthest_point_sample, gather_points).

아키텍처 확인:

```bash
cuobjdump --list-elf mmdet3d/ops/bev_pool/bev_pool_ext*.so   # -> ...sm_120.cubin
```

---

## 저장소 코드 변경 2건

### 패치 1 — `setup.py`: `TORCH_CUDA_ARCH_LIST` 를 존중

`make_cuda_ext` 가 `-gencode` 를 sm_70~86으로 하드코딩하고 있었다.
torch의 `cpp_extension` 은 `extra_compile_args["nvcc"]` 에 `-gencode` 가 이미 있으면
`TORCH_CUDA_ARCH_LIST` 로부터 자기 플래그를 **주입하지 않는다**. 즉 환경변수를 아무리 설정해도
sm_120이 빠진 채 빌드된다. 환경변수가 설정된 경우에만 하드코딩 목록을 건너뛰도록 바꿨다.
(미설정 시 기존 동작 유지 → 다른 장비에서 회귀 없음)

### 패치 2 — `mmdet3d/ops/spconv/conv.py`: `register_module(force=True)` 10곳

mmcv 1.5+ 부터 `mmcv/ops/sparse_conv.py` 가 `SparseConv2d` / `SparseConv3d` /
`SubMConv3d` 등을 **같은 이름으로 CONV_LAYERS에 먼저 등록**한다.

```
KeyError: 'SparseConv2d is already registered in conv layer'
```

이 저장소의 모델은 자체 spconv 구현을 쓰므로 우리 쪽이 이겨야 한다 → `force=True`.
mmcv 1.4.0에는 이 파일이 없어서 원래는 충돌하지 않았다.

---

## 단일 노드 DDP: `NCCL_SOCKET_IFNAME=lo`

이 서버에는 `docker0` / `vxlan.calico` / 2개의 물리 NIC가 있어 NCCL이 `enp211s0f0np0`(10.20.20.31)을
잡고 실패한다. **`-np 1` 에서도** 발생한다.

```
ncclRemoteError: ... socketPollConnect: connect to 10.20.20.31<55363> returned Connection refused
```

`$CONDA_PREFIX/etc/conda/activate.d/bevfusion_env.sh` 에 넣어 `conda activate bevfusion` 시 자동 적용:

```bash
#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export TORCH_CUDA_ARCH_LIST="12.0"
export NCCL_SOCKET_IFNAME=lo
```

대응하는 `deactivate.d/bevfusion_env.sh` 에서 unset.

---

## 데이터

문서 2절의 `/home/data/...` 레이아웃은 이 서버에 없다. 실제 데이터는
`/home/external-user/navsim/download/` 이고, **카메라와 LiDAR가 같은 트리**에 있다
(`trainval_sensor_blobs/trainval/<log>/{CAM_*,MergedPointCloud}/`).
따라서 문서와 달리 `sensor_blobs` 와 `lidar` 심볼릭 링크가 같은 곳을 가리킨다.

```bash
B=/home/external-user/navsim/download
mkdir -p data/navsim data/infos
ln -sfn $B/trainval_sensor_blobs data/navsim/sensor_blobs
ln -sfn $B/trainval_sensor_blobs data/navsim/lidar
ln -sfn $B/maps                  data/navsim/maps
```

### 이미지 해상도

원 문서는 800x450 재추출본을 전제하지만 이 서버에는 **원본 1920x1080** 이 있다.
**코드 기본값을 1920으로 옮겼다** (변경 9번). 따라서 `--image-width` 도,
`augment2d.resize` override 도 더 이상 붙일 필요가 없다.

| | 800px (원 문서) | **1920px (현재 기본값)** |
|---|---|---|
| `--image-width` | `800` | `1920` (default) |
| eval resize | `0.88` | `0.36667` (= 704/1920) |
| train resize | `[0.76, 1.0]` | `[0.31667, 0.41667]` |
| 결과 폭 | 608 ~ 800 px | **동일** |

두 설정은 **최종 픽셀 크기가 완전히 같다** (800x0.76 = 1920x0.31667 = 608,
800x1.0 = 1920x0.41667 = 800). 증강 동작은 그대로고 소스 디테일만 좋아진다.

### `.pcd` 인코딩

`DATA binary` (비압축), `FIELDS x y z intensity lidar_info ring` — 문서 4절과 일치.
비압축이므로 `python-lzf` 는 필요 없다.

### 현재 확보된 split

- `navtrain` 1192개 로그 → **전부 존재**
- `navtest` 136개 로그 → **0개**. val/eval을 하려면 navsim 쪽 `download_test.sh` 필요

---

## 학습 명령어

아래 전부 `conda activate bevfusion` 상태에서 저장소 루트 기준.
activate 훅이 `NCCL_SOCKET_IFNAME` / `CUDA_HOME` 을 잡아주므로 별도 export 불필요.

### 0. info 파일 — **이미 생성 완료**

```
data/infos/navsim_infos_train.pkl                     1.8G  103,288 frames / 1192 logs / 8,640,908 boxes
data/infos/navsim_infos_val_sanity_from_navtrain.pkl   21M    1,192 frames (로그당 1장)
data/infos/navsim_infos_val.pkl -> navsim_infos_val_sanity_from_navtrain.pkl
```

`navsim_infos_train.pkl` 은 공식 navtrain 토큰 수(103,288)와 정확히 일치한다.
무작위 300 프레임 표본에서 이미지/LiDAR/sweep 결손 0건, `cam_intrinsic` fx=1545(1920 스케일) 확인.

> **`navsim_infos_val.pkl` 은 공식 navtest가 아니다.** navtest 136개 로그가 아직 안 받아져 있어
> (아래 "test split" 항목) navtrain에서 뽑은 **sanity 세트**이고, 학습 데이터와 겹친다.
> 학습이 시작조차 못 하는 것을 막는 용도이며, **여기서 나온 mAP/NDS는 성능 지표가 아니다.**
> 심볼릭 링크로 걸어둔 것도 `ls -l` 에서 출처가 바로 보이게 하기 위해서다.
>
> trainval 다운로드에 sensor blob이 navtrain 1192개 로그분만 들어있어서
> (navtrain에 안 쓰이는 118개 로그는 annotation만 존재) 누수 없는 val을 만드는 것은 불가능했다.

재생성이 필요하면:

```bash
SF=/home/external-user/navsim/navsim/planning/script/config/common/train_test_split/scene_filter
B=/home/external-user/navsim/download

# train (navtrain 103,288 tokens)
python tools/data_converter/navsim_converter.py \
    --navsim-logs $B/trainval_navsim_logs/trainval \
    --sensor-root $B/trainval_sensor_blobs \
    --split trainval \
    --scene-filter $SF/navtrain.yaml \
    --keyframes-only --check-files --workers 24 \
    --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 \
    --out data/infos/navsim_infos_train.pkl
```

`--scene-filter` + `--keyframes-only` 조합에서는 sweep LiDAR 결손이 **0%** 다(12개 로그 723 프레임 표본, 100% 존재).
반대로 scene filter 없이 변환하면 로그 경계의 비공식 프레임이 섞여 sweep의 약 17%가 없어 학습 도중
`FileNotFoundError` 가 난다. **문서 레시피대로 반드시 필터를 걸 것.**

공식 val/eval은 test split을 받은 뒤에 만든다 (그때 위 심볼릭 링크를 실제 파일로 교체):

```bash
# navsim 저장소에서 bash download_test.sh 를 먼저 실행한 뒤
python tools/data_converter/navsim_converter.py \
    --navsim-logs $B/test_navsim_logs/test \
    --sensor-root $B/test_sensor_blobs \
    --split test \
    --scene-filter $SF/navtest.yaml \
    --keyframes-only --check-files --workers 24 \
    --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 \
    --out data/infos/navsim_infos_val.pkl
```

이때 `data/navsim/sensor_blobs` / `lidar` 가 test 트리도 함께 볼 수 있어야 한다.

> `configs/navsim/default.yaml` 의 `augment2d.resize` 는 이미 1920px 기준
> (`[[0.31667, 0.41667], [0.36667, 0.36667]]`) 이므로 별도 override가 필요 없다.
> 다른 해상도로 재추출했다면 `--image-width` 와 eval resize(`704 / new_width`)를 맞춰준다.

### 1. stage 1 — LiDAR only

```bash
CUDA_VISIBLE_DEVICES=0,1 torchpack dist-run -np 2 python tools/train.py \
    configs/navsim/det/transfusion/secfpn/lidar/voxelnet.yaml \
    --run-dir runs/navsim-lidar-only
```

GPU 6장을 다 쓰려면 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` 와 `-np 6`.
`data.samples_per_gpu` 기본값은 4이므로 `-np 6` 이면 global batch 24가 된다.

### 2. stage 2 — camera + LiDAR (distillation teacher)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchpack dist-run -np 2 python tools/train.py \
    configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
        pretrained/swint-nuimages-pretrained.pth \
    --load_from runs/navsim-lidar-only/latest.pth \
    --run-dir runs/navsim-fusion
```

`pretrained/swint-nuimages-pretrained.pth` 는 이미 받아져 있다 (`tools/download_pretrained.sh`).
8개 카메라를 다 쓰려면 config를 `swint_convfuser_8cam.yaml` 로 바꾼다.

### 3. camera only (LiDAR 없이)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchpack dist-run -np 2 python tools/train.py \
    configs/navsim/det/transfusion/secfpn/camera/swint.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
        pretrained/swint-nuimages-pretrained.pth \
    --run-dir runs/navsim-camera-only
```

### 4. 평가

```bash
CUDA_VISIBLE_DEVICES=0,1 torchpack dist-run -np 2 python tools/test.py \
    runs/navsim-fusion/configs.yaml \
    runs/navsim-fusion/latest.pth \
    --eval bbox
```

체크포인트 옆에 덤프된 `configs.yaml` 을 쓰면 학습 당시 설정이 그대로 재현된다.

### 5. teacher 캐시 생성 (distillation용)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchpack dist-run -np 2 python tools/cache_teacher.py \
    configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \
    runs/navsim-fusion/latest.pth \
    --cache-dir data/teacher_cache --split train
```

### 6. smoke test (환경 확인용)

문서 7절의 `tools/make_navsim_smoke_data.py` 는 `navsim_map_infos_*.pkl` 을 요구하는데
이 서버에는 없다. 대신 `navsim_converter.py` 로 만든 **실제 데이터** smoke를
`data/navsim_smoke/` 에 두고 기존 `smoke.yaml` 을 그대로 쓴다
(config 기본값이 1920px이 된 뒤로는 별도 config가 필요 없다).

```bash
CUDA_VISIBLE_DEVICES=0 torchpack dist-run -np 1 python tools/train.py \
    configs/navsim/det/transfusion/secfpn/camera+lidar/smoke.yaml \
    --run-dir runs/navsim-smoke
```

---

## 설치 검증 결과

```
torch           2.8.0+cu128   CUDA 12.8   arch_max=sm_120   gpus=6
torchvision     0.23.0+cu128
mmcv-full       1.7.2  (소스 빌드)
mmdet           2.28.2
numpy           1.23.5
numba           0.56.4
Pillow          9.5.0
opencv-python   4.9.0.80
yapf            0.40.1
Shapely         1.8.5.post1
nuscenes-devkit 1.1.11
torchpack       0.3.1
mpi4py          3.1.4  (conda-forge)
setuptools      59.5.0
```

- `pip check` → No broken requirements found
- CUDA extension 13/13 빌드, `sm_120.cubin` 내장 확인
- op 실행 확인: `Voxelization`, `iou3d.nms_gpu`, `bev_pool`, spconv forward+backward, mmcv `nms`
- config 4종 모델 빌드 + GPU 이동 OK
  - `lidar/voxelnet` 8.3M / `camera/swint` 35.6M
  - `camera+lidar/swint_convfuser` 40.6M / `swint_convfuser_8cam` 40.6M
- smoke test **1 GPU**: exit 0, loss 255.58 → 44.12 (32 iter), `epoch_1.pth` 저장, NAVSIM evaluator 동작
- smoke test **2 GPU**: exit 0, loss 269.41 → 114.93, 체크포인트 저장, traceback 0건
- 1920px 기본값 전환 후 재실행(override 없음): exit 0, loss 255.32 → 43.78 — override를 쓰던 때와
  사실상 동일하므로 기본값이 올바르게 반영됨
- 실제 변환 데이터로 dataset 구성 확인: train 103,288 / val 1,192,
  샘플 `img (3, 3, 256, 704)`, `points (233262, 5)` (keyframe + 2 sweeps)

smoke의 mAP 0.0은 정상이다 — 64프레임 x 1 epoch x random init이며, 문서도
"The metrics it reports are meaningless" 라고 명시한다.

---

## 전방 전용 BEV로 전환 (설계 변경)

학습 시간 때문에 360° 인코딩을 버리고 전방만 계산하도록 바꿨다.

| 항목 | before | after |
|---|---|---|
| `point_cloud_range` | `[-32,-32,-3, 32,32,5]` | `[0,-32,-3, 32,32,5]` |
| `voxel_size` | `[0.08, 0.08, 0.2]` | `[0.08, 0.08, 0.2]` (등방 유지) |
| `bev_roi` | `[0,-32,32,32]` | `null` |
| `camera_fov.apply_within` | `${bev_roi}` | `null` |
| `xbound` (LSS) | `[-32, 32, 0.32]` | `[0, 32, 0.16]` |
| `sparse_shape` / `grid_size` / `out_size_factor` | `[800,800,41]` / 8 | 그대로 |

`point_cloud_range` 에서 후방을 빼고 `sparse_shape`/`grid_size` 를 `[400, 800, 41]` 로 줄이면
out_size_factor 8에서 **50 x 100 격자, 0.64 m 정사각 셀**이 된다.

> 이력: 한때 `voxel_size [0.04, 0.08, 0.2]` + `sparse_shape [800,800,41]` 로 100 x 100
> (0.32 m 종 x 0.64 m 횡)을 썼다. PARA-SSR student(`bev_h=bev_w=100`)와 셀 단위로 일치시켜
> distillation을 단순 복사로 만들려던 것이다. 이후 정사각 셀로 전환하면서 종축이 student의
> 절반 해상도가 되었으므로, distillation 시 종축 50 -> 100 resample이 필요하거나
> student를 50 x 100 으로 옮겨야 한다.

### 왜 이전 구조는 시간이 안 줄었나

`bev_roi` 마스크는 `dense_heatmap` 의 detach 사본에만 곱해져 **proposal 선택만** 제한했다
(`transfusion.py:329`). 마스킹은 voxelize와 sparse conv **이후**에 일어나므로 후방 연산은
그대로 다 했다. 실제로 비용을 줄이려면 `point_cloud_range` 에서 후방을 빼는 수밖에 없다.

### 실측 효과

| | 360° | 전방 전용 |
|---|---|---|
| points/sample | 225k | 114k |
| voxels (B=4) | 381,699 | 244,095 |
| iter (B=4, 1 GPU, fp32) | 695 ms | **448 ms** (1.55x) |
| 실학습 s/iter (2 GPU) | 0.548 | **0.39** |
| epoch (2 GPU) | 11h 46m | **8h 18m** |

학습 동역학 확인 (2 GPU, 1500 iter): loss 687 -> 2.97, `matched_ious` 0.010 -> 0.290.

### 대가와 주의점

- **후방 supervision 상실**: 라벨의 약 46%가 사라진다. 후방 포인트가 sparse conv의
  receptive field로 전방 feature에 기여하던 것도 없어진다.
- **단순 크롭이 아니다**: voxel x가 0.08 -> 0.04이라 전방 해상도가 2배가 된다. 8x downsample
  후 receptive field가 전방으로 미터 기준 절반이 되고, heatmap 타깃의 `min_radius: 2` /
  `gaussian_overlap: 0.1` 은 셀 단위라 Gaussian이 전방으로 물리적으로 좁아진다. 튜닝 여지.
- **loss 절대값이 이전과 비교 불가**: `avg_factor` 가 달라진다. 스케일 차이일 뿐이다.
- **sweep 커버리지는 무사하다**: `PointsRangeFilter` 가 `LoadPointsFromMultiSweeps` **뒤**라
  각 sweep은 360° 전체로 현재 프레임에 변환된 후 잘린다. 디스크의 `.pcd` 는 360°로 유지할 것.
- 정사각 셀 ablation: `voxel_size [0.08,0.08,0.2]` + `sparse_shape`/`grid_size` `[400,800,41]`
  -> 50 x 100 @ 0.64 m. student 쪽 주석도 같은 ablation을 예고하고 있다.

---

## 전방 카메라 FOV 필터 (기본 OFF)

GT 판정은 **SafeDrive와 동일**하게 맞췄다. SafeDrive의 `boxes_in_bev_roi` 는
박스 **중심**이 사각형 `0 <= x <= 32`, `-32 <= y <= 32` 안이면 유지하고 가시성 검사는 하지 않는다.
`point_cloud_range` + `ObjectRangeFilter` 가 이미 그것과 동일하다 (`in_range_bev` 도 중심 기준).
실제로 파이프라인 GT 수가 SafeDrive predicate와 정확히 일치함을 확인했다 (150 프레임, 3067 = 3067).

SafeDrive의 투영 테스트(`point_sampling`)는 BEV **feature** 마스킹용이지 라벨을 건드리지 않는다.
BEVFusion은 LSS가 이미 카메라가 보는 곳에만 feature를 올리므로 대응물이 이미 있다.

실험용으로 라벨 단위 필터를 남겨뒀다 (`camera_fov`, 기본 `enabled: false`).
박스 8개 코너를 `lidar2image` 로 투영해 하나라도 이미지 안에 양의 깊이로 찍히면 유지한다
(`mmdet3d/datasets/fov_utils.py`).

카메라 3대 실측 방위각 (CAM_F0 63.7°@yaw 0.2, L0 @+55.2, R0 @-55.5) 합집합은 **-87.3 ~ +87.0°**.
켰을 때 ROI 내 박스의 5.2%가 제거되는데, **근거리에 집중**된다 (navtrain 400 프레임):

| 거리 | 박스 수 | 제거율 |
|---|---|---|
| 2-5 m | 153 | 28.8% |
| 5-10 m | 762 | 16.1% |
| 10-20 m | 2486 | 4.1% |
| 20-32 m | 3255 | 2.8% |

전방 카메라가 발밑을 못 보기 때문이고, LiDAR는 그 객체들을 완벽히 본다. camera+LiDAR teacher에서
이걸 끄는 이유다.

**켠다고 student와 같아지지도 않는다.** PARA-SSR은 `det_fov_half_angle_deg: 80` 의
중심 기준 각도 wedge를 쓴다. 두 predicate는 ROI 내 박스의 93.1%에서 일치하고 2.9%에서 갈린다
(student만 유지 0.7%, teacher만 유지 2.3%). student와 정확히 맞추려면 투영이 아니라 wedge 모드가 필요하다.

`camera_fov` 는 두 곳이 함께 참조한다: 학습 파이프라인의 `ObjectFOVFilter`(증강 전 배치),
그리고 `eval_cfg.camera_fov`(평가 GT와 **예측** 양쪽). `load_gt` 와 `_format_bbox` 는 파이프라인을
타지 않으므로 별도 적용이 필요하고, GT만 거르면 FOV 밖 예측이 근거 없는 FP가 된다.
`camera_fov.image_wh` 는 반드시 실제 저장 해상도로 둘 것 — NAVSIM 주점이 수직으로 중심이 아니라
(`cy=560`, 실제 1080) intrinsics로 유도하면 높이가 40px 과대평가된다.

---

## 증강: 전방 전용 ROI에서 반드시 꺼야 하는 것

`RandomFlip3D` 의 `vertical` 은 이미지 상하 반전이 아니라 **좌표계 x -> -x, 즉 전후 반전**이다.
클래스 docstring이 "Disable this whenever the point cloud only covers one half of the BEV"
라고 명시하는데, config가 인자 없이 써서 기본 True로 켜져 있었다.

전방 전용 ROI에서 이게 켜져 있으면 원래 후방이던 장면이 전방 ROI로 들어온다. FOV 필터까지 켠 상태로
측정하면 타깃 유지율이 **41%** 까지 떨어지고 GT가 완전히 빈 프레임이 급증했다 (150 중 50~83개).
빈 프레임은 mmdet3d의 `prepare_train_data` 가 `_rand_another` 로 **무작위 다른 프레임으로 대체**하므로
조용히 학습 분포가 망가진다.

`vertical: false` 적용 후 유지율은 94.8%로 회복되고 빈 프레임도 baseline과 같아진다.

회전은 `[-0.7854, 0.7854]`(±45°)에서 `[-0.1745, 0.1745]`(**±10°**)로 줄였다. ROI는 방위각 180° 폭이고
카메라 커버리지는 174°라, θ 회전 시 ROI의 `(|θ|+3)/180` 이 카메라 사각지대가 된다 (±45°면 27%).
그 영역은 카메라 feature 자체가 비어서 테스트 때 없는 분포를 학습하게 된다.
다만 flip을 고친 뒤 측정하면 ±45°에서도 유지율 88~89%로 ±10°(94.4%)와 차이가 5pp 남짓이므로,
**±10°는 수학적 최적값이 아니라 경험적 절충값**이다.

> 측정 주의: `ds[i]` 로 세면 `_rand_another` 때문에 "필터를 켰는데 박스가 늘어나는" 값이 나온다.
> `ds.pipeline(ds.get_data_info(i))` 로 직접 호출해 우회하고, 증강 난수를 고정해 짝지어 비교할 것.

---

## 실행 시 참고사항

- **test split 미확보**: 현재 `trainval` 만 있다. 공식 navtest(12,146 tokens) 평가에는 navsim 쪽
  `download_test.sh` 가 필요하다 — HuggingFace에서 tarball 64개(카메라 32 + LiDAR 32)를 받는다.
  trainval sensor blob이 446G인 점을 감안하면 test는 대략 50~80G 규모로 예상되고, 디스크 여유는 2.3T다.
  받고 나면 위 0번의 navtest 명령으로 `navsim_infos_val.pkl` 을 만들어 심볼릭 링크를 교체하면 된다.
- **학습은 val info 없이 시작조차 안 된다**: `tools/train.py` 가 `validate=True` 를 무조건 넘기고
  `evaluation.interval: 1` 이라, `cfg.data.val` 이 학습 시작 시점에 build된다.
- **mmcv 재설치 시**: `python tools/patch_mmcv_for_torch2.py` 를 반드시 다시 실행.
- **extension 재빌드 시**: `TORCH_CUDA_ARCH_LIST=12.0` 이 반드시 설정돼 있어야 한다.
  없으면 `setup.py` 가 sm_70~86 기본값으로 돌아가 5090에서 실행 불가한 `.so` 가 나온다.
- **`--check-files` 는 sweep LiDAR 경로를 검사하지 않는다.** keyframe 이미지/LiDAR만 본다.
  scene filter를 쓰면 실무상 문제 없지만, 필터 없이 변환할 때는 별도 확인이 필요하다.
- **종료 시 경고**: `destroy_process_group() was not called before program exit` 는
  torchpack/mmcv 조합의 teardown 노이즈로 무해하다.
- **GPU 점유**: 작업 시점 기준 6장 모두 유휴. 학습 전 `nvidia-smi` 확인 권장.
