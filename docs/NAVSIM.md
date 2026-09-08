# BEVFusion on NAVSIM

This fork trains BEVFusion on [NAVSIM](https://github.com/autonomousvision/navsim)
(OpenScene / nuPlan) so it can act as the **BEV distillation teacher** for an
end-to-end planner. It mirrors the existing Bench2Drive port: a dataset class,
a converter, and a config tree, with no changes to the detector itself beyond
two small additions noted at the bottom.

---

## 1. Environment

Built and verified with conda (`conda-forge` only, so Anaconda's default
channels and their ToS never come into play):

```bash
conda create -y -n bevfusion -c conda-forge --override-channels python=3.8
conda activate bevfusion

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install "setuptools==59.5.0" "Pillow==8.4.0" tqdm "numpy==1.23.5"
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install "mmdet==2.20.0" "yapf==0.40.1" "numba==0.56.4" nuscenes-devkit \
    torchpack "mpi4py==3.1.4" opencv-python "shapely<2.0" tensorboard

# nvcc 11.3 + gcc 9 (CUDA 11.3 does not accept the system gcc 12) + OpenMPI
conda install -y --override-channels \
    -c nvidia/label/cuda-11.3.1 -c conda-forge \
    cuda-toolkit=11.3.1 gxx_linux-64=9.5.0 gcc_linux-64=9.5.0 openmpi=4.1

export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST="8.6"      # RTX A6000; add other archs as needed
python setup.py develop
```

`mmdet3d/models/backbones/radar_encoder.py` imports `flash_attn` only for the
radar branch; the import is now optional, so no flash-attn build is required.

---

## 2. Data layout

```
/home/data/
  navsim_logs/
    trainval/<log_name>.pkl          # 1310 logs, per-frame annotations
    test/<log_name>.pkl              #  147 logs
  sensor_blobs_800x450/
    trainval/<log_name>/CAM_{F0,L0,L1,L2,R0,R1,R2,B0}/<sample_token>.jpg
    test/...
  sensor_blobs_lidar_front/                 # front-cropped LiDAR, see section 4
    trainval/<log_name>/MergedPointCloud/<frame_token>.pcd
    test/...
  navsim_resmap/
    navsim_cfg/train_test_split/scene_filter/{navtrain,navtest}.yaml
  maps/                              # nuPlan GPKG maps (not used by detection)
```

Inside the repo:

```
data/navsim/sensor_blobs -> /home/data/sensor_blobs_800x450
data/navsim/lidar        -> /home/data/sensor_blobs_lidar_front
data/infos/navsim_infos_{train,val}.pkl
```

### Image resolution

NAVSIM images are 1920x1080. The stored blobs are 800x450 (exactly 1920/2.4),
which is **sufficient and requires no re-export**: the model input is 256x704,
and 800 -> 704 is only a 1.14x downscale, so almost nothing is lost relative to
feeding the original 1920px frames. `augment2d.resize` in
`configs/navsim/default.yaml` is expressed relative to the *stored* size
(`[0.76, 1.0]` train, `0.88` eval), so no upsampling ever happens.

If you re-export at another width, pass `--image-width` to the converter (it
rescales `cam_intrinsic` by `width / 1920`) and update `augment2d.resize` to
`704 / new_width`.

---

## 3. Building the info files

```bash
SF=/home/data/navsim_resmap/navsim_cfg/train_test_split/scene_filter

python tools/data_converter/navsim_converter.py \
    --navsim-logs /home/data/navsim_logs/trainval \
    --sensor-root /home/data/sensor_blobs_800x450 \
    --split trainval --image-width 800 \
    --scene-filter $SF/navtrain.yaml \
    --keyframes-only --check-files --workers 24 \
    --lidar-prefix lidar --lidar-ext .pcd \
    --out data/infos/navsim_infos_train.pkl

python tools/data_converter/navsim_converter.py \
    --navsim-logs /home/data/navsim_logs/test \
    --sensor-root /home/data/sensor_blobs_800x450 \
    --split test --image-width 800 \
    --scene-filter $SF/navtest.yaml \
    --keyframes-only --check-files --workers 24 \
    --lidar-prefix lidar --lidar-ext .pcd \
    --out data/infos/navsim_infos_val.pkl
```

`--keyframes-only` restricts the output to the official split token lists that
ship inside the scene-filter yamls (103,288 navtrain / 12,146 navtest), which
is exactly the frame set an agent is trained and scored on -- so the teacher
cache covers the student's frames one-for-one.

### Coordinate conventions (verified against the data)

* The ego frame is the **rear axle** and `lidar2ego` is the identity. NAVSIM's
  box `z` is the **centre**, not the bottom; this matches its official camera
  projection code, which builds corners around origin `(0.5, 0.5, 0.5)`.
  Box elevations vary with road grade and object placement. The configured
  front-only range is
  `point_cloud_range = [0.0, -32.0, -3.0, 32.0, 32.0, 5.0]`.
* `cams[*]['sensor2lidar_*']` is camera -> LiDAR. (In the older
  `navsim_map_infos_*.pkl`, the equivalent field is called `extrinsics` and
  points the other way: `sensor2lidar = inv(extrinsics)`.)
* NAVSIM boxes are `[x, y, z_centre, length, width, height, heading]`. The
  converter rewrites them to the mmdet3d LiDAR convention used everywhere
  else in this repo: `[x, y, z_centre, width, length, height,
  -heading - pi/2]`, wrapped to `[-pi, pi)`. `NavsimDataset` therefore builds
  `LiDARInstance3DBoxes(..., origin=(0.5, 0.5, 0.5))`.
* Classes are the seven nuPlan `TrackedObjectType` names: `vehicle`,
  `pedestrian`, `bicycle`, `traffic_cone`, `barrier`, `czone_sign`,
  `generic_object`.

---

## 4. LiDAR and the BEV grid

### Reading the clouds

NAVSIM's `MergedPointCloud/*.pcd` is read **directly** -- there is no
conversion step. The NAVSIM encoding (float32 `x,y,z` + uint8
`intensity,lidar_info,ring` = 15 bytes/point) is denser than a float32 `.bin`
would be (20 bytes/point), so converting would *grow* the dataset by a third.
`LoadPointsFromFile` and `LoadPointsFromMultiSweeps` have a `.pcd` branch
backed by `loading_utils.load_pcd_point_cloud`, which handles the `binary`,
`binary_compressed` (LZF) and `ascii` encodings and selects fields by name.
Install `python-lzf` for a faster compressed path; a pure-python fallback is
used otherwise.

The info files point at the clouds through `lidar_path`, so all that is needed
is a symlink, e.g. `data/navsim/lidar -> <sensor blobs>`, built by running the
converter with `--lidar-prefix lidar --lidar-ext .pcd`.

`tools/data_converter/navsim_lidar_to_bin.py` can still crop the clouds into
`.bin` (`--point-cloud-range`, `--voxel`, `--dims`, `--delete-source`) and
`tools/prune_unused_lidar.py` deletes clouds no info file references (on the
official split that is 112,183 of 227,617 files, ~86 GB).

### The grid: encode 360 degrees, query the forward half

```
point_cloud_range: [-32.0, -32.0, -3.0, 32.0, 32.0, 5.0]
voxel_size:        [0.08, 0.08, 0.2]      -> 800 x 800 x 40 voxels
                                          -> 100 x 100 head grid at 0.64 m
bev_roi:           [0.0, -32.0, 32.0, 32.0]
```

* **The extent is the one every NAVSIM baseline uses.** TransFuser, WoTE and
  SeerDrive all rasterise LiDAR over `lidar_min/max_x/y = +-32`, and PARA-SSR's
  `pc_range` is the same box.
* **The resolution matches PARA-SSR's `bev_h = bev_w = 100`.** At 0.64 m per
  cell the teacher's forward half lands on the student's BEV one-to-one, so
  distillation is a crop rather than a resample.
* **`bev_roi` restricts query initialisation to the forward half** (grid rows
  50-99). The encoder still runs over the full +-32 m -- rear points feed the
  sparse convolutions and the dense heatmap is supervised everywhere -- but
  every proposal, and therefore every cached teacher output, lands in the
  region the planner reads. This is the BEVFusion analogue of WoTE encoding
  +-32 m while rasterising only `x in [0, 32]`. Set `bev_roi: null` to query
  the whole grid.
* `eval_cfg.point_cloud_range` mirrors `bev_roi` so rear GT is not scored
  against a model that was never asked to look there.

This configuration **requires the uncropped 360-degree clouds.** The
front-cropped set (`x in [0, 32]`) that produced
`runs/navsim-fusion/epoch_20.pth` is a different grid; that checkpoint is
reproduced by pointing `tools/test.py` at the config dumped beside it
(`runs/navsim-fusion/configs.yaml`), which torchpack loads standalone.

### Multi-sweep

Upstream BEVFusion accumulates 9 of the 10 previous LiDAR records per
keyframe, transforms each into the keyframe with `sensor2lidar_*` and writes
the time offset into point channel 4. `navsim_converter.py --max-sweeps`
(default 10) fills the same `sweeps` field, using the previous frames of the
log and `R_cur^T R_prev`, `R_cur^T (t_prev - t_cur)` from the ego poses;
`sweeps_num` in the config picks how many are used.

**The temporal scale is not nuScenes'.** nuScenes sweeps are 20 Hz
non-keyframes, so 10 of them span 0.5 s. NAVSIM has no intermediate LiDAR, so
one sweep is already 0.5 s and a median 2.6 m of ego motion (measured over
10,690 navtest frame pairs; p90 speed 9.3 m/s). `sweeps_num: 2` therefore
accumulates roughly the ego displacement nuScenes gets from its full stack.
Going further smears moving objects, and on *cropped* clouds it also opens a
hole: the previous frame's sensed wedge, transformed into the current frame,
covers only

| sweeps back | 1 | 2 | 4 | 6 | 9 |
| --- | --- | --- | --- | --- | --- |
| coverage of the BEV box, cropped clouds | 91% | 81% | 63% | 45% | 19% |
| coverage, 360-degree clouds | 100% | 100% | 100% | 100% | 100% |

so the sweep stack only makes sense on the uncropped data, which is the other
reason this config assumes it.

Note that both the student and the NAVSIM baselines use a *single* LiDAR frame
(`lidar_seq_len: 1`, `agent_input.lidars[-1]`) and take ego velocity through
`status_feature` instead, so sweeps are a teacher-side improvement rather than
something the student needs matched.

---

## 5. Training

Two stages, following the upstream BEVFusion recipe.

```bash
# stage 1 -- LiDAR only
torchpack dist-run -np 2 python tools/train.py \
    configs/navsim/det/transfusion/secfpn/lidar/voxelnet.yaml \
    --run-dir runs/navsim-lidar-only

# stage 2 -- camera + LiDAR (the teacher)
torchpack dist-run -np 2 python tools/train.py \
    configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
        pretrained/swint-nuimages-pretrained.pth \
    --load_from runs/navsim-lidar-only/latest.pth \
    --run-dir runs/navsim-fusion
```

Camera-only (no LiDAR on disk yet) uses the same head and class set, so the
cached teacher tensors have identical shapes either way:

```bash
torchpack dist-run -np 2 python tools/train.py \
    configs/navsim/det/transfusion/secfpn/camera/swint.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
        pretrained/swint-nuimages-pretrained.pth \
    --run-dir runs/navsim-camera-only
```

### Choosing the camera set

`camera_names` in `configs/navsim/default.yaml` picks which of the eight
cameras reach the model, in order:

```yaml
camera_names: [CAM_F0, CAM_L0, CAM_R0]   # NAVSIM's standard front view
```

BEVFusion is camera-count agnostic -- `DepthLSSTransform` lifts each view into
the shared BEV grid using its own intrinsics/extrinsics, so `N` is just a
tensor dimension. The shared BEV itself is front-only for both modalities.
`swint_convfuser_8cam.yaml` reads all eight cameras, but camera features that
land outside the front-only BEV ROI are discarded during BEV pooling.

---

## 6. Caching teacher outputs for distillation

```bash
torchpack dist-run -np 2 python tools/cache_teacher.py \
    configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \
    runs/navsim-fusion/latest.pth \
    --cache-dir data/teacher_cache --split train
```

This writes `teacher_cache.pkl`, a `{frame_token: {heatmap, center, height,
dim, rot, vel}}` map in fp16. The keys are NAVSIM frame tokens, so the student
can look them up directly.

---

## 7. Smoke test

`tools/make_navsim_smoke_data.py` builds a few dozen frames with **real**
images, intrinsics and extrinsics but **synthetic** LiDAR and boxes, so the
whole loop can be exercised without any point clouds on disk:

```bash
python tools/make_navsim_smoke_data.py \
    --map-infos /home/data/navsim_resmap/infos/navsim_map_infos_navtest.pkl \
    --sensor-root /home/data/sensor_blobs_800x450 --split test \
    --num-frames 64 --val-frames 16 --out-dir data/navsim_smoke

torchpack dist-run -np 1 python tools/train.py \
    configs/navsim/det/transfusion/secfpn/camera+lidar/smoke.yaml \
    --run-dir runs/navsim-smoke
```

The metrics it reports are meaningless; what it verifies is that the ops build,
the calibration chain, the losses, checkpointing and the evaluator all work.

---

## 8. Changes to shared code

Three edits outside the `navsim_*` files, all additive:

* `mmdet3d/models/fusion_models/bevfusion.py` -- every sensor argument of
  `forward` / `forward_single` now defaults to `None`, so a config with
  `encoders.camera: null` (or `encoders.lidar: null`) no longer has to load the
  other modality just to satisfy the signature. Stage 1 therefore skips JPEG
  decoding entirely, and the camera-only config runs with no `.bin` on disk.
* `mmdet3d/models/heads/bbox/transfusion.py` -- three changes:
  a `NAVSIM` branch alongside the existing `nuScenes` / `Waymo` /
  `Bench2Drive` ones, disabling local-max NMS for the five small classes;
  the heatmap-target allocation in `get_targets` now follows `(x, y)` so
  non-square BEV grids work; `_nms` no longer zeroes the outer ring of
  `local_max` (that suppressed every peak on the BEV border -- worth +5.1 AP
  on vehicles here, and invisible on nuScenes only because its border sits
  past every `class_range`); and the optional `bev_roi` described in
  section 4.
* `mmdet3d/datasets/pipelines/loading{,_utils}.py` -- a PCD reader, wired into
  both point loaders so NAVSIM's `.pcd` clouds are read in place.
* `mmdet3d/datasets/pipelines/transforms_3d.py` -- `RandomFlip3D` takes
  `horizontal` / `vertical` switches (both default `True`, so the nuScenes and
  Bench2Drive configs are unchanged).
* `mmdet3d/models/backbones/radar_encoder.py` -- the unused `flash_attn`
  import is now optional.
