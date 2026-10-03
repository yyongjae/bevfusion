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

As the ordinary NAVSIM download leaves it -- images and merged point clouds
share one tree per split:

```
<openscene>/
  trainval_navsim_logs/
    trainval/<log_name>.pkl          # 1310 logs, per-frame annotations
  test_navsim_logs/
    test/<log_name>.pkl              #  147 logs
  trainval_sensor_blobs/
    trainval/<log_name>/CAM_{F0,L0,L1,L2,R0,R1,R2,B0}/<sample_token>.jpg
    trainval/<log_name>/MergedPointCloud/<frame_token>.pcd
  test_sensor_blobs/
    test/...
  maps/                              # nuPlan GPKG maps (not used by detection)
```

The `{navtrain,navtest}.yaml` scene filters live in the navsim repo, under
`navsim/planning/script/config/common/train_test_split/scene_filter/`.

Inside this repo:

```
data/navsim/sensor_blobs -> <openscene>/trainval_sensor_blobs
data/navsim/lidar        -> <openscene>/trainval_sensor_blobs
data/infos/navsim_infos_{train,val}.pkl
```

Both links point at the same tree unless the clouds were exported separately
(e.g. the front-cropped set described in section 4), in which case `lidar`
points at that export instead.

### Image resolution

NAVSIM images are 1920x1080, and that is what the converter and the configs
now assume. `augment2d.resize` in `configs/navsim/default.yaml` is expressed
relative to the *stored* size, so at 1920 px it reads `[0.31667, 0.41667]`
train and `0.36667` eval (`704 / 1920`), and no upsampling ever happens.

An earlier revision of this port ran against an 800x450 re-export with
`[0.76, 1.0]` / `0.88`. **The two produce identical output sizes** -- 608 px at
the low end of the training range, 800 px at the high end, 704 px at eval --
so only the source detail differs, and the 800px numbers remain correct for
anyone still using that export.

| stored width | train resize | eval resize | resulting width |
| --- | --- | --- | --- |
| 1920 (default) | `[0.31667, 0.41667]` | `0.36667` | 608 - 800 px, 704 at eval |
| 800 | `[0.76, 1.0]` | `0.88` | the same |

If you re-export at another width, pass `--image-width` to the converter (it
rescales `cam_intrinsic` by `width / 1920`) and set `augment2d.resize` eval to
`704 / new_width`, keeping the training range at the same ratio to it.

---

## 3. Building the info files

```bash
SF=<navsim>/navsim/planning/script/config/common/train_test_split/scene_filter

python tools/data_converter/navsim_converter.py \
    --navsim-logs <openscene>/trainval_navsim_logs/trainval \
    --sensor-root <openscene>/trainval_sensor_blobs \
    --split trainval \
    --scene-filter $SF/navtrain.yaml \
    --keyframes-only --check-files --workers 24 \
    --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 \
    --out data/infos/navsim_infos_train.pkl

python tools/data_converter/navsim_converter.py \
    --navsim-logs <openscene>/test_navsim_logs/test \
    --sensor-root <openscene>/test_sensor_blobs \
    --split test \
    --scene-filter $SF/navtest.yaml \
    --keyframes-only --check-files --workers 24 \
    --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 \
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

### The grid: front only

```
point_cloud_range: [0.0, -32.0, -3.0, 32.0, 32.0, 5.0]
voxel_size:        [0.08, 0.08, 0.2]      -> 400 x 800 x 40 voxels
                                          -> 50 x 100 head grid
                                          -> 0.64 m square cells
bev_roi:           null
```

* **The ROI is the region the cameras can see.** A three-front-camera teacher
  has no rear observation, so encoding `x in [-32, 0]` bought supervision the
  student cannot use and cost roughly half the LiDAR compute: points per sample
  drop 225k -> 114k and a fp32 iteration (B=4, one RTX 5090) drops 695 -> 448 ms.
* **The cells are square.** `voxel_size` is isotropic at 0.08 m and
  `sparse_shape` / `grid_size` are `[400, 800, 41]`, so with `out_size_factor` 8
  the head grid is 50 x 100 at 0.64 m in both axes. This is the square-cell
  variant the student's config comment anticipates ("see the report for the
  square-cell 50 x 100 ablation that should follow").
* **It no longer matches PARA-SSR cell for cell.** The student's `pc_range` is
  `(-32, 0, -2, 32, 32, 2)` with `bev_h = bev_w = 100`, i.e. the same physical
  box but 0.32 m longitudinal cells. The lateral axes still agree at 0.64 m; the
  longitudinal one does not, so distillation needs a 50 -> 100 resample along it
  unless the student is moved to 50 x 100 as well. The axes also differ in
  convention: mmdet3d LiDAR is x-forward / y-left, the student is y-forward /
  x-right.
* **Reverting** to the 100 x 100 teacher means `voxel_size [0.04, 0.08, 0.2]`,
  `sparse_shape` / `grid_size` `[800, 800, 41]`, and the camera `xbound` step
  back to 0.16.
* **`bev_roi` is no longer needed.** It existed to keep query initialisation in
  the forward half of a 360-degree grid; the grid is now that half. The head
  still supports it, so an ROI inside the front region can be set if wanted.
* `eval_cfg.point_cloud_range` is identical to `point_cloud_range`, so rear GT
  is not scored against a model that was never asked to look there.

> Earlier revisions encoded the full 360 degrees and masked the heatmap with
> `bev_roi` before proposal selection. That kept the rear supervised by LiDAR --
> about 46% of the labels -- but the masking happened *after* voxelisation and
> the sparse convolutions, so none of the rear compute was ever saved. Dropping
> the rear from `point_cloud_range` is what actually makes it cheaper, and the
> cost is that rear supervision.

### Camera field of view

The GT predicate follows SafeDrive's, which is the reference this teacher is
compared against: `boxes_in_bev_roi` keeps a box when its **centre** lies in the
rectangle `0 <= x <= 32`, `-32 <= y <= 32`, with no visibility test at all.
`point_cloud_range` plus `ObjectRangeFilter` already reproduce that exactly --
`in_range_bev` tests the same centre -- so nothing further is applied by
default, and the pipeline's GT count matches SafeDrive's predicate box for box.

SafeDrive's projection test (`point_sampling`) masks BEV *features*, deciding
which BEV queries attend to which camera; it never touches labels. BEVFusion's
LSS already restricts camera features to what the cameras see, so the analogue
is in place.

An optional label-level filter is available for experiments. `camera_fov`
(default `enabled: false`) projects the eight box corners through `lidar2image`
and keeps a box when at least one corner lands in an image with positive depth
-- see `mmdet3d/datasets/fov_utils.py`.  Measured on 400 navtrain frames, the
three front cameras (CAM_F0 63.7 degrees at yaw 0.2, CAM_L0 at +55.2, CAM_R0 at
-55.5) span -87.3 to +87.0 degrees, and the filter removes 5.2% of the in-ROI
boxes:

| distance | boxes | removed |
| --- | --- | --- |
| 2-5 m | 153 | 28.8% |
| 5-10 m | 762 | 16.1% |
| 10-20 m | 2486 | 4.1% |
| 20-32 m | 3255 | 2.8% |

The near field dominates because the forward-facing cameras cannot look down far
enough, and the LiDAR sees those objects perfectly -- which is why this is off
for a camera+LiDAR teacher.

Note that enabling it does **not** reproduce PARA-SSR's label set either. The
student applies `det_fov_half_angle_deg: 80`, a centre-based angular wedge; the
two predicates agree on 93.1% of in-ROI boxes and disagree on 2.9% (0.7% the
student keeps and this drops, 2.3% the reverse). Matching the student exactly
would need a wedge mode, not this projection.

Both places that consume it are driven by the same `camera_fov` block:
`ObjectFOVFilter` in the training pipelines (placed before `ImageAug3D`, so the
test runs on raw sensor geometry and the label set is identical every epoch),
and `eval_cfg.camera_fov`, which masks evaluation GT *and* predictions --
`NavsimDataset.load_gt` and `_format_bbox` never run the pipeline. Masking only
the GT would leave detections outside the region counting as false positives
with no GT to match. `camera_fov.image_wh` must be the stored image size:
NAVSIM's principal point is not centred vertically (cy = 560 on 1080 rows), so
inferring it from intrinsics overestimates the height by 40 px.

**A front-only ROI is not the same thing as front-cropped clouds, and this
configuration still requires the uncropped 360-degree ones.**  `PointsRangeFilter`
runs *after* `LoadPointsFromMultiSweeps`, so each sweep is transformed into the
current frame in full and only the accumulated cloud is cut to
`point_cloud_range`.  Every sweep therefore covers the whole ROI (see the table
below).  Pre-cropping the files on disk -- the `x in [0, 32]` set that produced
`runs/navsim-fusion/epoch_20.pth` -- loses that, and is a different grid besides;
that checkpoint is reproduced by pointing `tools/test.py` at the config dumped
beside it (`runs/navsim-fusion/configs.yaml`), which torchpack loads standalone.

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
Going further smears moving objects, and on clouds *cropped on disk* it also
opens a hole: the previous frame's sensed wedge, transformed into the current
frame, covers only

| sweeps back | 1 | 2 | 4 | 6 | 9 |
| --- | --- | --- | --- | --- | --- |
| coverage of the BEV box, cropped clouds | 91% | 81% | 63% | 45% | 19% |
| coverage, 360-degree clouds | 100% | 100% | 100% | 100% | 100% |

so the sweep stack only makes sense on the uncropped files, which is the other
reason this config reads them and crops afterwards.

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
    --sensor-root <openscene>/trainval_sensor_blobs --split test \
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
* `mmdet3d/datasets/fov_utils.py` (new) -- the projection-based visibility test,
  shared by the pipeline transform and the evaluator.
* `mmdet3d/datasets/pipelines/transforms_3d.py` -- adds `ObjectFOVFilter`.
* `mmdet3d/datasets/navsim_dataset.py` -- `load_gt` applies the same test to the
  evaluation GT via `eval_cfg['camera_fov']`.
