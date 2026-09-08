"""NAVSIM (OpenScene / nuPlan) -> BEVFusion info converter.

Reads the raw NAVSIM log pickles (``{split}_navsim_logs/<log_name>.pkl``) and
emits a single info pickle consumable by
:class:`mmdet3d.datasets.NavsimDataset`.

Input (one entry per frame inside each log pkl)::

    token, frame_idx, timestamp, log_name, scene_token, map_location,
    roadblock_ids, traffic_lights, driving_command, ego_dynamic_state,
    ego2global_translation (3,), ego2global_rotation (4, wxyz),
    lidar_path (str, relative to the sensor-blob root),
    cams: {CAM_*: {data_path, sensor2lidar_rotation (3,3),
                   sensor2lidar_translation (3,), cam_intrinsic (3,3),
                   distortion (5,)}},
    anns: {gt_boxes (N,7), gt_names (N,), gt_velocity_3d (N,3),
           instance_tokens, track_tokens}

Output (list of dicts)::

    token, scene_token, log_name, frame_idx, timestamp, location,
    in_split_tokens, lidar_path, sweeps,
    lidar2ego_translation/-rotation, ego2global_translation/-rotation,
    cams: {CAM_*: {data_path, sensor2lidar_rotation,
                   sensor2lidar_translation, cam_intrinsic}},
    gt_boxes (N,7), gt_names, gt_velocity (N,2), num_lidar_pts, valid_flag,
    ego_dynamic_state, driving_command

Box convention
--------------
NAVSIM ``gt_boxes`` are ``[x, y, z_center, length, width, height, heading]``
in the ego (== LiDAR) frame.  They are rewritten to the mmdet3d LiDAR box
convention used throughout BEVFusion::

    [x, y, z_center, width, length, height, -heading - pi/2]

Camera intrinsics
-----------------
NAVSIM images are 1920x1080.  If the sensor blobs on disk were down-scaled
(e.g. to 800x450), pass ``--image-width`` so that ``cam_intrinsic`` is scaled
to match the stored images.

Usage
-----
    python tools/data_converter/navsim_converter.py \
        --navsim-logs   /path/to/openscene/trainval_navsim_logs \
        --sensor-root   /home/data/sensor_blobs_800x450 \
        --split         trainval \
        --image-width   800 \
        --scene-filter  /home/data/navsim_resmap/navsim_cfg/train_test_split/scene_filter/navtrain.yaml \
        --out           data/infos/navsim_infos_train.pkl
"""

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from os import path as osp
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

ORIGINAL_IMAGE_WIDTH = 1920
ORIGINAL_IMAGE_HEIGHT = 1080

NAVSIM_CAMERAS = [
    "CAM_F0",
    "CAM_L0",
    "CAM_R0",
    "CAM_L1",
    "CAM_R1",
    "CAM_L2",
    "CAM_R2",
    "CAM_B0",
]


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def _candidate_rel_paths(raw_path: str, split: str) -> List[str]:
    """Candidate paths relative to the sensor-blob root.

    Different NAVSIM releases store ``data_path`` either with or without the
    leading split directory, and sometimes as an absolute path.
    """
    raw = str(raw_path).replace("\\", "/").lstrip("/")
    cands = [raw]
    if not raw.startswith(split + "/"):
        cands.append(f"{split}/{raw}")
    # Absolute paths from another machine: keep the tail after 'sensor_blobs/'.
    if "sensor_blobs/" in raw:
        tail = raw.split("sensor_blobs/")[-1]
        cands.append(tail)
        if not tail.startswith(split + "/"):
            cands.append(f"{split}/{tail}")
    # De-duplicate, preserving order.
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_rel_path(raw_path: str, sensor_root: str, split: str) -> Optional[str]:
    """Return the sensor-root-relative path that actually exists on disk."""
    for cand in _candidate_rel_paths(raw_path, split):
        if osp.exists(osp.join(sensor_root, cand)):
            return cand
    return None


# --------------------------------------------------------------------------- #
# Scene filtering (mirrors navsim.common.dataloader.filter_scenes)
# --------------------------------------------------------------------------- #
def compute_split_tokens(
    frames: List[Dict[str, Any]],
    num_history_frames: int = 4,
    num_future_frames: int = 10,
    frame_interval: int = 1,
    has_route: bool = True,
) -> set:
    """Tokens that are the reference frame of a valid NAVSIM scene.

    This reproduces the official navtrain / navtest token set, i.e. the frames
    an agent is actually trained and evaluated on.
    """
    num_frames = num_history_frames + num_future_frames
    tokens = set()
    for start in range(0, len(frames), frame_interval):
        chunk = frames[start : start + num_frames]
        if len(chunk) < num_frames:
            continue
        ref = chunk[num_history_frames - 1]
        if has_route and len(ref.get("roadblock_ids", [])) == 0:
            continue
        tokens.add(ref["token"])
    return tokens


def load_scene_filter(scene_filter_yaml: Optional[str]):
    """Return ``(log_names, tokens)`` from a navsim scene_filter yaml.

    The official navtrain / navtest filters ship an explicit ``tokens`` list
    (103,288 and 12,146 entries respectively).  When it is present it defines
    the split exactly, and :func:`compute_split_tokens` is not needed.
    """
    if scene_filter_yaml is None:
        return None, None
    import yaml

    with open(scene_filter_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("log_names")
    tokens = cfg.get("tokens")
    return (set(names) if names else None), (set(tokens) if tokens else None)


# --------------------------------------------------------------------------- #
# Per-frame conversion
# --------------------------------------------------------------------------- #
def convert_boxes(anns: Dict[str, Any]):
    """NAVSIM annotations -> (gt_boxes, gt_names, gt_velocity)."""
    boxes = np.asarray(anns["gt_boxes"], dtype=np.float32).reshape(-1, 7)
    names = np.asarray(anns["gt_names"]).reshape(-1)

    if len(boxes) == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            names.astype("<U32"),
            np.zeros((0, 2), dtype=np.float32),
        )

    xyz = boxes[:, 0:3]
    length, width, height = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    heading = boxes[:, 6]

    # mmdet3d LiDAR boxes: dims = (dx, dy, dz) = (width, length, height)
    # with yaw = -heading - pi/2  (same convention as the nuScenes converter).
    dims = np.stack([width, length, height], axis=-1)
    # NAVSIM headings are not wrapped (they reach +-2*pi); normalise to
    # [-pi, pi) so the stored boxes stay readable and comparable.
    yaw = -heading - np.pi / 2
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    gt_boxes = np.concatenate([xyz, dims, yaw[:, None]], axis=-1).astype(np.float32)

    velocity = np.asarray(anns.get("gt_velocity_3d"), dtype=np.float32)
    if velocity.size == 0:
        gt_velocity = np.zeros((len(boxes), 2), dtype=np.float32)
    else:
        gt_velocity = np.nan_to_num(velocity.reshape(-1, 3)[:, :2], nan=0.0)

    return gt_boxes, names.astype("<U32"), gt_velocity.astype(np.float32)


def ego_pose(frame: Dict[str, Any]):
    """(R, t) of the frame's ego (== LiDAR) pose in global coordinates."""
    from pyquaternion import Quaternion

    R = Quaternion(np.asarray(frame["ego2global_rotation"], dtype=np.float64)).rotation_matrix
    t = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    return R, t


def build_sweeps(
    frames: List[Dict[str, Any]],
    index: int,
    max_sweeps: int,
    split: str,
    lidar_prefix: str,
    lidar_ext: str,
) -> List[Dict[str, Any]]:
    """Previous LiDAR frames expressed in the current frame, nuScenes-style.

    BEVFusion's ``LoadPointsFromMultiSweeps`` expects each sweep to carry a
    ``sensor2lidar_rotation`` / ``sensor2lidar_translation`` that maps its
    points into the *current* keyframe, plus a timestamp for the 5th point
    channel.  nuScenes fills this from the 20 Hz non-keyframe LiDAR records;
    NAVSIM has no intermediate LiDAR, so a "sweep" is the previous 2 Hz frame
    of the same log.  The transform is built the same way either way::

        p_cur = R_cur^T R_prev p_prev + R_cur^T (t_prev - t_cur)

    NOTE the temporal scale: 10 nuScenes sweeps span 0.5 s, 10 NAVSIM frames
    span 5 s (median 2.6 m of ego motion each). Accumulating that far smears
    moving objects; see docs/NAVSIM.md for the measured coverage per sweep.
    """
    if max_sweeps <= 0:
        return []

    R_cur, t_cur = ego_pose(frames[index])
    R_cur_T = R_cur.T
    sweeps = []
    for k in range(1, max_sweeps + 1):
        j = index - k
        if j < 0:
            break
        prev = frames[j]
        if not prev.get("lidar_path"):
            break
        R_prev, t_prev = ego_pose(prev)

        rel = str(prev["lidar_path"]).replace("\\", "/").lstrip("/")
        if "sensor_blobs/" in rel:
            rel = rel.split("sensor_blobs/")[-1]
        if not rel.startswith(split + "/"):
            rel = f"{split}/{rel}"

        sweeps.append(dict(
            data_path=osp.join(lidar_prefix, osp.splitext(rel)[0] + lidar_ext),
            timestamp=int(prev["timestamp"]),
            sensor2lidar_rotation=(R_cur_T @ R_prev).astype(np.float32),
            sensor2lidar_translation=(R_cur_T @ (t_prev - t_cur)).astype(np.float32),
        ))
    return sweeps


def convert_frame(
    frame: Dict[str, Any],
    log_name: str,
    split: str,
    sensor_root: str,
    intrinsic_scale: float,
    sensor_prefix: str,
    lidar_prefix: str,
    lidar_ext: str,
    camera_names: List[str],
    check_files: bool,
    in_split: bool,
    sweeps: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    cams: Dict[str, Dict[str, Any]] = {}
    for cam_name in camera_names:
        cam = frame["cams"].get(cam_name)
        if cam is None:
            return None

        rel = resolve_rel_path(cam["data_path"], sensor_root, split)
        if rel is None:
            if check_files:
                return None
            # Fall back to the recorded path so the pkl stays usable.
            rel = _candidate_rel_paths(cam["data_path"], split)[-1]

        K = np.asarray(cam["cam_intrinsic"], dtype=np.float32).copy()
        K[:2, :] *= intrinsic_scale

        cams[cam_name] = dict(
            data_path=osp.join(sensor_prefix, rel),
            sensor2lidar_rotation=np.asarray(
                cam["sensor2lidar_rotation"], dtype=np.float32
            ),
            sensor2lidar_translation=np.asarray(
                cam["sensor2lidar_translation"], dtype=np.float32
            ),
            cam_intrinsic=K,
        )

    # LiDAR: point at the converted .bin, mirroring the raw pcd layout.
    lidar_path = None
    raw_lidar = frame.get("lidar_path")
    if raw_lidar:
        rel = str(raw_lidar).replace("\\", "/").lstrip("/")
        if "sensor_blobs/" in rel:
            rel = rel.split("sensor_blobs/")[-1]
        if not rel.startswith(split + "/"):
            rel = f"{split}/{rel}"
        lidar_path = osp.join(lidar_prefix, osp.splitext(rel)[0] + lidar_ext)

    gt_boxes, gt_names, gt_velocity = convert_boxes(frame["anns"])

    return dict(
        token=frame["token"],
        scene_token=frame.get("scene_token", ""),
        log_name=log_name,
        frame_idx=int(frame.get("frame_idx", 0)),
        timestamp=int(frame["timestamp"]),
        location=frame.get("map_location", frame.get("location", "")),
        in_split_tokens=bool(in_split),
        lidar_path=lidar_path,
        sweeps=sweeps,
        # NAVSIM annotations live in the ego frame and lidar2ego is the identity.
        lidar2ego_translation=[0.0, 0.0, 0.0],
        lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
        ego2global_translation=np.asarray(
            frame["ego2global_translation"], dtype=np.float64
        ).tolist(),
        ego2global_rotation=np.asarray(
            frame["ego2global_rotation"], dtype=np.float64
        ).tolist(),
        cams=cams,
        gt_boxes=gt_boxes,
        gt_names=gt_names,
        gt_velocity=gt_velocity,
        num_lidar_pts=np.full(len(gt_boxes), -1, dtype=np.int32),
        valid_flag=np.ones(len(gt_boxes), dtype=bool),
        ego_dynamic_state=np.asarray(
            frame.get("ego_dynamic_state", [0.0] * 4), dtype=np.float32
        ),
        driving_command=np.asarray(
            frame.get("driving_command", [0] * 4), dtype=np.float32
        ),
    )


def convert_log(args_tuple):
    (log_path, split, sensor_root, intrinsic_scale, sensor_prefix, lidar_prefix,
     lidar_ext, camera_names, check_files, keyframes_only, filter_kwargs,
     official_tokens, max_sweeps) = args_tuple

    log_name = osp.basename(log_path).replace(".pkl", "")
    try:
        with open(log_path, "rb") as f:
            frames = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        return log_name, [], f"failed to read: {exc}"

    if official_tokens is not None:
        split_tokens = official_tokens
    else:
        split_tokens = compute_split_tokens(frames, **filter_kwargs)

    infos, dropped = [], 0
    for idx, frame in enumerate(frames):
        in_split = frame["token"] in split_tokens
        if keyframes_only and not in_split:
            continue
        sweeps = build_sweeps(
            frames, idx, max_sweeps, split, lidar_prefix, lidar_ext
        )
        info = convert_frame(
            frame, log_name, split, sensor_root, intrinsic_scale, sensor_prefix,
            lidar_prefix, lidar_ext, camera_names, check_files, in_split, sweeps,
        )
        if info is None:
            dropped += 1
            continue
        infos.append(info)

    msg = f"dropped {dropped} frames (missing images)" if dropped else None
    return log_name, infos, msg


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="NAVSIM -> BEVFusion info converter")
    p.add_argument("--navsim-logs", required=True,
                   help="directory of raw NAVSIM log pickles (<log_name>.pkl)")
    p.add_argument("--sensor-root", required=True,
                   help="root of the sensor blobs (contains trainval/ and test/)")
    p.add_argument("--split", required=True, choices=["trainval", "test", "mini"])
    p.add_argument("--out", required=True, help="output .pkl path")
    p.add_argument("--image-width", type=int, default=800,
                   help="width of the stored images; intrinsics are scaled by "
                        "image-width / 1920 (default: 800)")
    p.add_argument("--scene-filter", default=None,
                   help="navsim scene_filter yaml, used only for its log_names list")
    p.add_argument("--sensor-prefix", default="sensor_blobs",
                   help="prefix written into data_path, relative to dataset_root")
    p.add_argument("--lidar-prefix", default="lidar_bin",
                   help="prefix written into lidar_path, relative to dataset_root")
    p.add_argument("--lidar-ext", default=".bin",
                   help="extension of the converted point clouds (default: .bin)")
    p.add_argument("--cameras", nargs="+", default=NAVSIM_CAMERAS,
                   help="cameras to record in the info file")
    p.add_argument("--keyframes-only", action="store_true",
                   help="keep only frames in the official navtrain/navtest token "
                        "set (recommended; ~2x smaller)")
    p.add_argument("--check-files", action="store_true",
                   help="drop frames whose images are missing on disk")
    p.add_argument("--max-sweeps", type=int, default=10,
                   help="previous LiDAR frames recorded per sample, matching "
                        "nuScenes' max_sweeps=10. How many the pipeline then "
                        "uses is `sweeps_num` in the config (default: 10)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-logs", type=int, default=None,
                   help="debug: only convert the first N logs")
    # Scene-filter parameters (defaults match navtrain/navtest).
    p.add_argument("--num-history-frames", type=int, default=4)
    p.add_argument("--num-future-frames", type=int, default=10)
    p.add_argument("--frame-interval", type=int, default=1)
    p.add_argument("--no-route-filter", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    log_files = sorted(
        osp.join(args.navsim_logs, f)
        for f in os.listdir(args.navsim_logs)
        if f.endswith(".pkl")
    )
    keep, official_tokens = load_scene_filter(args.scene_filter)
    if keep is not None:
        log_files = [f for f in log_files
                     if osp.basename(f).replace(".pkl", "") in keep]
    if official_tokens is not None:
        print(f"[navsim_converter] using the official token list "
              f"({len(official_tokens)} tokens) from {args.scene_filter}")
    if args.max_logs:
        log_files = log_files[: args.max_logs]

    intrinsic_scale = args.image_width / ORIGINAL_IMAGE_WIDTH
    print(f"[navsim_converter] {len(log_files)} logs, split={args.split}")
    print(f"[navsim_converter] intrinsic scale = {intrinsic_scale:.6f} "
          f"({ORIGINAL_IMAGE_WIDTH} -> {args.image_width})")

    filter_kwargs = dict(
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        frame_interval=args.frame_interval,
        has_route=not args.no_route_filter,
    )
    jobs = [
        (f, args.split, args.sensor_root, intrinsic_scale, args.sensor_prefix,
         args.lidar_prefix, args.lidar_ext, args.cameras, args.check_files,
         args.keyframes_only, filter_kwargs, official_tokens, args.max_sweeps)
        for f in log_files
    ]

    all_infos, warnings = [], []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for log_name, infos, msg in tqdm(
                pool.map(convert_log, jobs), total=len(jobs), desc="logs"
            ):
                all_infos.extend(infos)
                if msg:
                    warnings.append(f"{log_name}: {msg}")
    else:
        for job in tqdm(jobs, desc="logs"):
            log_name, infos, msg = convert_log(job)
            all_infos.extend(infos)
            if msg:
                warnings.append(f"{log_name}: {msg}")

    for w in warnings[:20]:
        print(f"[warn] {w}")
    if len(warnings) > 20:
        print(f"[warn] ... and {len(warnings) - 20} more")

    n_split = sum(1 for i in all_infos if i["in_split_tokens"])
    n_sweeps = np.mean([len(i["sweeps"]) for i in all_infos]) if all_infos else 0
    n_boxes = sum(len(i["gt_boxes"]) for i in all_infos)
    print(f"[navsim_converter] mean {n_sweeps:.2f} sweeps/frame")
    print(f"[navsim_converter] {len(all_infos)} frames "
          f"({n_split} in official token set), {n_boxes} boxes")

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(all_infos, f, protocol=4)
    print(f"[navsim_converter] wrote {args.out}")


if __name__ == "__main__":
    main()
