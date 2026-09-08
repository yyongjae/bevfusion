"""Build a tiny synthetic NAVSIM dataset for smoke-testing the pipeline.

The real NAVSIM annotations (``*_navsim_logs``) and the merged point clouds
are large downloads.  This script produces a few hundred frames that exercise
every stage of the training loop *before* that download completes:

* **Real** camera images, intrinsics and extrinsics, taken from the info
  pickles already on disk (``navsim_map_infos_*.pkl``) and the sensor blobs.
  Intrinsics are rescaled from 1920 px to the stored width (``--image-width``,
  1920 by default).
* **Synthetic** LiDAR and 3D boxes: a ground plane plus a handful of
  box-shaped point clusters, with GT boxes that match those clusters, so the
  detection loss is actually learnable and a falling loss curve means the
  geometry conventions line up.

The output is written in exactly the format
:class:`mmdet3d.datasets.NavsimDataset` expects, so the same configs run
unchanged once the real infos replace these.

Usage
-----
    python tools/make_navsim_smoke_data.py \
        --map-infos /home/data/navsim_resmap/infos/navsim_map_infos_navtest.pkl \
        --sensor-root /path/to/openscene/trainval_sensor_blobs \
        --split test --num-frames 64 --out-dir data/navsim_smoke
"""

import argparse
import os
import pickle
from os import path as osp

import numpy as np
from pyquaternion import Quaternion

CLASS_SPECS = [
    # name,           weight, (length, width, height) mean,      size jitter
    ("vehicle",        0.55, (4.6, 1.95, 1.65), 0.25),
    ("pedestrian",     0.20, (0.7, 0.7, 1.75), 0.10),
    ("bicycle",        0.10, (1.7, 0.6, 1.4), 0.10),
    ("traffic_cone",   0.08, (0.4, 0.4, 0.8), 0.05),
    ("barrier",        0.07, (2.0, 0.4, 1.0), 0.10),
]


def sample_boxes(rng, n_objects, max_range=40.0):
    """Return NAVSIM-style boxes [x, y, z_center, l, w, h, heading]."""
    names, boxes, velocities = [], [], []
    weights = np.array([c[1] for c in CLASS_SPECS])
    weights = weights / weights.sum()

    for _ in range(n_objects):
        idx = rng.choice(len(CLASS_SPECS), p=weights)
        name, _, (l, w, h), jitter = CLASS_SPECS[idx]

        # Keep objects off the ego footprint.
        while True:
            x = rng.uniform(-max_range, max_range)
            y = rng.uniform(-max_range, max_range)
            if abs(x) > 3.0 or abs(y) > 2.0:
                break

        l = float(l * (1 + rng.uniform(-jitter, jitter)))
        w = float(w * (1 + rng.uniform(-jitter, jitter)))
        h = float(h * (1 + rng.uniform(-jitter, jitter)))
        heading = float(rng.uniform(-np.pi, np.pi))

        # z = 0 is the ground plane in the NAVSIM ego frame.
        boxes.append([x, y, h / 2.0, l, w, h, heading])
        names.append(name)
        speed = rng.uniform(0, 12) if name == "vehicle" else rng.uniform(0, 1.5)
        velocities.append([speed * np.cos(heading), speed * np.sin(heading)])

    return (
        np.array(boxes, dtype=np.float32).reshape(-1, 7),
        np.array(names, dtype="<U32"),
        np.array(velocities, dtype=np.float32).reshape(-1, 2),
    )


def navsim_to_mmdet3d_boxes(boxes):
    """[x, y, z, l, w, h, heading] -> [x, y, z, w, l, h, -heading - pi/2]."""
    if len(boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    out = np.empty_like(boxes)
    out[:, 0:3] = boxes[:, 0:3]
    out[:, 3] = boxes[:, 4]  # dx = width
    out[:, 4] = boxes[:, 3]  # dy = length
    out[:, 5] = boxes[:, 5]  # dz = height
    out[:, 6] = -boxes[:, 6] - np.pi / 2
    return out.astype(np.float32)


def synth_points(rng, boxes, n_ground=40000, n_per_box=350, pc_range=None):
    """Ground plane + points on each box surface -> (N, 5) float32."""
    x0, y0, z0, x1, y1, z1 = pc_range
    clouds = []

    # Ground: a noisy plane at z ~= 0 with a radial density falloff.
    r = np.sqrt(rng.uniform(1.0, (max(x1, y1) ** 2), size=n_ground))
    theta = rng.uniform(-np.pi, np.pi, size=n_ground)
    ground = np.stack(
        [
            r * np.cos(theta),
            r * np.sin(theta),
            rng.normal(0.0, 0.03, size=n_ground),
            rng.uniform(0.0, 0.3, size=n_ground),
            rng.integers(0, 32, size=n_ground).astype(np.float32),
        ],
        axis=-1,
    )
    clouds.append(ground)

    for box in boxes:
        cx, cy, cz, l, w, h, heading = box
        n = int(n_per_box * max(0.15, 1.0 - np.hypot(cx, cy) / 60.0))
        if n <= 0:
            continue
        local = np.stack(
            [
                rng.uniform(-l / 2, l / 2, size=n),
                rng.uniform(-w / 2, w / 2, size=n),
                rng.uniform(-h / 2, h / 2, size=n),
            ],
            axis=-1,
        )
        # Push points onto the box surface so the cluster has a real shape.
        axis = rng.integers(0, 3, size=n)
        extent = np.array([l / 2, w / 2, h / 2], dtype=np.float32)
        sign = rng.choice([-1.0, 1.0], size=n)
        local[np.arange(n), axis] = sign * extent[axis]

        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        world = local @ rot.T + np.array([cx, cy, cz])
        world += rng.normal(0.0, 0.02, size=world.shape)

        clouds.append(
            np.concatenate(
                [
                    world,
                    rng.uniform(0.2, 1.0, size=(n, 1)),
                    rng.integers(0, 32, size=(n, 1)).astype(np.float32),
                ],
                axis=-1,
            )
        )

    pts = np.concatenate(clouds, axis=0).astype(np.float32)
    m = (
        (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
        & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
    )
    return pts[m]


def resolve_image(raw_path, sensor_root, split):
    rel = str(raw_path).replace("\\", "/").lstrip("/")
    if "sensor_blobs/" in rel:
        rel = rel.split("sensor_blobs/")[-1]
    for cand in (rel, f"{split}/{rel}"):
        if osp.exists(osp.join(sensor_root, cand)):
            return cand
    return None


def parse_args():
    p = argparse.ArgumentParser(description="synthetic NAVSIM smoke dataset")
    p.add_argument("--map-infos", required=True,
                   help="existing navsim_map_infos_*.pkl (cameras + poses)")
    p.add_argument("--sensor-root", required=True)
    p.add_argument("--split", default="test", choices=["trainval", "test"])
    p.add_argument("--out-dir", default="data/navsim_smoke")
    p.add_argument("--num-frames", type=int, default=64)
    p.add_argument("--val-frames", type=int, default=16)
    p.add_argument("--image-width", type=int, default=1920)
    p.add_argument("--objects-per-frame", type=int, default=18)
    p.add_argument("--point-cloud-range", nargs=6, type=float,
                   default=[-51.2, -51.2, -3.0, 51.2, 51.2, 5.0])
    p.add_argument("--max-sweeps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    scale = args.image_width / 1920.0

    with open(args.map_infos, "rb") as f:
        src_infos = pickle.load(f)
    src_infos = [i for i in src_infos if i.get("in_split_tokens", True)]

    lidar_dir = osp.join(args.out_dir, "lidar_bin", args.split)
    os.makedirs(lidar_dir, exist_ok=True)

    total = args.num_frames + args.val_frames
    infos, used = [], 0
    prev_by_log = {}
    for src in src_infos:
        if used >= total:
            break

        cams, ok = {}, True
        for cam_name, cam in src["cams"].items():
            rel = resolve_image(cam["img_fpath"], args.sensor_root, args.split)
            if rel is None:
                ok = False
                break
            # `extrinsics` in these pkls is lidar->camera; the dataset wants
            # camera->lidar (verified: inv() puts CAM_F0 at +1.67 m forward,
            # +1.52 m up, which matches the real rig).
            sensor2lidar = np.linalg.inv(np.asarray(cam["extrinsics"], np.float64))
            K = np.asarray(cam["intrinsics"], dtype=np.float32).copy()
            K[:2, :] *= scale
            cams[cam_name] = dict(
                data_path=osp.join("sensor_blobs", rel),
                sensor2lidar_rotation=sensor2lidar[:3, :3].astype(np.float32),
                sensor2lidar_translation=sensor2lidar[:3, 3].astype(np.float32),
                cam_intrinsic=K,
            )
        if not ok:
            continue

        n_obj = int(rng.integers(max(1, args.objects_per_frame // 2),
                                 args.objects_per_frame + 1))
        raw_boxes, names, velocity = sample_boxes(rng, n_obj)
        points = synth_points(rng, raw_boxes, pc_range=args.point_cloud_range)

        lidar_rel = osp.join("lidar_bin", args.split, f"{src['token']}.bin")
        points.tofile(osp.join(args.out_dir, lidar_rel))

        # nuScenes-style sweeps from the frames already emitted for this log,
        # using their real ego poses so the transforms are meaningful.
        R_cur = Quaternion(src["e2g_rotation"]).rotation_matrix
        t_cur = np.asarray(src["e2g_translation"], dtype=np.float64)
        sweeps = []
        for prev in reversed(prev_by_log.get(src["log_name"], [])[-args.max_sweeps:]):
            R_prev = Quaternion(prev["ego2global_rotation"]).rotation_matrix
            t_prev = np.asarray(prev["ego2global_translation"], dtype=np.float64)
            sweeps.append(dict(
                data_path=prev["lidar_path"],
                timestamp=prev["timestamp"],
                sensor2lidar_rotation=(R_cur.T @ R_prev).astype(np.float32),
                sensor2lidar_translation=(R_cur.T @ (t_prev - t_cur)).astype(np.float32),
            ))

        infos.append(dict(
            token=src["token"],
            scene_token=src.get("scene_token", ""),
            log_name=src["log_name"],
            frame_idx=int(src.get("frame_idx", 0)),
            timestamp=int(src["timestamp"]),
            location=src.get("location", ""),
            in_split_tokens=True,
            lidar_path=lidar_rel,
            sweeps=sweeps,
            lidar2ego_translation=list(src["lidar2ego_translation"]),
            lidar2ego_rotation=list(src["lidar2ego_rotation"]),
            ego2global_translation=list(src["e2g_translation"]),
            ego2global_rotation=list(src["e2g_rotation"]),
            cams=cams,
            gt_boxes=navsim_to_mmdet3d_boxes(raw_boxes),
            gt_names=names,
            gt_velocity=velocity,
            num_lidar_pts=np.full(len(raw_boxes), 100, dtype=np.int32),
            valid_flag=np.ones(len(raw_boxes), dtype=bool),
            ego_dynamic_state=np.zeros(4, dtype=np.float32),
            driving_command=np.zeros(4, dtype=np.float32),
        ))
        prev_by_log.setdefault(src["log_name"], []).append(infos[-1])
        used += 1

    if used < total:
        raise RuntimeError(f"only built {used}/{total} frames -- check --sensor-root")

    info_dir = osp.join(args.out_dir, "infos")
    os.makedirs(info_dir, exist_ok=True)
    for name, subset in (
        ("navsim_infos_smoke_train.pkl", infos[: args.num_frames]),
        ("navsim_infos_smoke_val.pkl", infos[args.num_frames :]),
    ):
        with open(osp.join(info_dir, name), "wb") as f:
            pickle.dump(subset, f, protocol=4)
        n_boxes = sum(len(i["gt_boxes"]) for i in subset)
        print(f"wrote {osp.join(info_dir, name)}: {len(subset)} frames, "
              f"{n_boxes} boxes")

    size_mb = sum(
        osp.getsize(osp.join(args.out_dir, i["lidar_path"])) for i in infos
    ) / 1e6
    print(f"synthetic LiDAR: {used} clouds, {size_mb:.1f} MB total")
    print(f"cameras per frame: {len(infos[0]['cams'])}")


if __name__ == "__main__":
    main()
