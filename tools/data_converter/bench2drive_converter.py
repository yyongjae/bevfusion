"""Bench2Drive data converter for BEVFusion.

Converts raw Bench2Drive (Carla simulator) data to BEVFusion-compatible
info files (.pkl). All coordinate systems are converted from Carla's
left-handed system to the right-handed system consistent with NuScenes.

Based on:
  /home/yongjae/e2e/Bench2DriveZoo/mmcv/datasets/prepare_B2D.py

Additional features vs the original:
  - Adds LiDAR file paths (lidar_path) to the info dict
  - Outputs pkl files directly consumable by Bench2DriveDataset in BEVFusion

Usage:
    cd tools/
    python data_converter/bench2drive_converter.py \
        --data-root /path/to/bench2drive \
        --out-dir  /path/to/output/infos \
        --workers  8

Expected B2D folder structure:
    {data_root}/
      v1/
        Town01_Route001_Weather01/
          anno/           # 00001.json.gz, 00002.json.gz, ...
          camera/
            rgb_front/    # 00001.jpg, ...
            rgb_front_left/
            rgb_front_right/
            rgb_back/
            rgb_back_left/
            rgb_back_right/
          lidar/          # 00001.npy, ...   (optional, for camera+LiDAR)
          expert_assessment/ # -0001.npz, 00001.npz, ...
      splits/
        bench2drive_base_train_val_split.json

LiDAR data format (if available):
    Each .npy file: shape (N, 4), dtype float32, columns = [x, y, z, intensity]
    Points are in CARLA's left-handed LiDAR coordinate system.
    The converter transforms them to right-handed LiDAR coordinates using
    lidar_to_righthand_ego so that the stored lidar2ego in the pkl is consistent.
"""

import argparse
import gzip
import json
import multiprocessing
import os
import pickle
from os.path import join

import cv2
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

# =========================================================================== #
# Coordinate-system constants (same as prepare_B2D.py)
# =========================================================================== #

# UE4 (stand) -> CARLA camera convention
stand_to_ue4_rotate = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)

# Raw CARLA LiDAR (left-handed) -> right-handed ego
lidar_to_righthand_ego = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)

# left-handed ego -> raw CARLA LiDAR
lefthand_ego_to_lidar = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)

# Flip Y axis: left-handed <-> right-handed
left2right = np.eye(4)
left2right[1, 1] = -1


# =========================================================================== #
# Visibility filter helpers (from prepare_B2D.py)
# =========================================================================== #

# Number of edges for box visibility check
EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def calculate_cube_vertices(center, extent):
    cx, cy, cz = center
    ex, ey, ez = extent
    verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append([cx + sx * ex, cy + sy * ey, cz + sz * ez])
    return verts


def calculate_occlusion_stats(
    pts_img, pts_depth, depth_img, max_render_depth=75
):
    """Return (num_visible, num_invisible, num_outside, colored_pts)."""
    if len(pts_img) == 0:
        return 0, 0, 8, []

    h, w = depth_img.shape[:2]
    num_visible = 0
    num_invisible = 0
    num_outside = 0
    colored_pts = []

    for pt, depth in zip(pts_img, pts_depth):
        x, y = int(pt[0]), int(pt[1])
        if x < 0 or x >= w or y < 0 or y >= h:
            num_outside += 1
            continue
        if depth <= 0 or depth > max_render_depth:
            num_invisible += 1
            continue
        depth_val = depth_img[y, x] / 255.0 * max_render_depth
        if abs(depth - depth_val) < 3.0:
            num_visible += 1
            colored_pts.append(pt)
        else:
            num_invisible += 1

    return num_visible, num_invisible, num_outside, colored_pts


def get_image_point(loc, K, w2c):
    point = np.array([loc[0], loc[1], loc[2], 1])
    point_camera = np.dot(w2c, point)[:3]
    depth = point_camera[2]
    if depth <= 0:
        return np.zeros(2), depth
    pt_img = np.dot(K, point_camera)
    pt_img[0] /= pt_img[2]
    pt_img[1] /= pt_img[2]
    return pt_img[:2], depth


def apply_trans(vec, mat):
    vec4 = np.concatenate([vec, [1.0]])
    return (mat @ vec4)[:3]


def get_npc2world(npc):
    for key in ["world2vehicle", "world2ego", "world2sign", "world2ped"]:
        if key in npc:
            npc2world = np.linalg.inv(np.array(npc[key]))
            yaw_from_matrix = np.arctan2(npc2world[1, 0], npc2world[0, 0])
            yaw = npc["rotation"][-1] / 180 * np.pi
            if abs(yaw - yaw_from_matrix) > 0.01:
                npc2world[:3, :3] = Quaternion(
                    axis=[0, 0, 1], radians=yaw
                ).rotation_matrix
            npc2world = left2right @ npc2world @ left2right
            return npc2world
    npc2world = np.eye(4)
    npc2world[:3, :3] = Quaternion(
        axis=[0, 0, 1], radians=npc["rotation"][-1] / 180 * np.pi
    ).rotation_matrix
    npc2world[:3, 3] = np.array(npc["location"])
    return left2right @ npc2world @ left2right


def get_action(index):
    DISCRETE_ACTIONS = {
        0: (0, 0, 1), 1: (0.7, -0.5, 0), 2: (0.7, -0.3, 0),
        3: (0.7, -0.2, 0), 4: (0.7, -0.1, 0), 5: (0.7, 0, 0),
        6: (0.7, 0.1, 0), 7: (0.7, 0.2, 0), 8: (0.7, 0.3, 0),
        9: (0.7, 0.5, 0), 10: (0.3, -0.7, 0), 11: (0.3, -0.5, 0),
        12: (0.3, -0.3, 0), 13: (0.3, -0.2, 0), 14: (0.3, -0.1, 0),
        15: (0.3, 0, 0), 16: (0.3, 0.1, 0), 17: (0.3, 0.2, 0),
        18: (0.3, 0.3, 0), 19: (0.3, 0.5, 0), 20: (0.3, 0.7, 0),
        21: (0, -1, 0), 22: (0, -0.6, 0), 23: (0, -0.3, 0),
        24: (0, -0.1, 0), 25: (1, 0, 0), 26: (0, 0.1, 0),
        27: (0, 0.3, 0), 28: (0, 0.6, 0), 29: (0, 1.0, 0),
        30: (0.5, -0.5, 0), 31: (0.5, -0.3, 0), 32: (0.5, -0.2, 0),
        33: (0.5, -0.1, 0), 34: (0.5, 0, 0), 35: (0.5, 0.1, 0),
        36: (0.5, 0.2, 0), 37: (0.5, 0.3, 0), 38: (0.5, 0.5, 0),
    }
    return DISCRETE_ACTIONS[index]  # (throttle, steer, brake)


# =========================================================================== #
# Filter settings
# =========================================================================== #

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CAMERA_TO_FOLDER_MAP = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}

MAX_DISTANCE = 75
FILTER_Z_THRESHOLD = 10
FILTER_INVISIBLE = True
NUM_VISIBLE_THRESHOLD = 1
NUM_OUTPOINT_THRESHOLD = 7


# =========================================================================== #
# Per-process worker
# =========================================================================== #


def _preprocess_worker(args):
    (
        folder_list,
        worker_idx,
        data_root,
        out_dir,
        tmp_dir,
        split_name,
        with_lidar,
    ) = args

    final_data = []
    iterable = tqdm(folder_list) if worker_idx == 0 else folder_list

    for folder_name in iterable:
        folder_path = join(data_root, folder_name)
        last_position_dict = {}

        ann_names = sorted(
            os.listdir(join(folder_path, "anno")),
            key=lambda x: int(x.split(".")[0]),
        )

        for ann_name in ann_names:
            frame_data = {}
            cam_gray_depth = {}

            with gzip.open(
                join(folder_path, "anno", ann_name), "rt", encoding="utf-8"
            ) as gz_file:
                anno = json.load(gz_file)

            # ---- Basic metadata ----
            frame_data["folder"] = folder_name
            frame_data["town_name"] = folder_name.split("/")[1].split("_")[1]
            frame_data["frame_idx"] = int(ann_name.split(".")[0])

            # Navigation commands
            frame_data["command_far_xy"] = np.array(
                [anno["x_command_far"], -anno["y_command_far"]]
            )
            frame_data["command_far"] = anno["command_far"]
            frame_data["command_near_xy"] = np.array(
                [anno["x_command_near"], -anno["y_command_near"]]
            )
            frame_data["command_near"] = anno["command_near"]

            # Ego motion
            frame_data["ego_yaw"] = (
                -np.nan_to_num(anno["theta"], nan=np.pi) + np.pi / 2
            )
            frame_data["ego_translation"] = np.array(
                [anno["x"], -anno["y"], 0], dtype=np.float32
            )
            frame_data["ego_vel"] = np.array(
                [anno["speed"], 0, 0], dtype=np.float32
            )
            frame_data["ego_accel"] = np.array(
                [
                    anno["acceleration"][0],
                    -anno["acceleration"][1],
                    anno["acceleration"][2],
                ],
                dtype=np.float32,
            )
            frame_data["ego_rotation_rate"] = -np.array(
                anno["angular_velocity"], dtype=np.float32
            )
            ego_bbox = anno["bounding_boxes"][0]
            frame_data["ego_size"] = np.array(
                [
                    ego_bbox["extent"][1],
                    ego_bbox["extent"][0],
                    ego_bbox["extent"][2],
                ],
                dtype=np.float32,
            ) * 2

            world2ego = left2right @ np.array(ego_bbox["world2ego"]) @ left2right
            frame_data["world2ego"] = world2ego

            # Expert action
            frame_idx = frame_data["frame_idx"]
            if frame_idx == 0:
                expert_path = join(folder_path, "expert_assessment", "-0001.npz")
            else:
                expert_path = join(
                    folder_path,
                    "expert_assessment",
                    str(frame_idx - 1).zfill(5) + ".npz",
                )
            expert_data = np.load(expert_path, allow_pickle=True)["arr_0"]
            throttle, steer, brake = get_action(int(expert_data[-1]))
            frame_data["brake"] = brake
            frame_data["throttle"] = throttle
            frame_data["steer"] = steer

            # ---- Sensor info ----
            sensor_infos = {}

            # Cameras
            for cam in CAMERAS:
                sensor_infos[cam] = {}
                sensor_infos[cam]["cam2ego"] = (
                    left2right
                    @ np.array(anno["sensors"][cam]["cam2ego"])
                    @ stand_to_ue4_rotate
                ).astype(np.float32)
                sensor_infos[cam]["intrinsic"] = np.array(
                    anno["sensors"][cam]["intrinsic"], dtype=np.float32
                )
                sensor_infos[cam]["world2cam"] = (
                    np.linalg.inv(stand_to_ue4_rotate)
                    @ np.array(anno["sensors"][cam]["world2cam"])
                    @ left2right
                ).astype(np.float32)
                sensor_infos[cam]["data_path"] = join(
                    folder_name,
                    "camera",
                    CAMERA_TO_FOLDER_MAP[cam],
                    ann_name.split(".")[0] + ".jpg",
                )

                # Depth image for visibility filtering
                depth_path = join(
                    data_root,
                    sensor_infos[cam]["data_path"],
                ).replace("rgb_", "depth_").replace(".jpg", ".png")
                cam_gray_depth[cam] = cv2.imread(depth_path)
                if cam_gray_depth[cam] is not None:
                    cam_gray_depth[cam] = cam_gray_depth[cam][:, :, 0]
                else:
                    cam_gray_depth[cam] = np.zeros((900, 1600), dtype=np.uint8)

            # LiDAR
            sensor_infos["LIDAR_TOP"] = {}
            sensor_infos["LIDAR_TOP"]["lidar2ego"] = (
                left2right
                @ np.array(anno["sensors"]["LIDAR_TOP"]["lidar2ego"])
                @ left2right
                @ lidar_to_righthand_ego
            ).astype(np.float32)
            world2lidar = (
                lefthand_ego_to_lidar
                @ np.array(anno["sensors"]["LIDAR_TOP"]["world2lidar"])
                @ left2right
            ).astype(np.float32)
            sensor_infos["LIDAR_TOP"]["world2lidar"] = world2lidar

            # LiDAR data path (optional) – support .npy and .laz
            if with_lidar:
                frame_id_str = str(frame_data["frame_idx"]).zfill(5)
                for ext in (".npy", ".laz"):
                    lidar_filename = frame_id_str + ext
                    lidar_rel_path = join(folder_name, "lidar", lidar_filename)
                    lidar_abs_path = join(data_root, lidar_rel_path)
                    if os.path.exists(lidar_abs_path):
                        sensor_infos["LIDAR_TOP"]["data_path"] = lidar_rel_path
                        break

            frame_data["sensors"] = sensor_infos

            # ---- Bounding boxes ----
            gt_boxes = []
            gt_names = []
            gt_ids = []
            num_points_list = []
            npc2world_list = []
            position_dict = {}

            for npc in anno["bounding_boxes"]:
                if npc["class"] == "ego_vehicle":
                    continue
                if npc["distance"] > MAX_DISTANCE:
                    continue
                if (
                    abs(npc["location"][2] - ego_bbox["location"][2])
                    > FILTER_Z_THRESHOLD
                ):
                    continue

                # Convert center from left-hand to right-hand
                center = np.array(
                    [npc["center"][0], -npc["center"][1], npc["center"][2]],
                    dtype=np.float32,
                )
                # wlh from CARLA extent (xyz half-size) -> (w, l, h) half-size
                extent = np.array(
                    [npc["extent"][1], npc["extent"][0], npc["extent"][2]],
                    dtype=np.float32,
                )
                position_dict[npc["id"]] = center

                local_center = apply_trans(center, world2lidar)
                size = extent * 2  # full WLH

                # Compute yaw in LiDAR frame
                if "world2vehicle" in npc:
                    world2vehicle = left2right @ np.array(npc["world2vehicle"]) @ left2right
                    vehicle2lidar = world2lidar @ np.linalg.inv(world2vehicle)
                    yaw_local = np.arctan2(vehicle2lidar[1, 0], vehicle2lidar[0, 0])
                else:
                    yaw_local = (
                        -npc["rotation"][-1] / 180 * np.pi
                        - frame_data["ego_yaw"]
                        + np.pi / 2
                    )

                # NuScenes LiDAR box yaw convention: -yaw - pi/2
                yaw_lidar_box = -yaw_local - np.pi / 2
                # Normalise to [-pi, pi]
                while yaw_local < -np.pi:
                    yaw_local += 2 * np.pi
                while yaw_local > np.pi:
                    yaw_local -= 2 * np.pi

                # Velocity
                if "speed" in npc:
                    if "vehicle" in npc["class"]:
                        speed = npc["speed"]
                    else:
                        if npc["id"] in last_position_dict:
                            speed = (
                                np.linalg.norm(
                                    (center - last_position_dict[npc["id"]])[:2]
                                )
                                * 10
                            )
                        else:
                            speed = 0.0
                else:
                    speed = 0.0

                num_pts = npc.get("num_points", -1)

                speed_x = speed * np.cos(yaw_local)
                speed_y = speed * np.sin(yaw_local)

                npc2world = get_npc2world(npc)

                # ---- Visibility filter ----
                if FILTER_INVISIBLE:
                    valid = False
                    box2lidar = np.eye(4, dtype=np.float32)
                    box2lidar[:3, :3] = Quaternion(
                        axis=[0, 0, 1], radians=yaw_local
                    ).rotation_matrix
                    box2lidar[:3, 3] = local_center
                    lidar2box = np.linalg.inv(box2lidar)

                    raw_verts = calculate_cube_vertices(local_center, extent)
                    verts = []
                    for rv in raw_verts:
                        tmp = lidar2box @ [rv[0], rv[1], rv[2], 1]
                        tmp[:3] += local_center
                        verts.append(tmp[:3].tolist())

                    for cam in CAMERAS:
                        lidar2cam = (
                            np.linalg.inv(sensor_infos[cam]["cam2ego"])
                            @ sensor_infos["LIDAR_TOP"]["lidar2ego"]
                        )
                        test_points = []
                        test_depths = []
                        for vert in verts:
                            pt, depth = get_image_point(
                                vert,
                                sensor_infos[cam]["intrinsic"],
                                lidar2cam,
                            )
                            if depth > 0:
                                test_points.append(pt)
                                test_depths.append(depth)

                        num_vis, _, num_out, _ = calculate_occlusion_stats(
                            np.array(test_points),
                            np.array(test_depths),
                            cam_gray_depth[cam],
                            max_render_depth=MAX_DISTANCE,
                        )
                        if (
                            num_vis > NUM_VISIBLE_THRESHOLD
                            and num_out < NUM_OUTPOINT_THRESHOLD
                        ):
                            valid = True
                            break
                else:
                    valid = True

                if valid:
                    gt_boxes.append(
                        np.array(
                            [
                                local_center[0],
                                local_center[1],
                                local_center[2],
                                size[0],
                                size[1],
                                size[2],
                                yaw_lidar_box,
                                speed_x,
                                speed_y,
                            ],
                            dtype=np.float32,
                        )
                    )
                    gt_names.append(npc["type_id"])
                    gt_ids.append(int(npc["id"]))
                    num_points_list.append(int(num_pts))
                    npc2world_list.append(npc2world)

            if len(gt_boxes) == 0:
                continue

            last_position_dict = position_dict.copy()

            frame_data["gt_ids"] = np.array(gt_ids)
            frame_data["gt_names"] = np.array(gt_names)
            frame_data["gt_boxes"] = np.stack(gt_boxes)
            frame_data["num_points"] = np.array(num_points_list)
            frame_data["npc2world"] = np.stack(npc2world_list)

            # Add LiDAR path at top level for easy access by dataset class
            if (
                "data_path" in sensor_infos.get("LIDAR_TOP", {})
            ):
                frame_data["lidar_path"] = sensor_infos["LIDAR_TOP"]["data_path"]

            final_data.append(frame_data)

    os.makedirs(join(out_dir, tmp_dir), exist_ok=True)
    out_path = join(
        out_dir, tmp_dir, "b2d_infos_{}_{}.pkl".format(split_name, worker_idx)
    )
    with open(out_path, "wb") as f:
        pickle.dump(final_data, f)


def generate_infos(folder_list, workers, split_name, data_root, out_dir, tmp_dir, with_lidar):
    """Parallel info generation for a train/val split."""
    n = len(folder_list)
    boundaries = [n * i // workers for i in range(workers + 1)]

    args_list = [
        (
            folder_list[boundaries[i]: boundaries[i + 1]],
            i,
            data_root,
            out_dir,
            tmp_dir,
            split_name,
            with_lidar,
        )
        for i in range(workers)
    ]

    pool = multiprocessing.Pool(workers)
    pool.map(_preprocess_worker, args_list)
    pool.close()
    pool.join()

    # Merge partial results
    merged = []
    for i in range(workers):
        part_path = join(
            out_dir,
            tmp_dir,
            "b2d_infos_{}_{}.pkl".format(split_name, i),
        )
        with open(part_path, "rb") as f:
            merged.extend(pickle.load(f))

    out_path = join(out_dir, "b2d_infos_{}.pkl".format(split_name))
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    print(
        "Saved {} {} samples -> {}".format(len(merged), split_name, out_path)
    )


def create_bench2drive_infos(
    data_root,
    out_dir,
    split_file,
    workers=4,
    with_lidar=True,
    tmp_dir="tmp_b2d",
):
    """Entry point: generate train and val info pkl files.

    Args:
        data_root (str): Root of the Bench2Drive dataset.
        out_dir (str): Output directory for pkl files.
        split_file (str): JSON file with 'val' list of folder names.
        workers (int): Number of parallel processes.
        with_lidar (bool): Whether to include LiDAR file paths.
        tmp_dir (str): Temporary directory for partial results.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(split_file, "r") as f:
        train_val_split = json.load(f)

    # Collect all folders
    v1_dir = join(data_root, "v1")
    all_folders = [
        join("v1", d)
        for d in os.listdir(v1_dir)
        if "Town" in d and "Route" in d and "Weather" in d
    ]

    val_set = set(train_val_split["val"])
    train_list = [f for f in all_folders if f not in val_set]
    val_list = list(val_set)

    print(
        "Found {} train folders, {} val folders".format(
            len(train_list), len(val_list)
        )
    )

    print("Processing train split...")
    generate_infos(
        train_list, workers, "train", data_root, out_dir, tmp_dir, with_lidar
    )

    print("Processing val split...")
    generate_infos(
        val_list, workers, "val", data_root, out_dir, tmp_dir, with_lidar
    )

    print("Done! Info files written to:", out_dir)


# =========================================================================== #
# CLI
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bench2Drive data converter for BEVFusion"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root path of Bench2Drive dataset (contains v1/, splits/)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/infos",
        help="Output directory for info pkl files",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help=(
            "Path to bench2drive_base_train_val_split.json. "
            "Defaults to {data_root}/splits/bench2drive_base_train_val_split.json"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--no-lidar",
        action="store_true",
        help="Skip LiDAR path generation (camera-only mode)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="tmp_b2d",
        help="Temporary directory for partial pkl files",
    )
    args = parser.parse_args()

    split_file = args.split_file
    if split_file is None:
        split_file = join(
            args.data_root,
            "splits",
            "bench2drive_base_train_val_split.json",
        )

    create_bench2drive_infos(
        data_root=args.data_root,
        out_dir=args.out_dir,
        split_file=split_file,
        workers=args.workers,
        with_lidar=not args.no_lidar,
        tmp_dir=args.tmp_dir,
    )
