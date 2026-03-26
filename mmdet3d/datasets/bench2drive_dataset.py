import json
import tempfile
from os import path as osp
from typing import Any, Dict

import mmcv
import numpy as np
import pyquaternion
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS

from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .nuscenes_styled_eval_utils import (
    DetectionBox,
    DetectionMetricDataList,
    DetectionMetrics,
    EvalBoxes,
    accumulate,
    calc_ap,
    calc_tp,
    center_distance,
)


@DATASETS.register_module()
class Bench2DriveDataset(Custom3DDataset):
    """Bench2Drive Dataset for BEVFusion.

    Adapts Bench2Drive (Carla simulator) data format for use with BEVFusion.
    Data is assumed to be pre-converted from left-handed to right-handed
    coordinate system by bench2drive_converter.py (adapted from prepare_B2D.py).

    Args:
        ann_file (str): Path of annotation file (.pkl).
        pipeline (list[dict]): Pipeline used for data processing.
        dataset_root (str): Path of dataset root.
        object_classes (tuple[str]): Classes used in the dataset.
        load_interval (int): Interval of loading the dataset.
        with_velocity (bool): Whether to include velocity in boxes.
        modality (dict): Modality config (use_camera, use_lidar, etc.).
        box_type_3d (str): Type of 3D box ('LiDAR', 'Depth', 'Camera').
        filter_empty_gt (bool): Whether to filter empty GT.
        test_mode (bool): Whether the dataset is in test mode.
        eval_cfg (dict): Evaluation configuration.
        name_mapping (dict): Override default name mapping.
        use_valid_flag (bool): Whether to use valid_flag to filter boxes.
    """

    # Maps Carla type IDs to standard class names
    NameMapping = {
        # ---- bicycle ----
        "vehicle.bh.crossbike": "bicycle",
        "vehicle.diamondback.century": "bicycle",
        "vehicle.gazelle.omafiets": "bicycle",
        # ---- car ----
        "vehicle.audi.etron": "car",
        "vehicle.chevrolet.impala": "car",
        "vehicle.dodge.charger_2020": "car",
        "vehicle.dodge.charger_police": "car",
        "vehicle.dodge.charger_police_2020": "car",
        "vehicle.lincoln.mkz_2017": "car",
        "vehicle.lincoln.mkz_2020": "car",
        "vehicle.mini.cooper_s_2021": "car",
        "vehicle.mercedes.coupe_2020": "car",
        "vehicle.ford.mustang": "car",
        "vehicle.nissan.patrol_2021": "car",
        "vehicle.audi.tt": "car",
        "vehicle.ford.crown": "car",
        "vehicle.tesla.model3": "car",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": "car",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": "car",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": "car",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": "car",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": "car",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": "car",
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": "car",
        # ---- van ----
        "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
        "vehicle.ford.ambulance": "van",
        # ---- truck ----
        "vehicle.carlamotors.firetruck": "truck",
        # ---- traffic_sign ----
        "traffic.speed_limit.30": "traffic_sign",
        "traffic.speed_limit.40": "traffic_sign",
        "traffic.speed_limit.50": "traffic_sign",
        "traffic.speed_limit.60": "traffic_sign",
        "traffic.speed_limit.90": "traffic_sign",
        "traffic.speed_limit.120": "traffic_sign",
        "traffic.stop": "traffic_sign",
        "traffic.yield": "traffic_sign",
        # ---- traffic_light ----
        "traffic.traffic_light": "traffic_light",
        # ---- traffic_cone ----
        "static.prop.warningconstruction": "traffic_cone",
        "static.prop.warningaccident": "traffic_cone",
        "static.prop.trafficwarning": "traffic_cone",
        "static.prop.constructioncone": "traffic_cone",
        # ---- pedestrian ----
        "walker.pedestrian.0001": "pedestrian",
        "walker.pedestrian.0003": "pedestrian",
        "walker.pedestrian.0004": "pedestrian",
        "walker.pedestrian.0005": "pedestrian",
        "walker.pedestrian.0007": "pedestrian",
        "walker.pedestrian.0010": "pedestrian",
        "walker.pedestrian.0013": "pedestrian",
        "walker.pedestrian.0014": "pedestrian",
        "walker.pedestrian.0015": "pedestrian",
        "walker.pedestrian.0016": "pedestrian",
        "walker.pedestrian.0017": "pedestrian",
        "walker.pedestrian.0018": "pedestrian",
        "walker.pedestrian.0019": "pedestrian",
        "walker.pedestrian.0020": "pedestrian",
        "walker.pedestrian.0021": "pedestrian",
        "walker.pedestrian.0022": "pedestrian",
        "walker.pedestrian.0025": "pedestrian",
        "walker.pedestrian.0027": "pedestrian",
        "walker.pedestrian.0030": "pedestrian",
        "walker.pedestrian.0031": "pedestrian",
        "walker.pedestrian.0032": "pedestrian",
        "walker.pedestrian.0034": "pedestrian",
        "walker.pedestrian.0035": "pedestrian",
        "walker.pedestrian.0041": "pedestrian",
        "walker.pedestrian.0042": "pedestrian",
        "walker.pedestrian.0046": "pedestrian",
        "walker.pedestrian.0047": "pedestrian",
        # ---- others ----
        "static.prop.dirtdebris01": "others",
        "static.prop.dirtdebris02": "others",
    }

    CLASSES = (
        "car",
        "van",
        "truck",
        "bicycle",
        "traffic_sign",
        "traffic_cone",
        "traffic_light",
        "pedestrian",
        "others",
    )

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_cfg=None,
        name_mapping=None,
        use_valid_flag=False,
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag

        if name_mapping is not None:
            self.NameMapping = name_mapping

        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.with_velocity = with_velocity

        if eval_cfg is not None:
            self.eval_cfg = eval_cfg
        else:
            self.eval_cfg = {
                "dist_ths": [0.5, 1.0, 2.0, 4.0],
                "dist_th_tp": 2.0,
                "min_recall": 0.1,
                "min_precision": 0.1,
                "mean_ap_weight": 5,
                "class_names": [
                    "car",
                    "van",
                    "truck",
                    "bicycle",
                    "traffic_sign",
                    "traffic_cone",
                    "traffic_light",
                    "pedestrian",
                ],
                "tp_metrics": ["trans_err", "scale_err", "orient_err", "vel_err"],
                "err_name_maping": {
                    "trans_err": "mATE",
                    "scale_err": "mASE",
                    "orient_err": "mAOE",
                    "vel_err": "mAVE",
                    "attr_err": "mAAE",
                },
                "class_range": {
                    "car": (50, 50),
                    "van": (50, 50),
                    "truck": (50, 50),
                    "bicycle": (40, 40),
                    "traffic_sign": (30, 30),
                    "traffic_cone": (30, 30),
                    "traffic_light": (30, 30),
                    "pedestrian": (40, 40),
                },
            }

        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def load_annotations(self, ann_file):
        """Load annotations from B2D pkl file.

        B2D pkl files are directly a list of frame dicts (unlike NuScenes
        which wraps them in {'infos': [...], 'metadata': {...}}).
        """
        data = mmcv.load(ann_file)
        if isinstance(data, list):
            data_infos = data
        elif isinstance(data, dict) and "infos" in data:
            # Support wrapped format just in case
            data_infos = data["infos"]
        else:
            raise ValueError(
                f"Unexpected B2D data format in {ann_file}. "
                "Expected a list or dict with 'infos' key."
            )
        data_infos = data_infos[:: self.load_interval]
        return data_infos

    def _apply_name_mapping(self, gt_names):
        """Map Carla type IDs to standard class names."""
        return np.array([self.NameMapping.get(n, n) for n in gt_names])

    def get_data_info(self, index: int) -> Dict[str, Any]:
        """Get data info for BEVFusion pipeline.

        Converts B2D sensor format to BEVFusion-compatible format:
        - Builds ego2global from ego_yaw + ego_translation
        - Derives lidar2camera / camera2lidar from cam2ego + lidar2ego matrices
        - Pads 3x3 intrinsics to 4x4
        """
        info = self.data_infos[index]

        # Unique sample token
        token = "{}_{}".format(info["folder"], info["frame_idx"])

        data = dict(
            token=token,
            sample_idx=token,
            timestamp=float(info["frame_idx"]),
            location=info.get("town_name", ""),
        )

        # ------------------------------------------------------------------ #
        # LiDAR info
        # ------------------------------------------------------------------ #
        lidar2ego = np.array(
            info["sensors"]["LIDAR_TOP"]["lidar2ego"], dtype=np.float32
        )
        data["lidar2ego"] = lidar2ego

        # LiDAR file path (added by bench2drive_converter.py)
        # Check top-level "lidar_path" first, then fall back to
        # sensors/LIDAR_TOP/data_path (older pkl files may only have the latter).
        lidar_path = info.get(
            "lidar_path",
            info["sensors"].get("LIDAR_TOP", {}).get("data_path"),
        )
        if lidar_path is not None:
            if not osp.isabs(lidar_path):
                lidar_path = osp.join(self.dataset_root, lidar_path)
            data["lidar_path"] = lidar_path
        data["sweeps"] = []  # B2D has no temporal LiDAR sweeps

        # ------------------------------------------------------------------ #
        # Ego-to-global transform
        # Reconstructed from ego_yaw (radians, right-handed) + ego_translation
        # ------------------------------------------------------------------ #
        ego_yaw = float(np.nan_to_num(info["ego_yaw"], nan=np.pi / 2))
        ego_translation = np.array(info["ego_translation"], dtype=np.float32)
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = Quaternion(
            axis=[0, 0, 1], radians=ego_yaw
        ).rotation_matrix
        ego2global[:3, 3] = ego_translation
        data["ego2global"] = ego2global

        # ------------------------------------------------------------------ #
        # Camera info
        # ------------------------------------------------------------------ #
        if self.modality.get("use_camera", True):
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for sensor_type, cam_info in info["sensors"].items():
                if "CAM" not in sensor_type:
                    continue

                # Image path (relative in info, prepend dataset_root)
                img_path = cam_info["data_path"]
                if not osp.isabs(img_path):
                    img_path = osp.join(self.dataset_root, img_path)
                data["image_paths"].append(img_path)

                # cam2ego: 4x4 matrix (already in right-handed coords)
                cam2ego = np.array(cam_info["cam2ego"], dtype=np.float32)
                data["camera2ego"].append(cam2ego)

                # lidar -> camera = inv(cam2ego) @ lidar2ego
                lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
                data["lidar2camera"].append(lidar2cam.astype(np.float32))

                # camera -> lidar
                camera2lidar = np.linalg.inv(lidar2cam)
                data["camera2lidar"].append(camera2lidar.astype(np.float32))

                # Intrinsics: 3x3 -> 4x4
                intrinsic = np.array(cam_info["intrinsic"])
                camera_intrinsics = np.eye(4, dtype=np.float32)
                camera_intrinsics[
                    : intrinsic.shape[0], : intrinsic.shape[1]
                ] = intrinsic
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar -> image = intrinsics(4x4) @ lidar2cam(4x4)
                lidar2image = camera_intrinsics @ lidar2cam
                data["lidar2image"].append(lidar2image.astype(np.float32))

        # ------------------------------------------------------------------ #
        # Annotations
        # ------------------------------------------------------------------ #
        annos = self.get_ann_info(index)
        data["ann_info"] = annos
        return data

    def get_ann_info(self, index):
        """Get annotation info.

        B2D gt_boxes format: [x, y, z, w, l, h, yaw_lidar_box, vx, vy]
        where yaw_lidar_box = -yaw_local - pi/2  (NuScenes LiDAR box convention)
        Box origin is (0.5, 0.5, 0.5) — true 3D center.
        """
        info = self.data_infos[index]

        gt_bboxes_3d = np.array(info["gt_boxes"], dtype=np.float32)
        gt_names_raw = np.array(info["gt_names"])

        # Map Carla type IDs to standard class names
        gt_names_3d = self._apply_name_mapping(gt_names_raw)

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if not self.with_velocity:
            gt_bboxes_3d = gt_bboxes_3d[:, :7]

        # B2D boxes are centered at (0.5, 0.5, 0.5) — true center, not bottom
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5),
        ).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

    def get_cat_ids(self, idx):
        """Get category ids for sampler (e.g. CBGS)."""
        info = self.data_infos[idx]
        gt_names = self._apply_name_mapping(np.array(info["gt_names"]))
        cat_ids = []
        for name in set(gt_names):
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    # ------------------------------------------------------------------ #
    # Formatting & Evaluation
    # ------------------------------------------------------------------ #

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert model outputs to B2D evaluation JSON format."""
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            box3d = det["boxes_3d"]
            scores = det["scores_3d"]
            labels = det["labels_3d"]

            box_gravity_center = box3d.gravity_center
            box_dims = box3d.dims
            box_yaw = box3d.yaw.numpy()
            # Convert from BEVFusion internal yaw to NuScenes box rotation
            box_yaw = -box_yaw - np.pi / 2

            info = self.data_infos[sample_id]
            sample_token = "{}_{}".format(info["folder"], info["frame_idx"])

            for i in range(len(box3d)):
                quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                velocity = [
                    box3d.tensor[i, 7].item(),
                    box3d.tensor[i, 8].item(),
                ]
                name = mapped_class_names[labels[i]]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat,
                    velocity=velocity,
                    detection_name=name,
                    detection_score=scores[i].item(),
                    attribute_name=name,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_b2d.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format results to JSON for evaluation."""
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update({name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def load_gt(self):
        """Build GT EvalBoxes from data_infos for evaluation."""
        all_annotations = EvalBoxes()
        class_range = self.eval_cfg.get("class_range", {})

        for i in range(len(self.data_infos)):
            sample_data = self.data_infos[i]
            sample_token = "{}_{}".format(
                sample_data["folder"], sample_data["frame_idx"]
            )
            sample_boxes = []

            gt_boxes = np.array(sample_data["gt_boxes"])
            gt_names = self._apply_name_mapping(np.array(sample_data["gt_names"]))
            num_points = np.array(sample_data["num_points"])

            for j in range(gt_boxes.shape[0]):
                class_name = gt_names[j]
                if class_name not in class_range:
                    continue
                range_x, range_y = class_range[class_name]
                if abs(gt_boxes[j, 0]) > range_x or abs(gt_boxes[j, 1]) > range_y:
                    continue

                # Convert yaw from lidar_box convention back to quaternion
                # yaw_lidar_box = -yaw_local - pi/2  =>  yaw_local = -yaw_lidar_box - pi/2
                box_yaw_for_quat = -gt_boxes[j, 6] - np.pi / 2
                rotation = list(
                    Quaternion(axis=[0, 0, 1], radians=box_yaw_for_quat)
                )

                sample_boxes.append(
                    DetectionBox(
                        sample_token=sample_token,
                        translation=tuple(gt_boxes[j, 0:3].tolist()),
                        size=tuple(gt_boxes[j, 3:6].tolist()),
                        rotation=rotation,
                        velocity=tuple(gt_boxes[j, 7:9].tolist()),
                        num_pts=int(num_points[j]),
                        detection_name=class_name,
                        detection_score=-1.0,
                        attribute_name=class_name,
                    )
                )
            all_annotations.add_boxes(sample_token, sample_boxes)
        return all_annotations

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        """Evaluate with B2D-style (nuScenes-styled) metrics.

        Returns:
            dict[str, float]: Evaluation metrics (mAP, NDS, per-class AP/TP).
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        # Resolve result path
        if isinstance(result_files, dict):
            result_path = result_files.get(
                "pts_bbox", list(result_files.values())[0]
            )
        else:
            result_path = result_files

        with open(result_path) as f:
            result_data = json.load(f)

        pred_boxes = EvalBoxes.deserialize(result_data["results"], DetectionBox)
        gt_boxes = self.load_gt()

        # Accumulate metrics
        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg["class_names"]:
            for dist_th in self.eval_cfg["dist_ths"]:
                md = accumulate(
                    gt_boxes, pred_boxes, class_name, center_distance, dist_th
                )
                metric_data_list.set(class_name, dist_th, md)

        metrics = DetectionMetrics(self.eval_cfg)
        for class_name in self.eval_cfg["class_names"]:
            for dist_th in self.eval_cfg["dist_ths"]:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(
                    metric_data,
                    self.eval_cfg["min_recall"],
                    self.eval_cfg["min_precision"],
                )
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in self.eval_cfg["tp_metrics"]:
                metric_data = metric_data_list[
                    (class_name, self.eval_cfg["dist_th_tp"])
                ]
                tp = calc_tp(metric_data, self.eval_cfg["min_recall"], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()

        # ---- Print summary ----
        err_name_mapping = self.eval_cfg.get("err_name_maping", self.ErrNameMapping)
        print("mAP: %.4f" % metrics_summary["mean_ap"])
        for tp_name, tp_val in metrics_summary["tp_errors"].items():
            print("%s: %.4f" % (err_name_mapping.get(tp_name, tp_name), tp_val))
        print("NDS: %.4f" % metrics_summary["nd_score"])

        print("\nPer-class results:")
        print("Object Class\tAP\tATE\tASE\tAOE\tAVE")
        class_aps = metrics_summary["mean_dist_aps"]
        class_tps = metrics_summary["label_tp_errors"]
        for class_name in class_aps.keys():
            print(
                "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
                % (
                    class_name,
                    class_aps[class_name],
                    class_tps[class_name]["trans_err"],
                    class_tps[class_name]["scale_err"],
                    class_tps[class_name]["orient_err"],
                    class_tps[class_name]["vel_err"],
                )
            )

        # ---- Build detail dict for logging ----
        detail = dict()
        metric_prefix = "bbox_B2D"
        for name in self.eval_cfg["class_names"]:
            for k, v in metrics_summary["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_AP_dist_{}".format(metric_prefix, name, k)] = val
            for k, v in metrics_summary["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_{}".format(metric_prefix, name, k)] = val
            for k, v in metrics_summary["tp_errors"].items():
                val = float("{:.4f}".format(v))
                detail[
                    "{}/{}".format(metric_prefix, err_name_mapping.get(k, k))
                ] = val
        detail["{}/NDS".format(metric_prefix)] = metrics_summary["nd_score"]
        detail["{}/mAP".format(metric_prefix)] = metrics_summary["mean_ap"]

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return detail
