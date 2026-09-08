"""NAVSIM (OpenScene / nuPlan) dataset for BEVFusion.

Consumes the info pickles produced by
``tools/data_converter/navsim_converter.py``.

Coordinate conventions
----------------------
NAVSIM stores everything in the *ego* frame, whose origin is the rear axle
projected onto the ground plane (z = 0 is the ground, +x forward, +y left,
+z up).  ``lidar2ego`` is the identity, so the ego frame *is* the LiDAR frame
and no extra transform is needed anywhere in this file.

``gt_boxes`` in the info pkl are already stored in the mmdet3d LiDAR box
convention (see the converter for the exact transformation from NAVSIM's
``[x, y, z, length, width, height, heading]``)::

    [x, y, z_center, dx, dy, dz, yaw, vx, vy]
    dx, dy, dz = width, length, height
    yaw        = -heading - pi / 2

The z coordinate is the **box center**, hence ``origin=(0.5, 0.5, 0.5)`` when
building :class:`LiDARInstance3DBoxes`.
"""

import json
import tempfile
from os import path as osp
from typing import Any, Dict, List, Optional

import mmcv
import numpy as np
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

# The eight NAVSIM cameras, in the order used by ``navsim.common.dataclasses``.
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

# nuPlan ``TrackedObjectType`` names that appear in ``anns['gt_names']``.
NAVSIM_CLASSES = [
    "vehicle",
    "pedestrian",
    "bicycle",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
]


@DATASETS.register_module()
class NavsimDataset(Custom3DDataset):
    """NAVSIM dataset for BEVFusion.

    Args:
        ann_file (str): Path of the info pickle.
        pipeline (list[dict]): Processing pipeline.
        dataset_root (str): Root that relative ``data_path`` / ``lidar_path``
            entries in the info pkl are resolved against.
        object_classes (list[str]): Classes used by the detector.
        camera_names (list[str], optional): Subset of :data:`NAVSIM_CAMERAS`
            to feed to the model, in the given order. ``None`` uses all eight.
            Use e.g. ``[CAM_F0, CAM_L0, CAM_R0]`` for the standard NAVSIM
            front-view setup.
        load_interval (int): Take every ``load_interval``-th sample.
        with_velocity (bool): Append (vx, vy) to the boxes.
        modality (dict): ``use_camera`` / ``use_lidar`` switches.
        eval_cfg (dict): nuScenes-styled evaluation configuration.
        use_valid_flag (bool): Filter GT with the ``valid_flag`` field.
        filter_by_split (bool): Keep only frames whose ``in_split_tokens`` /
            ``is_key_frame`` flag is True (the official navtrain/navtest
            token set). Ignored when the field is absent.
    """

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

    DefaultEvalCfg = {
        "dist_ths": [0.5, 1.0, 2.0, 4.0],
        "dist_th_tp": 2.0,
        "min_recall": 0.1,
        "min_precision": 0.1,
        "mean_ap_weight": 5,
        "class_names": NAVSIM_CLASSES,
        "tp_metrics": ["trans_err", "scale_err", "orient_err", "vel_err"],
        "err_name_maping": ErrNameMapping,
        "class_range": {
            "vehicle": (50, 50),
            "pedestrian": (40, 40),
            "bicycle": (40, 40),
            "traffic_cone": (30, 30),
            "barrier": (30, 30),
            "czone_sign": (30, 30),
            "generic_object": (30, 30),
        },
    }

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        camera_names=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_cfg=None,
        use_valid_flag=True,
        filter_by_split=True,
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.filter_by_split = filter_by_split
        self.camera_names = list(camera_names) if camera_names else NAVSIM_CAMERAS

        unknown = set(self.camera_names) - set(NAVSIM_CAMERAS)
        assert not unknown, f"Unknown NAVSIM cameras: {sorted(unknown)}"

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
        self.eval_cfg = eval_cfg if eval_cfg is not None else self.DefaultEvalCfg

        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_annotations(self, ann_file):
        """Load the info pkl (a plain list of per-frame dicts)."""
        data = mmcv.load(ann_file)
        if isinstance(data, dict):
            self.metadata = data.get("metadata", {})
            data_infos = data["infos"]
        else:
            self.metadata = {}
            data_infos = data

        if self.filter_by_split:
            flagged = [i for i in data_infos if i.get("in_split_tokens", True)]
            # Only apply the filter when it does not wipe out the dataset.
            if flagged:
                data_infos = flagged

        data_infos = sorted(data_infos, key=lambda e: (e["log_name"], e["timestamp"]))
        return data_infos[:: self.load_interval]

    def _abs(self, path: str) -> str:
        if path is None or osp.isabs(path):
            return path
        return osp.join(self.dataset_root, path)

    # ------------------------------------------------------------------ #
    # Sample construction
    # ------------------------------------------------------------------ #
    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            token=info["token"],
            sample_idx=info["token"],
            timestamp=info["timestamp"],
            location=info.get("location", ""),
            lidar_path=self._abs(info.get("lidar_path")),
            # sweep paths are stored relative to dataset_root like lidar_path,
            # but LoadPointsFromMultiSweeps reads them verbatim
            sweeps=[
                {**s, "data_path": self._abs(s["data_path"])}
                for s in info.get("sweeps", [])
            ],
        )

        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for cam_name in self.camera_names:
                cam = info["cams"][cam_name]
                data["image_paths"].append(self._abs(cam["data_path"]))

                # NAVSIM gives camera -> LiDAR directly.
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = cam["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = cam["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

                lidar2camera = np.linalg.inv(camera2lidar).astype(np.float32)
                data["lidar2camera"].append(lidar2camera)

                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam["cam_intrinsic"]
                data["camera_intrinsics"].append(camera_intrinsics)

                data["lidar2image"].append(camera_intrinsics @ lidar2camera)

                # lidar2ego is the identity in NAVSIM, so camera2ego == camera2lidar,
                # but compose it explicitly to stay correct if that ever changes.
                data["camera2ego"].append(
                    (lidar2ego @ camera2lidar).astype(np.float32)
                )

        data["ann_info"] = self.get_ann_info(index)
        return data

    def get_ann_info(self, index):
        info = self.data_infos[index]

        gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float32).reshape(-1, 7)
        gt_names = np.asarray(info["gt_names"]).reshape(-1)

        if self.use_valid_flag and "valid_flag" in info:
            mask = np.asarray(info["valid_flag"], dtype=bool)
        else:
            mask = np.ones(len(gt_boxes), dtype=bool)

        gt_boxes = gt_boxes[mask]
        gt_names = gt_names[mask]

        if self.with_velocity:
            gt_velocity = np.asarray(
                info["gt_velocity"], dtype=np.float32
            ).reshape(-1, 2)[mask]
            gt_velocity = np.nan_to_num(gt_velocity, nan=0.0)
            gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)

        gt_labels = np.array(
            [self.CLASSES.index(n) if n in self.CLASSES else -1 for n in gt_names],
            dtype=np.int64,
        )

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels,
            gt_names=gt_names,
        )

    def get_cat_ids(self, idx):
        """Category ids of one frame, used by :class:`CBGSDataset`."""
        info = self.data_infos[idx]
        gt_names = np.asarray(info["gt_names"]).reshape(-1)
        if self.use_valid_flag and "valid_flag" in info:
            gt_names = gt_names[np.asarray(info["valid_flag"], dtype=bool)]
        return [self.cat2id[n] for n in set(gt_names.tolist()) if n in self.CLASSES]

    # ------------------------------------------------------------------ #
    # Formatting & evaluation (nuScenes-styled, same as Bench2DriveDataset)
    # ------------------------------------------------------------------ #
    def _in_eval_range(self, x, y, class_name):
        """Return whether a box centre lies in both class and model ranges."""
        class_range = self.eval_cfg.get("class_range", {})
        if class_name not in class_range:
            return False
        range_x, range_y = class_range[class_name]
        if abs(x) > range_x or abs(y) > range_y:
            return False

        point_cloud_range = self.eval_cfg.get("point_cloud_range")
        if point_cloud_range is not None:
            return (
                point_cloud_range[0] <= x <= point_cloud_range[3]
                and point_cloud_range[1] <= y <= point_cloud_range[4]
            )
        return True

    def _format_bbox(self, results, jsonfile_prefix=None):
        navsim_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            box3d = det["boxes_3d"]
            scores = det["scores_3d"]
            labels = det["labels_3d"]

            box_gravity_center = box3d.gravity_center
            box_dims = box3d.dims
            # mmdet3d yaw -> NAVSIM heading
            box_yaw = -box3d.yaw.numpy() - np.pi / 2

            sample_token = self.data_infos[sample_id]["token"]

            for i in range(len(box3d)):
                name = mapped_class_names[labels[i]]
                center_x = box_gravity_center[i, 0].item()
                center_y = box_gravity_center[i, 1].item()
                if not self._in_eval_range(center_x, center_y, name):
                    continue
                annos.append(
                    dict(
                        sample_token=sample_token,
                        translation=box_gravity_center[i].tolist(),
                        size=box_dims[i].tolist(),
                        rotation=list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i])),
                        velocity=[
                            box3d.tensor[i, 7].item(),
                            box3d.tensor[i, 8].item(),
                        ],
                        detection_name=name,
                        detection_score=scores[i].item(),
                        attribute_name=name,
                    )
                )
            navsim_annos[sample_token] = annos

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_navsim.json")
        print("Results writes to", res_path)
        mmcv.dump({"meta": self.modality, "results": navsim_annos}, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            return self._format_bbox(results, jsonfile_prefix), tmp_dir

        result_files = {}
        for name in results[0]:
            print(f"\nFormating bboxes of {name}")
            results_ = [out[name] for out in results]
            result_files[name] = self._format_bbox(
                results_, osp.join(jsonfile_prefix, name)
            )
        return result_files, tmp_dir

    def load_gt(self):
        """Build the GT :class:`EvalBoxes` from ``data_infos``."""
        all_annotations = EvalBoxes()
        for info in self.data_infos:
            sample_boxes = []
            gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float32).reshape(-1, 7)
            gt_names = np.asarray(info["gt_names"]).reshape(-1)
            gt_velocity = np.asarray(
                info.get("gt_velocity", np.zeros((len(gt_boxes), 2))), dtype=np.float32
            ).reshape(-1, 2)
            num_pts = np.asarray(
                info.get("num_lidar_pts", np.full(len(gt_boxes), -1))
            ).reshape(-1)

            for j in range(len(gt_boxes)):
                class_name = gt_names[j]
                if not self._in_eval_range(
                    gt_boxes[j, 0], gt_boxes[j, 1], class_name
                ):
                    continue

                heading = -gt_boxes[j, 6] - np.pi / 2
                sample_boxes.append(
                    DetectionBox(
                        sample_token=info["token"],
                        translation=tuple(gt_boxes[j, 0:3].tolist()),
                        size=tuple(gt_boxes[j, 3:6].tolist()),
                        rotation=list(Quaternion(axis=[0, 0, 1], radians=heading)),
                        velocity=tuple(gt_velocity[j].tolist()),
                        num_pts=int(num_pts[j]),
                        detection_name=class_name,
                        detection_score=-1.0,
                        attribute_name=class_name,
                    )
                )
            all_annotations.add_boxes(info["token"], sample_boxes)
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
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            result_path = result_files.get("pts_bbox", list(result_files.values())[0])
        else:
            result_path = result_files

        with open(result_path) as f:
            result_data = json.load(f)

        pred_boxes = EvalBoxes.deserialize(result_data["results"], DetectionBox)
        gt_boxes = self.load_gt()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg["class_names"]:
            for dist_th in self.eval_cfg["dist_ths"]:
                metric_data_list.set(
                    class_name,
                    dist_th,
                    accumulate(
                        gt_boxes, pred_boxes, class_name, center_distance, dist_th
                    ),
                )

        metrics = DetectionMetrics(self.eval_cfg)
        for class_name in self.eval_cfg["class_names"]:
            for dist_th in self.eval_cfg["dist_ths"]:
                metrics.add_label_ap(
                    class_name,
                    dist_th,
                    calc_ap(
                        metric_data_list[(class_name, dist_th)],
                        self.eval_cfg["min_recall"],
                        self.eval_cfg["min_precision"],
                    ),
                )
            for metric_name in self.eval_cfg["tp_metrics"]:
                metrics.add_label_tp(
                    class_name,
                    metric_name,
                    calc_tp(
                        metric_data_list[(class_name, self.eval_cfg["dist_th_tp"])],
                        self.eval_cfg["min_recall"],
                        metric_name,
                    ),
                )

        metrics_summary = metrics.serialize()
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

        detail = {}
        prefix = "bbox_NAVSIM"
        for name in self.eval_cfg["class_names"]:
            for k, v in metrics_summary["label_aps"][name].items():
                detail[f"{prefix}/{name}_AP_dist_{k}"] = float(f"{v:.4f}")
            for k, v in metrics_summary["label_tp_errors"][name].items():
                detail[f"{prefix}/{name}_{k}"] = float(f"{v:.4f}")
        for k, v in metrics_summary["tp_errors"].items():
            detail[f"{prefix}/{err_name_mapping.get(k, k)}"] = float(f"{v:.4f}")
        detail[f"{prefix}/NDS"] = metrics_summary["nd_score"]
        detail[f"{prefix}/mAP"] = metrics_summary["mean_ap"]

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return detail
