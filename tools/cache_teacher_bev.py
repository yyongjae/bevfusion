"""Cache BEVFusion BEV features and predictions for PARA-SSR distillation.

Writes one ``.npz`` per NAVSIM frame token, bucketed by the token's first two
characters, plus a ``manifest.json`` describing the geometry and conventions a
consumer needs in order to use the tensors correctly.

    <cache-dir>/
        manifest.json
        samples/
            43/431ae29947e95c26.npz
            ...

Contents of each npz
--------------------
    bev_feature     (256, H, W)     float16  the tensor entering the detection
                                             decoder, i.e. the fused camera +
                                             LiDAR BEV.  H and W come from the
                                             config (grid_size / out_size_factor)
                                             and are recorded in the manifest;
                                             the current NAVSIM grid is 50 x 100.
    dense_heatmap   (7, H, W)       float16  dense per-class BEV heatmap logits
    pred_boxes_3d   (P, 9)          float32  decoded boxes, mmdet3d LiDAR
                                             convention [x, y, z, dx, dy, dz,
                                             yaw, vx, vy], z at gravity centre
    pred_scores_3d  (P,)            float32
    pred_labels_3d  (P,)            int16    index into ``class_names``
    raw_cls_logits  (7, P)          float16  per-query class logits
    raw_center      (2, P)          float16  BEV cell units, not metres
    raw_height      (1, P)          float16
    raw_dim         (3, P)          float16
    raw_rot         (2, P)          float16  (sin, cos)
    raw_vel         (2, P)          float16

Axis convention -- read this before using ``bev_feature``
---------------------------------------------------------
The teacher is mmdet3d LiDAR: ``x`` forward, ``y`` left, so the tensor is
``(C, H = x_forward, W = y_left)``.  PARA-SSR is ``x`` right, ``y`` forward, so
its ``bev_embed`` is ``(C, bev_h = y_forward, bev_w = x_right)``.

Both cover 32 m forward and +-32 m laterally.  The lateral axes line up cell
for cell (0.64 m), and only the *direction* differs -- left versus right -- so
the lateral conversion is a flip with no transpose::

    student_bev = teacher_bev[:, :, ::-1]

The longitudinal axes line up only when the two grids agree.  On the current
50 x 100 teacher grid (0.64 m square cells) against PARA-SSR's 100 x 100
(0.32 m longitudinal), they do not: the student's forward axis is twice as fine,
so a consumer must either resample 50 -> 100 along it or move the student to
50 x 100, which its own config comment anticipates.  ``manifest.json`` records
the teacher's shape and states the required transform.

Usage
-----
    torchpack dist-run -np 4 python tools/cache_teacher_bev.py \\
        configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \\
        runs/navsim-fusion/epoch_20.pth \\
        --cache-dir data/teacher_cache/cache_train_100x100 --split train

``--split train`` caches ``data.train``'s ann_file through the test pipeline
(no augmentation); ``--split val`` uses ``data.val``.  Re-running with
``--skip-existing`` resumes an interrupted run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model
from torchpack import distributed as dist
from torchpack.utils.config import configs

from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

# Head outputs, in the npz names the student side expects.
RAW_KEYS = {
    "heatmap": "raw_cls_logits",
    "center": "raw_center",
    "height": "raw_height",
    "dim": "raw_dim",
    "rot": "raw_rot",
    "vel": "raw_vel",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="cache BEVFusion BEV features and predictions"
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="teacher checkpoint")
    parser.add_argument("--cache-dir", required=True, help="output directory")
    parser.add_argument(
        "--split", default="train", choices=["train", "val"],
        help="which ann_file to run over (always through the test pipeline)",
    )
    parser.add_argument(
        "--bev-dtype", default="float16", choices=["float16", "float32"],
        help="float16 halves the 4.9 MB/sample cost and is what the student "
             "consumes; float32 is only useful for numerical debugging",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="np.savez_compressed. Dense conv features compress poorly and it "
             "costs CPU per sample, so it is off by default",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="leave already-written npz files alone (resume an interrupted run)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="stop after this many samples per rank; for smoke tests",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_known_args()


def sha256_head(path, n_bytes=1 << 20):
    """Hash of the first MiB -- enough to tell checkpoints apart cheaply."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    args, opts = parse_args()
    dist.init()
    rank, world_size = get_dist_info()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    set_random_seed(args.seed, deterministic=False)

    samples_dir = os.path.join(args.cache_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # Always run the *test* pipeline: caching must see the same deterministic
    # geometry the student will be distilled against, not a sampled
    # augmentation.  Only the annotation file changes between splits.
    if args.split == "train":
        train_cfg = cfg.data.train
        ann_file = train_cfg.get("dataset", train_cfg)["ann_file"]
    else:
        ann_file = cfg.data.val["ann_file"]
    cfg.data.test["ann_file"] = ann_file
    cfg.data.test["test_mode"] = True
    cfg.data.test.pop("samples_per_gpu", None)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,  # one sample per step keeps the BEV hook unambiguous
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    if cfg.model.encoders.get("camera") is not None:
        cfg.model.encoders.camera.backbone.init_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDistributedDataParallel(
        model.cuda(), device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()

    # The fused BEV is whatever enters the detection decoder.  Hooking the
    # decoder rather than the fuser keeps this correct for the LiDAR-only and
    # camera-only configs too, where `fuser` is null and the decoder is fed by a
    # single encoder.
    captured = {}

    def capture_bev(module, inputs):
        captured["bev"] = inputs[0].detach()
        return None

    handle = model.module.decoder["backbone"].register_forward_pre_hook(capture_bev)

    bev_dtype = np.float16 if args.bev_dtype == "float16" else np.float32
    save = np.savez_compressed if args.compress else np.savez

    written = skipped = 0
    prog_bar = mmcv.ProgressBar(len(dataset)) if rank == 0 else None
    started = time.time()

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            if args.limit is not None and step >= args.limit:
                break

            metas = data.get("metas", data.get("img_metas"))
            if hasattr(metas, "data"):
                metas = metas.data[0]
            token = metas[0].get("token")
            if not token:
                raise RuntimeError(f"rank {rank}: sample {step} has no token")

            out_path = os.path.join(samples_dir, token[:2], f"{token}.npz")
            if args.skip_existing and os.path.exists(out_path):
                skipped += 1
                if prog_bar is not None:
                    prog_bar.update()
                continue

            captured.clear()
            outputs = model(return_loss=False, rescale=True, **data)
            bev = captured.get("bev")
            if bev is None:
                raise RuntimeError(f"rank {rank}: decoder hook captured no BEV")

            out = outputs[0]
            boxes = out["boxes_3d"]
            arrays = {
                "bev_feature": bev[0].cpu().numpy().astype(bev_dtype),
                "dense_heatmap": out["dense_heatmap"].cpu().numpy().astype(np.float16),
                "pred_boxes_3d": boxes.tensor.cpu().numpy().astype(np.float32),
                "pred_scores_3d": out["scores_3d"].cpu().numpy().astype(np.float32),
                "pred_labels_3d": out["labels_3d"].cpu().numpy().astype(np.int16),
            }
            for src, dst in RAW_KEYS.items():
                if src in out:
                    arrays[dst] = out[src].cpu().numpy().astype(np.float16)

            # Write to a rank-private temporary name and rename, so an
            # interrupted run never leaves a half-written npz that
            # --skip-existing would then trust.
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            tmp_path = f"{out_path}.rank{rank}.tmp.npz"
            save(tmp_path, **arrays)
            os.replace(tmp_path, out_path)
            written += 1

            if prog_bar is not None:
                prog_bar.update()

    handle.remove()
    elapsed = time.time() - started
    print(f"\n[rank {rank}] wrote {written}, skipped {skipped} in {elapsed/60:.1f} min")

    torch.distributed.barrier()
    if rank != 0:
        return

    total = sum(
        len(os.listdir(os.path.join(samples_dir, b)))
        for b in os.listdir(samples_dir)
        if os.path.isdir(os.path.join(samples_dir, b))
    )
    pcr = list(cfg.point_cloud_range)
    voxel = list(cfg.voxel_size)
    osf = cfg.model.heads.object.train_cfg.out_size_factor
    grid = list(cfg.model.heads.object.train_cfg.grid_size)
    bev_shape = [grid[0] // osf, grid[1] // osf]
    # Source and target are the same grid here: PARA-SSR's bev_h/bev_w match.
    bev_channels = int(cfg.model.decoder.backbone.in_channels)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256_head": sha256_head(args.checkpoint),
        "split": args.split,
        "ann_file": ann_file,
        "num_samples_expected": len(dataset),
        "num_samples_written": total,
        "layout": "samples/<token[:2]>/<token>.npz",
        "source_bev_shape": bev_shape,
        "target_bev_shape": bev_shape,
        "bev_channels": bev_channels,
        "num_proposals": cfg.model.heads.object.num_proposals,
        "class_names": list(cfg.object_classes),
        "point_cloud_range": pcr,
        "voxel_size": voxel,
        "out_size_factor": osf,
        "cell_size_m": {
            "longitudinal_x": voxel[0] * osf,
            "lateral_y": voxel[1] * osf,
        },
        "teacher_axes": "(C, H=x_forward, W=y_left), mmdet3d LiDAR frame",
        "student_axes": "(C, bev_h=y_forward, bev_w=x_right), PARA-SSR frame",
        "student_bev_shape_para_ssr": [100, 100],
        "to_student_transform": (
            "student_bev = teacher_bev[:, :, ::-1]"
            if bev_shape == [100, 100] else
            "student_bev = resample(teacher_bev[:, :, ::-1], "
            f"longitudinal {bev_shape[0]} -> 100); lateral already matches at "
            "0.64 m. Alternatively move PARA-SSR to "
            f"bev_h={bev_shape[0]}, bev_w={bev_shape[1]}."
        ),
        "raw_center_units": "BEV cell indices, add query position; not metres",
        "pred_boxes_3d_format":
            "[x, y, z_gravity_centre, dx, dy, dz, yaw, vx, vy], mmdet3d LiDAR",
        "dtypes": {
            "bev_feature": args.bev_dtype,
            "dense_heatmap": "float16",
            "pred_boxes_3d": "float32",
            "pred_scores_3d": "float32",
            "pred_labels_3d": "int16",
            "raw_*": "float16",
        },
        "compressed": bool(args.compress),
        "world_size": world_size,
    }
    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}: {total}/{len(dataset)} samples")


if __name__ == "__main__":
    main()
