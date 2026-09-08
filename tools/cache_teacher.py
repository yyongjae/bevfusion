"""Cache BEVFusion raw head outputs for HiP-AD distillation.

Caches TransFusion decoder outputs (heatmap, center, height, dim, rot, vel)
per sample token via a forward hook on the detection head.
BEV features (dense_heatmap) are excluded.

Usage:
    cd /home/yongjae/e2e/bevfusion
    torchpack dist-run -np 4 python tools/cache_teacher.py \
        configs/bench2drive/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \
        runs/stage/latest.pth \
        --cache-dir /home/yongjae/e2e/HiP-AD/data/teacher_cache
"""

import argparse
import os
import pickle

import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval


# TransFusion decoder outputs only — dense_heatmap (BEV feature) excluded
CACHE_KEYS = ["heatmap", "center", "height", "dim", "rot", "vel"]


def parse_args():
    parser = argparse.ArgumentParser(description="Cache BEVFusion teacher outputs")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--cache-dir", required=True,
                        help="output directory for cached teacher outputs")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="which split to cache (default: train)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    # Unknown flags are forwarded to torchpack as config overrides, matching
    # tools/train.py (e.g. --data.test.load_interval 100).
    args, opts = parser.parse_known_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args, opts


def main():
    args, opts = parse_args()
    dist.init()
    rank, world_size = get_dist_info()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    set_random_seed(args.seed, deterministic=False)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- config ----
    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # ---- dataset ----
    if args.split == "train":
        # data.train is either the dataset itself or a CBGSDataset wrapper.
        train_cfg = cfg.data.train
        cfg.data.test.ann_file = train_cfg.get(
            "dataset", train_cfg
        )["ann_file"]

    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", None)
    samples_per_gpu = 1  # force batch=1 per GPU to avoid OOM during caching

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # ---- model ----
    cfg.model.train_cfg = None
    if cfg.model.encoders.get("camera") is not None:
        cfg.model.encoders.camera.backbone.init_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()

    # ---- hook on detection head to capture raw pred_dict ----
    # pred_dict[0]: dict of batched tensors [B, C, num_proposals]
    _captured = {}

    def _hook_fn(module, inp, output):
        raw = output
        while isinstance(raw, (list, tuple)):
            raw = raw[0]
        if not isinstance(raw, dict):
            raise TypeError(f"unexpected object head output type: {type(raw)}")
        # TransFusion decoding mutates centre/dimension tensors in place after
        # this hook returns, so references to the original output are unsafe.
        _captured["pred_dict"] = {
            key: raw[key].detach().clone()
            for key in CACHE_KEYS
            if key in raw
        }

    hook_handle = model.module.heads["object"].register_forward_hook(_hook_fn)

    # ---- run inference & cache ----
    all_outputs = {}
    prog_bar = mmcv.ProgressBar(len(dataset)) if rank == 0 else None

    with torch.no_grad():
        for data in data_loader:
            _captured.clear()
            try:
                model(return_loss=False, rescale=True, **data)
            except Exception as e:
                raise RuntimeError(
                    f"teacher inference failed on rank {rank}"
                ) from e

            pred_dict = _captured.get("pred_dict")
            if pred_dict is None:
                raise RuntimeError("object head hook did not capture predictions")

            # Navigate through any list/tuple nesting to reach the actual prediction dict
            raw = pred_dict
            while isinstance(raw, (list, tuple)):
                raw = raw[0]
            batch_size = raw[next(iter(raw))].shape[0]

            metas = data.get("metas", data.get("img_metas"))
            if hasattr(metas, "data"):
                metas = metas.data[0]

            for i in range(batch_size):
                token = None
                if isinstance(metas, list) and len(metas) > i:
                    token = metas[i].get("token", None)
                if token is None:
                    raise RuntimeError(
                        f"sample metadata has no token on rank {rank}, batch index {i}"
                    )

                sample = {
                    key: raw[key][i].detach().cpu().half()
                    for key in CACHE_KEYS
                    if key in raw
                }
                all_outputs[token] = sample

            if prog_bar is not None:
                for _ in range(batch_size):
                    prog_bar.update()

    hook_handle.remove()

    # Save per-rank
    save_path = os.path.join(args.cache_dir, f"teacher_cache_rank{rank}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_outputs, f)
    print(f"\n[Rank {rank}] Saved {len(all_outputs)} samples to {save_path}")

    # Rank 0 merges all ranks
    torch.distributed.barrier()
    if rank == 0:
        merged = {}
        for r in range(world_size):
            path = os.path.join(args.cache_dir, f"teacher_cache_rank{r}.pkl")
            with open(path, "rb") as f:
                merged.update(pickle.load(f))
        save_path = os.path.join(args.cache_dir, "teacher_cache.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(merged, f)
        for r in range(world_size):
            os.remove(os.path.join(args.cache_dir, f"teacher_cache_rank{r}.pkl"))
        print(f"Merged all ranks: {len(merged)} samples -> {save_path}")


if __name__ == "__main__":
    main()
