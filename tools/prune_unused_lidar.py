"""Delete NAVSIM point clouds that no info file references.

The NAVSIM logs cover every frame, but training and evaluation only ever touch
the official navtrain / navtest token lists (103,288 + 12,146 = 115,434
frames). The rest of the clouds are non-keyframes that nothing reads.

Dry-run by default: it prints what it would remove and stops. Pass ``--apply``
to actually delete.

Usage
-----
    python tools/prune_unused_lidar.py \\
        --lidar-root /home/data/sensor_blobs_lidar_front \\
        --infos data/infos/navsim_infos_train.pkl \\
                data/infos/navsim_infos_val.pkl

    # ... then, after checking the numbers
    python tools/prune_unused_lidar.py ... --apply
"""

import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from os import path as osp


def parse_args():
    p = argparse.ArgumentParser(description="prune unreferenced NAVSIM clouds")
    p.add_argument("--lidar-root", required=True,
                   help="root containing <split>/<log>/MergedPointCloud")
    p.add_argument("--infos", nargs="+", required=True,
                   help="info pickles whose tokens must be kept")
    p.add_argument("--splits", nargs="+", default=["trainval", "test"])
    p.add_argument("--ext", default=".pcd")
    p.add_argument("--apply", action="store_true",
                   help="actually delete (default: dry run)")
    p.add_argument("--workers", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    keep = set()
    for path in args.infos:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            data = data["infos"]
        tokens = {i["token"] for i in data}
        keep |= tokens
        print(f"[prune] {path}: {len(tokens):,} tokens")
    print(f"[prune] keeping {len(keep):,} tokens in total")

    jobs = []
    for split in args.splits:
        base = osp.join(args.lidar_root, split)
        if not osp.isdir(base):
            continue
        jobs += [(split, lg) for lg in sorted(os.listdir(base))]

    def scan(job):
        split, log = job
        d = osp.join(args.lidar_root, split, log, "MergedPointCloud")
        if not osp.isdir(d):
            return 0, 0, 0, 0
        kn = ks = dn = ds = 0
        for name in os.listdir(d):
            if not name.endswith(args.ext):
                continue
            full = osp.join(d, name)
            size = osp.getsize(full)
            if name[: -len(args.ext)] in keep:
                kn += 1
                ks += size
            else:
                dn += 1
                ds += size
                if args.apply:
                    os.remove(full)
        return kn, ks, dn, ds

    with ThreadPoolExecutor(args.workers) as pool:
        res = list(pool.map(scan, jobs))

    kn = sum(r[0] for r in res)
    ks = sum(r[1] for r in res)
    dn = sum(r[2] for r in res)
    ds = sum(r[3] for r in res)

    verb = "deleted" if args.apply else "would delete"
    print(f"[prune] referenced : {kn:9,d} files  {ks / 1e9:7.1f} GB")
    print(f"[prune] {verb:<11s}: {dn:9,d} files  {ds / 1e9:7.1f} GB")
    if not args.apply:
        print("[prune] dry run -- pass --apply to remove them")


if __name__ == "__main__":
    main()
