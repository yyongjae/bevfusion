"""Convert NAVSIM ``MergedPointCloud/*.pcd`` to compact BEVFusion ``.bin``.

The raw NAVSIM merged point cloud is a PCD with six float32 fields
``(x, y, z, intensity, ring, lidar_id)``.  BEVFusion only ever consumes the
first five channels (and zeroes the fifth), and only points inside
``point_cloud_range``, so this script crops and rewrites each cloud as a raw
``float32`` array of shape ``(N, 5)`` -- the same layout nuScenes uses.

Typical saving is 60-75%% of the original file size.

The output mirrors the input tree, so a cloud at::

    <pcd-root>/trainval/<log>/MergedPointCloud/<token>.pcd

becomes::

    <out-root>/trainval/<log>/MergedPointCloud/<token>.bin

which is exactly the path ``navsim_converter.py`` writes into the info pkl.

Because the full NAVSIM sensor blobs do not fit on disk, the intended workflow
is to download one log (or one tar shard) at a time and run this script with
``--delete-source``, so the raw ``.pcd`` never accumulates.

Usage
-----
    # size check on 20 clouds before committing to the full download
    python tools/data_converter/navsim_lidar_to_bin.py \
        --pcd-root /path/to/sensor_blobs --out-root data/navsim/lidar_bin \
        --dry-run 20

    # real conversion
    python tools/data_converter/navsim_lidar_to_bin.py \
        --pcd-root /path/to/sensor_blobs --out-root data/navsim/lidar_bin \
        --workers 32 --delete-source
"""

import argparse
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from os import path as osp
from typing import List, Optional

import numpy as np
from tqdm import tqdm

_NUMPY_TYPE = {("F", 4): "f4", ("F", 8): "f8",
               ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
               ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8"}


# --------------------------------------------------------------------------- #
# Minimal PCD reader (binary / binary_compressed / ascii)
# --------------------------------------------------------------------------- #
def _lzf_decompress(data: bytes, expected_len: int) -> bytes:
    """Pure-python LZF decompression (fallback when python-lzf is absent)."""
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:  # literal run
            out += data[i : i + ctrl + 1]
            i += ctrl + 1
        else:
            length = ctrl >> 5
            if length == 7:
                length += data[i]
                i += 1
            ref = len(out) - ((ctrl & 0x1F) << 8) - data[i] - 1
            i += 1
            if ref < 0:
                raise ValueError("invalid LZF stream")
            for _ in range(length + 2):
                out.append(out[ref])
                ref += 1
    if len(out) != expected_len:
        raise ValueError(f"LZF size mismatch: {len(out)} != {expected_len}")
    return bytes(out)


def read_pcd(path: str) -> np.ndarray:
    """Read a PCD file into a structured numpy array."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"truncated PCD header: {path}")
            text = line.decode("utf-8", "replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, value = text.partition(" ")
            header[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break

        fields = header["FIELDS"].split()
        sizes = [int(s) for s in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(c) for c in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
        n_points = int(header["POINTS"]) if "POINTS" in header else (
            int(header["WIDTH"]) * int(header["HEIGHT"])
        )
        data_kind = header["DATA"].lower()

        names, formats = [], []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            fmt = _NUMPY_TYPE[(typ.upper(), size)]
            names.append(name)
            formats.append(f"{count}{fmt}" if count > 1 else fmt)
        dtype = np.dtype({"names": names, "formats": formats})

        if data_kind == "binary":
            buf = f.read(dtype.itemsize * n_points)
            return np.frombuffer(buf, dtype=dtype, count=n_points)

        if data_kind == "binary_compressed":
            comp_len, uncomp_len = struct.unpack("II", f.read(8))
            payload = f.read(comp_len)
            try:
                import lzf  # type: ignore

                raw = lzf.decompress(payload, uncomp_len)
            except ImportError:
                raw = _lzf_decompress(payload, uncomp_len)
            # binary_compressed is stored field-major.
            out = np.empty(n_points, dtype=dtype)
            offset = 0
            for name in dtype.names:
                sub = dtype[name]
                nbytes = sub.itemsize * n_points
                out[name] = np.frombuffer(
                    raw[offset : offset + nbytes], dtype=sub, count=n_points
                )
                offset += nbytes
            return out

        if data_kind == "ascii":
            return np.loadtxt(f, dtype=dtype)

        raise ValueError(f"unsupported PCD DATA type: {data_kind}")


# --------------------------------------------------------------------------- #
def pcd_to_points(path: str) -> np.ndarray:
    """Return an ``(N, 5)`` float32 array ``[x, y, z, intensity, ring]``."""
    arr = read_pcd(path)
    names = arr.dtype.names

    def col(name, default=0.0):
        return (
            arr[name].astype(np.float32)
            if name in names
            else np.full(len(arr), default, dtype=np.float32)
        )

    return np.stack(
        [col("x"), col("y"), col("z"), col("intensity"), col("ring")], axis=-1
    ).astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Keep one point per ``voxel``-sized cell (order preserving)."""
    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(keep)]


def convert_one(job):
    src, dst, pc_range, voxel, delete_source, overwrite, dims = job
    try:
        if osp.exists(dst) and not overwrite:
            return osp.getsize(dst), 0, 0, None
        points = pcd_to_points(src)
        n_in = len(points)

        if pc_range is not None:
            x0, y0, z0, x1, y1, z1 = pc_range
            m = (
                (points[:, 0] >= x0) & (points[:, 0] <= x1)
                & (points[:, 1] >= y0) & (points[:, 1] <= y1)
                & (points[:, 2] >= z0) & (points[:, 2] <= z1)
            )
            points = points[m]
        if voxel and voxel > 0:
            points = voxel_downsample(points, voxel)
        points = points[:, :dims]

        os.makedirs(osp.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        points.astype(np.float32).tofile(tmp)
        os.replace(tmp, dst)

        if delete_source:
            os.remove(src)
        return osp.getsize(dst), n_in, len(points), None
    except Exception as exc:  # noqa: BLE001
        return 0, 0, 0, f"{src}: {exc}"


def find_pcds(pcd_root: str, splits: List[str]) -> List[str]:
    out = []
    for split in splits:
        base = osp.join(pcd_root, split)
        if not osp.isdir(base):
            continue
        for log in sorted(os.listdir(base)):
            d = osp.join(base, log, "MergedPointCloud")
            if not osp.isdir(d):
                continue
            out.extend(
                osp.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".pcd")
            )
    return out


def parse_args():
    p = argparse.ArgumentParser(description="NAVSIM pcd -> BEVFusion bin")
    p.add_argument("--pcd-root", required=True,
                   help="sensor-blob root containing <split>/<log>/MergedPointCloud")
    p.add_argument("--out-root", required=True, help="output root for .bin files")
    p.add_argument("--splits", nargs="+", default=["trainval", "test"])
    p.add_argument("--point-cloud-range", nargs=6, type=float,
                   default=[-51.2, -51.2, -3.0, 51.2, 51.2, 5.0],
                   help="crop range; must match the training config")
    p.add_argument("--dims", type=int, default=5, choices=[4, 5],
                   help="channels to store: 4 = [x, y, z, intensity], "
                        "5 = + ring. LoadPointsFromMultiSweeps zeroes the 5th "
                        "channel anyway, so 4 costs nothing at train time and "
                        "saves 20%% of disk -- but it needs load_dim/use_dim/"
                        "in_channels: 4 in the config and gives up loading the "
                        "nuScenes LiDAR-pretrained weights (default: 5)")
    p.add_argument("--voxel", type=float, default=0.0,
                   help="optional voxel size for downsampling (e.g. 0.05); "
                        "0 disables it")
    p.add_argument("--tokens", default=None,
                   help="optional .txt/.pkl listing the tokens to keep "
                        "(one per line for .txt)")
    p.add_argument("--delete-source", action="store_true",
                   help="remove each .pcd after a successful conversion")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", type=int, default=0,
                   help="convert only N clouds and report the size statistics")
    p.add_argument("--project-frames", type=int, default=115434,
                   help="frame count used for the projected-size estimate "
                        "(default: the 103,288 navtrain + 12,146 navtest "
                        "official-split frames)")
    return p.parse_args()


def load_token_filter(path: Optional[str]) -> Optional[set]:
    if path is None:
        return None
    if path.endswith(".pkl"):
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            data = data.get("infos", [])
        return {i["token"] for i in data} if data and isinstance(data[0], dict) else set(data)
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def main():
    args = parse_args()
    tokens = load_token_filter(args.tokens)

    srcs = find_pcds(args.pcd_root, args.splits)
    if tokens is not None:
        srcs = [s for s in srcs if osp.splitext(osp.basename(s))[0] in tokens]
    if args.dry_run:
        srcs = srcs[: args.dry_run]

    if not srcs:
        print("[navsim_lidar] no .pcd files found -- check --pcd-root/--splits")
        sys.exit(1)

    print(f"[navsim_lidar] {len(srcs)} clouds -> {args.out_root}")
    jobs = []
    for src in srcs:
        rel = osp.relpath(src, args.pcd_root)
        dst = osp.join(args.out_root, osp.splitext(rel)[0] + ".bin")
        jobs.append((src, dst, args.point_cloud_range, args.voxel,
                     args.delete_source and not args.dry_run, args.overwrite,
                     args.dims))

    total_out, total_in_pts, total_out_pts, errors = 0, 0, 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for size, n_in, n_out, err in tqdm(
            pool.map(convert_one, jobs), total=len(jobs), desc="pcd"
        ):
            total_out += size
            total_in_pts += n_in
            total_out_pts += n_out
            if err:
                errors.append(err)

    n_ok = len(jobs) - len(errors)
    for e in errors[:10]:
        print(f"[error] {e}")
    if len(errors) > 10:
        print(f"[error] ... and {len(errors) - 10} more")

    if n_ok:
        mb = total_out / n_ok / 1e6
        keep = 100 * total_out_pts / max(total_in_pts, 1)
        print(f"[navsim_lidar] {n_ok} converted, mean {mb:.2f} MB/frame")
        print(f"[navsim_lidar] points: {total_in_pts / n_ok:.0f} in -> "
              f"{total_out_pts / n_ok:.0f} out per frame ({keep:.1f}% kept) "
              f"with range={args.point_cloud_range} voxel={args.voxel} "
              f"dims={args.dims}")
        print(f"[navsim_lidar] projected size for {args.project_frames:,} "
              f"frames: {mb * args.project_frames / 1000:.1f} GB")


if __name__ == "__main__":
    main()
