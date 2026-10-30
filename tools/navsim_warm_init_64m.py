"""Turn the 50 x 100 (0-32 m) NAVSIM teacher into a --load_from init for the
64 m / 100 x 100 config (camera+lidar/swint_convfuser_64m.yaml).

Why a converted file instead of loading epoch_20.pth directly: the camera
vtransform keeps its BEV grid as non-trainable nn.Parameters (base.py:56-58,
89), so they are in the state_dict. dx/bx/nx have the same shape (3,) in both
grids and would load *silently*, putting the camera back on the 32 m grid
(nx[0] 100 -> 50 BEV rows after downsample) while the LiDAR branch produces
100; the fuser then fails on the concat. They are dropped here and recomputed
from the config at build time.

Everything else is spatially agnostic (convolutions, Swin, TransFusion MLPs)
and is copied unchanged, except the depthnet output conv, whose rows are
[D depth logits | C feature channels]. dbound [1, 50, 0.5] -> [1, 72, 0.5]
keeps the first 98 depth bins identical, so:
  - rows 0..96 (1.0-49.0 m) and the C=80 feature rows are copied as they are
  - the old last bin (49.5 m, which absorbed everything beyond) is split over
    itself and the 44 new bins (50.0-71.5 m): same weights, bias - log(45),
    so the softmax mass beyond 49.5 m is preserved exactly at init.

usage: python tools/navsim_warm_init_64m.py ckpts/epoch_20.pth OUT.pth
"""
import argparse
import math

import numpy as np
import torch

OLD_DBOUND = (1.0, 50.0, 0.5)
NEW_DBOUND = (1.0, 72.0, 0.5)
C = 80
GRID_KEYS = ("dx", "bx", "nx", "frustum")
PREFIX = "encoders.camera.vtransform."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    args = p.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"])

    d_old = len(np.arange(*OLD_DBOUND))
    d_new = len(np.arange(*NEW_DBOUND))
    assert np.allclose(np.arange(*OLD_DBOUND), np.arange(*NEW_DBOUND)[:d_old])
    n_split = d_new - d_old + 1

    old_nx = sd[PREFIX + "nx"].tolist()
    assert old_nx == [100, 200, 1], f"expected the 0-32 m grid, got nx={old_nx}"
    for k in GRID_KEYS:
        del sd[PREFIX + k]

    w = sd[PREFIX + "depthnet.6.weight"]
    b = sd[PREFIX + "depthnet.6.bias"]
    assert w.shape[0] == d_old + C, w.shape
    w_new = torch.empty((d_new + C,) + tuple(w.shape[1:]), dtype=w.dtype)
    b_new = torch.empty(d_new + C, dtype=b.dtype)
    w_new[: d_old - 1] = w[: d_old - 1]
    b_new[: d_old - 1] = b[: d_old - 1]
    w_new[d_old - 1 : d_new] = w[d_old - 1]
    b_new[d_old - 1 : d_new] = b[d_old - 1] - math.log(n_split)
    w_new[d_new:] = w[d_old:]
    b_new[d_new:] = b[d_old:]
    sd[PREFIX + "depthnet.6.weight"] = w_new
    sd[PREFIX + "depthnet.6.bias"] = b_new

    # Only the weights: --load_from must not carry the old optimizer/epoch.
    out = {
        "state_dict": sd,
        "meta": {
            "source": args.src,
            "source_epoch": ck.get("meta", {}).get("epoch"),
            "note": "50x100 -> 64m/100x100 warm init (navsim_warm_init_64m.py)",
        },
    }
    torch.save(out, args.dst)
    print(
        f"dropped {[PREFIX + k for k in GRID_KEYS]}; depthnet.6 {tuple(w.shape)} -> "
        f"{tuple(w_new.shape)} (depth bins {d_old} -> {d_new}, last bin split x{n_split}); "
        f"{len(sd)} tensors -> {args.dst}"
    )


if __name__ == "__main__":
    main()
