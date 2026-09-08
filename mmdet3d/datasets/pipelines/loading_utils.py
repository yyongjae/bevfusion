import os

import numpy as np
import torch

__all__ = [
    "load_augmented_point_cloud",
    "reduce_LiDAR_beams",
    "read_pcd",
    "load_pcd_point_cloud",
]


def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    # NOTE: following Tianwei's implementation, it is hard coded for nuScenes
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    # NOTE: path definition different from Tianwei's implementation.
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points",
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )
    assert os.path.exists(seg_path)
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict["real_points"]
    # NOTE: add zero reflectance to virtual points instead of removing them from real points
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],
            np.zeros([data_dict["virtual_points"].shape[0], 1]),
            data_dict["virtual_points"][:, 3:],
        ],
        axis=-1,
    )

    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
        ],
        axis=1,
    )
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    # print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    # [?] pick the 14th beam
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    else:
        raise NotImplementedError
    # points = copy.copy(pts)
    points = pts[mask]
    # print(points.size())
    return points.numpy()



# --------------------------------------------------------------------------- #
# PCD reader
#
# NAVSIM ships its merged point clouds as PCD rather than the raw float32 dumps
# nuScenes uses.  Reading them directly avoids duplicating the whole LiDAR set
# on disk -- the NAVSIM encoding (float32 xyz + uint8 intensity/lidar_info/ring
# = 15 bytes/point) is actually *denser* than a float32 .bin would be.
# --------------------------------------------------------------------------- #

_PCD_NUMPY_TYPE = {
    ("F", 4): "f4", ("F", 8): "f8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
}


def _lzf_decompress(data, expected_len):
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


def read_pcd(path):
    """Read a PCD file into a structured numpy array.

    Supports the ``binary``, ``binary_compressed`` and ``ascii`` encodings and
    arbitrary field layouts; callers select fields by name.
    """
    import struct

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
        sizes = [int(x) for x in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [
            int(c)
            for c in header.get("COUNT", " ".join(["1"] * len(fields))).split()
        ]
        n_points = (
            int(header["POINTS"])
            if "POINTS" in header
            else int(header["WIDTH"]) * int(header["HEIGHT"])
        )
        data_kind = header["DATA"].lower()

        names, formats = [], []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            fmt = _PCD_NUMPY_TYPE[(typ.upper(), size)]
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
                import lzf

                raw = lzf.decompress(payload, uncomp_len)
            except ImportError:
                raw = _lzf_decompress(payload, uncomp_len)
            # binary_compressed stores the data field-major.
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


def load_pcd_point_cloud(path, load_dim=5):
    """Load a PCD as an ``(N, load_dim)`` float32 array.

    Columns follow the nuScenes layout ``[x, y, z, intensity, ring]``; missing
    fields are filled with zeros.
    """
    arr = read_pcd(path)
    names = arr.dtype.names

    def col(name):
        if name in names:
            return arr[name].astype(np.float32)
        return np.zeros(len(arr), dtype=np.float32)

    points = np.stack(
        [col("x"), col("y"), col("z"), col("intensity"), col("ring")], axis=-1
    )
    return np.ascontiguousarray(points[:, :load_dim], dtype=np.float32)
