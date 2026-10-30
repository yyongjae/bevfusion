"""Correct the student-transform text in a 64 m (100 x 100) teacher cache manifest.

cache_teacher_bev.py writes ``student_bev = teacher_bev[:, :, ::-1]`` whenever the
teacher grid is 100 x 100, which is only right for a student that also covers
0-64 m at 0.64 m.  PARA-SSR's 100 x 100 grid covers 0-32 m forward at 0.32 m, so it
needs the near half of the teacher rows resampled 50 -> 100.

    python tools/fix_manifest_64m.py <cache-dir>/manifest.json
"""
import json
import sys

path = sys.argv[1]
m = json.load(open(path))
assert m["source_bev_shape"] == [100, 100], m["source_bev_shape"]
assert m["point_cloud_range"][:2] == [0.0, -32.0] and m["point_cloud_range"][3:5] == [64.0, 32.0], m["point_cloud_range"]
m["to_student_transform"] = (
    "Teacher rows 0..49 are x = 0..32 m, rows 50..99 are x = 32..64 m. "
    "0-32 m consumers on the 50x100 S grid (refiner adapters BEV_H=50, PARA-SSR 50x100 "
    "student): student_bev = teacher_bev[:, :50, ::-1] (crop, do NOT resize 100 -> 50). "
    "PARA-SSR 100x100 (0-32 m at 0.32 m): student_bev = resample(teacher_bev[:, :50, ::-1], "
    "longitudinal 50 -> 100). 0-64 m student at 0.64 m (e.g. corridor c64): "
    "student_bev = teacher_bev[:, :, ::-1]."
)
m["teacher_range_m"] = {"x_forward": [0.0, 64.0], "y_left": [-32.0, 32.0]}
json.dump(m, open(path, "w"), indent=2)
print("fixed", path)
