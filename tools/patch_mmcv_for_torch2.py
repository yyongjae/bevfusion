"""Make mmcv-full 1.7.2 work with torch >= 2.1.

docs/NAVSIM.md pins torch 1.10.2+cu113, which has no sm_120 kernels, so on
Blackwell cards (RTX 5090) the stack has to move to cu128 / torch 2.8.  mmcv
1.7.2 is the newest 1.x and builds fine from source against it, but two of its
DataParallel helpers still use torch APIs that 2.x changed.  Both edits are
in site-packages, so re-run this after any mmcv reinstall.

    python tools/patch_mmcv_for_torch2.py

The script is idempotent and reports what it changed.
"""

import os
import sys

import mmcv

PATCHES = [
    (
        os.path.join("parallel", "_functions.py"),
        "            streams = [_get_stream(device) for device in target_gpus]",
        "            # torch>=2.0 takes a torch.device here, mmcv 1.x passes an int.\n"
        "            streams = [_get_stream(torch.device('cuda', device))\n"
        "                       for device in target_gpus]",
    ),
    (
        os.path.join("parallel", "distributed.py"),
        "        module_to_run = self._replicated_tensor_module if \\\n"
        "            self._use_replicated_tensor_module else self.module",
        "        # torch>=2.1 dropped the replicated-tensor DDP path together with\n"
        "        # these two attributes; fall back to the plain module when absent.\n"
        "        if getattr(self, '_use_replicated_tensor_module', False):\n"
        "            module_to_run = self._replicated_tensor_module\n"
        "        else:\n"
        "            module_to_run = self.module",
    ),
]


def main() -> int:
    root = os.path.dirname(mmcv.__file__)
    print(f"mmcv {mmcv.__version__} at {root}")
    failed = False
    for rel, old, new in PATCHES:
        path = os.path.join(root, rel)
        source = open(path).read()
        if new in source:
            print(f"  already patched: {rel}")
        elif source.count(old) == 1:
            open(path, "w").write(source.replace(old, new, 1))
            print(f"  patched:         {rel}")
        else:
            print(f"  FAILED, pattern not found ({source.count(old)} matches): {rel}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
