#!/usr/bin/env python3
"""
Smoke-check a Prostate3D output folder.

Usage:
  python scripts/smoke_check_outputs.py /path/to/outputs/<job_id>
"""

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        fail("Usage: python scripts/smoke_check_outputs.py <job_output_dir>")

    out_dir = Path(sys.argv[1]).expanduser().resolve()
    if not out_dir.exists():
        fail(f"Output directory not found: {out_dir}")

    seg_path = out_dir / "segmentation.nii.gz"
    if not seg_path.exists():
        fail(f"Missing segmentation file: {seg_path}")

    img = sitk.ReadImage(str(seg_path))
    arr = np.round(sitk.GetArrayFromImage(img)).astype(np.uint8)
    labels = set(np.unique(arr).tolist())

    allowed = {0, 1, 2, 3}
    if not labels.issubset(allowed):
        fail(f"Unexpected labels: {sorted(labels)} (allowed: {sorted(allowed)})")
    if np.sum(arr > 0) == 0:
        fail("Segmentation is empty (all background).")

    expected_pngs = [
        "view_anterior.png",
        "view_lateral.png",
        "view_oblique.png",
        "view_superior.png",
        "composite.png",
    ]
    missing = [name for name in expected_pngs if not (out_dir / name).exists()]
    if missing:
        fail(f"Missing render outputs: {missing}")

    print("[OK] Output smoke check passed")
    print(f"  dir: {out_dir}")
    print(f"  labels: {sorted(labels)}")
    print(f"  foreground voxels: {int(np.sum(arr > 0))}")


if __name__ == "__main__":
    main()
