"""
Unit tests for segment_prostate.py — pure-Python logic only.
No GPU, DICOM data, or running server required.
"""

import gzip
import io
import struct
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

# Allow importing segment_prostate from backend/
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
import segment_prostate as sp


# ── write_status ──────────────────────────────────────────────────────────────

def test_write_status_creates_file(tmp_path):
    sp.write_status(tmp_path, "running|50|Doing stuff")
    assert (tmp_path / "status.txt").read_text() == "running|50|Doing stuff"


def test_write_status_atomic_overwrite(tmp_path):
    sp.write_status(tmp_path, "first")
    sp.write_status(tmp_path, "second")
    assert (tmp_path / "status.txt").read_text() == "second"
    assert not (tmp_path / "status.tmp").exists()


# ── _find_dicom_dir ───────────────────────────────────────────────────────────

def test_find_dicom_dir_picks_largest(tmp_path):
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "a.dcm").touch()

    dense = tmp_path / "dense"
    dense.mkdir()
    for i in range(10):
        (dense / f"{i}.dcm").touch()

    assert sp._find_dicom_dir(tmp_path) == str(dense)


def test_find_dicom_dir_no_files_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No DICOM files found"):
        sp._find_dicom_dir(tmp_path)


def test_find_dicom_dir_ignores_dotfiles(tmp_path):
    (tmp_path / ".hidden").touch()
    with pytest.raises(RuntimeError, match="No DICOM files found"):
        sp._find_dicom_dir(tmp_path)


# ── _normalize_labels ─────────────────────────────────────────────────────────

def test_normalize_labels_passthrough():
    arr = np.array([0, 1, 2, 3], dtype=np.uint8)
    np.testing.assert_array_equal(sp._normalize_labels(arr), arr)


def test_normalize_labels_255_to_1():
    arr = np.array([0, 255, 0, 255], dtype=np.uint8)
    np.testing.assert_array_equal(sp._normalize_labels(arr), [0, 1, 0, 1])


def test_normalize_labels_binary_unchanged():
    arr = np.array([0, 1, 0, 1], dtype=np.uint8)
    np.testing.assert_array_equal(sp._normalize_labels(arr), arr)


# ── _extract_nifti_from_response ──────────────────────────────────────────────

def _make_gz(data: bytes = b"fake nifti") -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(data)
    return buf.getvalue()


def test_extract_nifti_bare_gzip():
    gz = _make_gz()
    assert sp._extract_nifti_from_response(gz) == gz


def test_extract_nifti_multipart_strips_boundary():
    gz = _make_gz()
    boundary = b"TestBoundary123"
    raw = (
        b"--" + boundary + b"\r\nContent-Type: application/octet-stream\r\n\r\n"
        + gz
        + b"\r\n--" + boundary + b"--\r\n"
    )
    assert sp._extract_nifti_from_response(raw) == gz


def test_extract_nifti_no_gzip_large_returns_raw():
    data = b"X" * 200
    assert sp._extract_nifti_from_response(data) == data


def test_extract_nifti_no_gzip_small_raises():
    with pytest.raises(RuntimeError):
        sp._extract_nifti_from_response(b"tiny")


# ── _write_stl_binary ─────────────────────────────────────────────────────────

def _simple_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    return verts, faces


def test_write_stl_binary_creates_file(tmp_path):
    verts, faces = _simple_mesh()
    out = tmp_path / "test.stl"
    sp._write_stl_binary(verts, faces, out)
    assert out.exists()


def test_write_stl_binary_correct_size(tmp_path):
    verts, faces = _simple_mesh()
    out = tmp_path / "test.stl"
    sp._write_stl_binary(verts, faces, out)
    data = out.read_bytes()
    # 80-byte header + 4-byte count + n_faces * 50-byte triangle records
    assert len(data) == 80 + 4 + len(faces) * 50


def test_write_stl_binary_face_count(tmp_path):
    verts, faces = _simple_mesh()
    out = tmp_path / "test.stl"
    sp._write_stl_binary(verts, faces, out)
    count = struct.unpack("<I", out.read_bytes()[80:84])[0]
    assert count == len(faces)


# ── seg_to_stl_files (synthetic NIfTI) ───────────────────────────────────────

def _write_nifti(path: Path, arr: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    img = nib.Nifti1Image(arr, affine)
    img.header.set_zooms(spacing)
    nib.save(img, str(path))


def test_seg_to_stl_binary_whole_gland(tmp_path):
    """Binary {0,1} segmentation → treated as Whole Gland only."""
    arr = np.zeros((30, 30, 30), dtype=np.uint8)
    arr[10:20, 10:20, 10:20] = 1
    seg = tmp_path / "seg.nii.gz"
    _write_nifti(seg, arr)

    results = sp.seg_to_stl_files(seg, tmp_path)

    assert len(results) == 1
    assert results[0]["name"] == "Whole Gland"
    assert (tmp_path / "label_1.stl").exists()


def test_seg_to_stl_volume_calculation(tmp_path):
    """10x10x10 cube at 1mm³/vox = 1000 mm³ = 1.0 cc."""
    arr = np.zeros((30, 30, 30), dtype=np.uint8)
    arr[10:20, 10:20, 10:20] = 1
    seg = tmp_path / "seg.nii.gz"
    _write_nifti(seg, arr, spacing=(1.0, 1.0, 1.0))

    results = sp.seg_to_stl_files(seg, tmp_path)
    assert results[0]["volume_cc"] == pytest.approx(1.0, abs=0.05)


def test_seg_to_stl_multi_label(tmp_path):
    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[5:20, 5:20, 5:20] = 1   # whole gland
    arr[6:14, 6:14, 6:14] = 2   # transition zone (inside)
    arr[22:28, 22:28, 22:28] = 3  # peripheral zone
    seg = tmp_path / "seg.nii.gz"
    _write_nifti(seg, arr)

    results = sp.seg_to_stl_files(seg, tmp_path)
    names = {r["name"] for r in results}
    assert {"Whole Gland", "Transition Zone", "Peripheral Zone"} == names


def test_seg_to_stl_empty_raises(tmp_path):
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    seg = tmp_path / "seg.nii.gz"
    _write_nifti(seg, arr)

    with pytest.raises(RuntimeError, match="No anatomical structures"):
        sp.seg_to_stl_files(seg, tmp_path)


def test_seg_to_stl_tiny_region_skipped(tmp_path):
    """Regions with < 50 voxels should be skipped."""
    arr = np.zeros((30, 30, 30), dtype=np.uint8)
    arr[10:20, 10:20, 10:20] = 1   # 1000 vox — kept
    arr[25, 25, 25] = 2             # 1 vox — skipped
    seg = tmp_path / "seg.nii.gz"
    _write_nifti(seg, arr)

    results = sp.seg_to_stl_files(seg, tmp_path)
    ids = [r["id"] for r in results]
    assert 2 not in ids
