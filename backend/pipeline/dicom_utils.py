"""
DICOM Utilities
================
Find T2 series, load as NIfTI
"""

import os
import pydicom
import SimpleITK as sitk
import numpy as np
from pathlib import Path


def find_best_series(dicom_folder):
    """Scan DICOM folder and return path to best T2-weighted series"""
    candidates = []

    for root, dirs, files in os.walk(dicom_folder):
        dcm_files = [f for f in files if not f.startswith('.')]
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(
                os.path.join(root, dcm_files[0]),
                stop_before_pixels=True
            )
            modality = getattr(ds, 'Modality', '')
            if modality != 'MR':
                continue

            ntemp    = getattr(ds, 'NumberOfTemporalPositions', None)
            scan_seq = str(getattr(ds, 'ScanningSequence', ''))
            desc     = str(getattr(ds, 'SeriesDescription', '')).upper()
            thick    = float(getattr(ds, 'SliceThickness', 99))
            sp       = getattr(ds, 'PixelSpacing', None)

            # Skip dynamic sequences
            if ntemp and int(str(ntemp)) > 3:
                continue

            score = 0
            if any(x in desc for x in ['T2', 'TSE', 'TRA', 'AX']):
                score += 10
            if 'SE' in scan_seq and 'EP' not in scan_seq:
                score += 5
            if 1.5 <= thick <= 5.0:
                score += 3
            if sp and 0.3 <= float(sp[0]) <= 1.0:
                score += 2
            if 16 <= len(dcm_files) <= 80:
                score += 2

            candidates.append({
                'folder': root,
                'score':  score,
                'files':  len(dcm_files),
                'desc':   desc,
                'thick':  thick,
            })
        except Exception:
            pass

    if not candidates:
        raise RuntimeError(
            "No MR series found in DICOM folder. "
            "Make sure you have T2-weighted MRI DICOM files."
        )

    best = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
    print(f"[DICOM] Best series: {best['folder']} "
          f"(score={best['score']}, files={best['files']}, desc={best['desc']})")
    return best['folder']


def load_series_as_nifti(series_folder, output_path):
    """Load DICOM series and save as NIfTI"""
    reader     = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(series_folder)

    if series_ids:
        # Pick the series with the most files for stable behavior when multiple
        # sub-series are present in the same folder.
        best_series = max(
            series_ids,
            key=lambda sid: len(reader.GetGDCMSeriesFileNames(series_folder, sid))
        )
        files = reader.GetGDCMSeriesFileNames(series_folder, best_series)
    else:
        files = sorted([
            os.path.join(series_folder, f)
            for f in os.listdir(series_folder)
            if not f.startswith('.')
        ])

    reader.SetFileNames(files)
    image = reader.Execute()

    # Handle vector images (e.g., multi-channel): keep first component.
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorIndexSelectionCast(image, 0)

    # Handle 4D temporal data: keep first timepoint as a 3D volume.
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        index = [0, 0, 0, 0]
        size[3] = 0  # collapse time dimension
        image = sitk.Extract(image, size, index)

    sitk.WriteImage(image, output_path)
    sp = image.GetSpacing()
    sz = image.GetSize()
    print(f"[DICOM] Saved NIfTI: size={sz} spacing={[round(s,2) for s in sp]}")
    return output_path
