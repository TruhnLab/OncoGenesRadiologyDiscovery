#!/usr/bin/env python3
"""
Segment liver + tumor + lesions for TCGA-LIHC using TotalSegmentator.

Uses three TotalSegmentator tasks:
  - 'total': liver whole organ
  - 'liver_lesions': liver lesions/tumors
  - 'liver_segments': Couinaud segments 1-8

Extracts features: volume, HU stats, heterogeneity, necrotic fraction, etc.

Usage:
    python radiomics/segment_lihc.py [--max-patients N]
"""
import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats, ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset_lihc import TCGALIHCDataset, load_dicom_volume, get_pixel_spacing
import config

log = logging.getLogger(__name__)

RESULTS_DIR = config.RESULTS_DIR / "lihc"
MASKS_DIR = RESULTS_DIR / "segmentation_masks"
FEATURES_CSV = RESULTS_DIR / "liver_features.csv"


def dicom_to_nifti(dicom_dir, volume=None, spacing=None):
    """Convert DICOM series to NIfTI using SimpleITK (handles all orientation edge cases)."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_names)
    sitk_img = reader.Execute()
    # SimpleITK → NIfTI preserves correct orientation, origin, and spacing
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    sitk.WriteImage(sitk_img, tmp.name)
    nifti = nib.load(tmp.name)
    nifti = nib.Nifti1Image(np.asarray(nifti.dataobj), nifti.affine, nifti.header)
    Path(tmp.name).unlink()
    return nifti


def run_totalsegmentator(nifti_path, output_dir, task="total"):
    from totalsegmentator.python_api import totalsegmentator
    # Never use fast mode for liver — liver segmentation needs full resolution
    totalsegmentator(input=nifti_path, output=output_dir, task=task,
                     device="gpu", fast=False, quiet=True)


def load_mask(mask_dir, name, vol_shape):
    p = mask_dir / f"{name}.nii.gz"
    if not p.exists() or p.stat().st_size < 100:
        return None
    try:
        m = nib.load(str(p)).get_fdata().transpose(2, 1, 0).astype(bool)
        return m if m.shape == vol_shape else None
    except Exception:
        return None


def compute_features(volume, mask, name, spacing):
    prefix = f"{name}_"
    if mask is None or not np.any(mask):
        keys = ['volume_ml', 'hu_mean', 'hu_std', 'hu_skew', 'hu_p10', 'hu_p90',
                'heterogeneity', 'necrotic_frac', 'enhancing_frac', 'ruggedness', 'n_voxels']
        return {f"{prefix}{k}": 0.0 for k in keys}

    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    vals = volume[mask].astype(float)
    n_voxels = int(np.sum(mask))

    # Ruggedness
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    sa = sum(np.sum(np.abs(np.diff(padded, axis=ax))) *
             np.prod([spacing[i] for i in range(3) if i != ax]) for ax in range(3))
    vol_mm3 = n_voxels * voxel_vol

    return {
        f"{prefix}volume_ml": vol_mm3 / 1000,
        f"{prefix}hu_mean": float(np.mean(vals)),
        f"{prefix}hu_std": float(np.std(vals)),
        f"{prefix}hu_skew": float(stats.skew(vals)),
        f"{prefix}hu_p10": float(np.percentile(vals, 10)),
        f"{prefix}hu_p90": float(np.percentile(vals, 90)),
        f"{prefix}heterogeneity": float(np.std(vals) / (abs(np.mean(vals)) + 1e-10)),
        f"{prefix}necrotic_frac": float(np.mean(vals < 20)),
        f"{prefix}enhancing_frac": float(np.mean(vals > 150)),
        f"{prefix}ruggedness": sa / (vol_mm3 ** (2/3)) if vol_mm3 > 0 else 0.0,
        f"{prefix}n_voxels": float(n_voxels),
    }


def process_patient(patient, max_slices=300):
    pid = patient.patient_id
    if not patient.ct_series:
        return None
    series = patient.ct_series[0]
    dicom_dir = series['dicom_dir']

    # Use SimpleITK for volume loading to match mask orientation
    import SimpleITK as sitk
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom_dir)))
        sitk_img = reader.Execute()
        volume = sitk.GetArrayFromImage(sitk_img)  # (z, y, x)
        sp = sitk_img.GetSpacing()  # (x, y, z)
        spacing = (sp[2], sp[1], sp[0])
    except Exception as e:
        log.warning("  SimpleITK failed, falling back to pydicom: %s", e)
        volume = load_dicom_volume(dicom_dir, max_slices=0)
        if volume is None:
            return None
        spacing = get_pixel_spacing(dicom_dir) or (5.0, 0.7, 0.7)

    # Center-crop large volumes
    if volume.shape[0] > max_slices:
        n = volume.shape[0]
        start = (n - max_slices) // 2
        volume = volume[start:start + max_slices]
        log.info("  Center-cropped %d → %d slices", n, max_slices)

    mask_dir = MASKS_DIR / pid
    liver_done = (mask_dir / "liver.nii.gz").exists() and (mask_dir / "liver_lesions.nii.gz").exists()

    if not liver_done:
        with tempfile.TemporaryDirectory() as tmpdir:
            nifti_path = Path(tmpdir) / "input.nii.gz"
            nifti_img = dicom_to_nifti(dicom_dir)
            nib.save(nifti_img, str(nifti_path))
            mask_dir.mkdir(parents=True, exist_ok=True)

            log.info("  Running TotalSegmentator (total)...")
            try:
                run_totalsegmentator(nifti_path, mask_dir, task="total")
            except Exception as e:
                log.error("  TotalSegmentator total failed: %s", e)
                return None

            log.info("  Running TotalSegmentator (liver_lesions)...")
            try:
                # liver_lesions outputs to a separate dir, merge into mask_dir
                lesion_dir = Path(tmpdir) / "lesions"
                run_totalsegmentator(nifti_path, lesion_dir, task="liver_lesions")
                # Copy lesion mask
                lesion_file = lesion_dir / "liver_lesions.nii.gz"
                if lesion_file.exists():
                    import shutil
                    shutil.copy(str(lesion_file), str(mask_dir / "liver_lesions.nii.gz"))
            except Exception as e:
                log.warning("  liver_lesions failed: %s", e)

    # Load masks
    liver_mask = load_mask(mask_dir, "liver", volume.shape)
    lesion_mask = load_mask(mask_dir, "liver_lesions", volume.shape)

    if liver_mask is None:
        log.warning("  No liver mask for %s", pid)
        return None

    # IMPORTANT: restrict lesions to within the liver mask
    # TotalSegmentator liver_lesions detects lesions globally, not just in liver
    if lesion_mask is not None and liver_mask is not None:
        n_before = int(np.sum(lesion_mask))
        lesion_mask = lesion_mask & liver_mask
        n_after = int(np.sum(lesion_mask))
        if n_before > 0 and n_after < n_before:
            log.info("  Restricted lesions to liver: %d → %d voxels (removed %d outside liver)",
                     n_before, n_after, n_before - n_after)

    # Features
    features = {"patient_id": pid}
    features.update(compute_features(volume, liver_mask, "liver", spacing))
    features.update(compute_features(volume, lesion_mask, "lesion", spacing))

    # Derived
    liver_vol = features["liver_volume_ml"]
    lesion_vol = features["lesion_volume_ml"]
    features["lesion_liver_ratio"] = lesion_vol / liver_vol if liver_vol > 0 else 0.0
    features["has_lesion"] = int(features["lesion_n_voxels"] > 10)
    features["n_slices"] = volume.shape[0]
    features["spacing_z"] = spacing[0]

    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patients", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = TCGALIHCDataset(require_both=False, modality="CT")
    patients = [p for p in dataset.patients if p.has_imaging]
    if args.max_patients:
        patients = patients[:args.max_patients]

    # Resume
    existing = set()
    all_features = []
    if FEATURES_CSV.exists():
        import pandas as pd
        df = pd.read_csv(FEATURES_CSV)
        existing = set(df['patient_id'])
        all_features = df.to_dict('records')
        log.info("Already processed: %d", len(existing))

    fieldnames = None
    for i, patient in enumerate(patients):
        if patient.patient_id in existing:
            continue
        log.info("[%d/%d] %s (%d slices)", i + 1, len(patients),
                 patient.patient_id, patient.ct_series[0]['n_images'])
        try:
            feats = process_patient(patient)
            if feats:
                all_features.append(feats)
                if fieldnames is None:
                    fieldnames = list(feats.keys())
                if len(all_features) % 5 == 0:
                    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
                    with open(FEATURES_CSV, 'w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                        w.writeheader()
                        w.writerows(all_features)
        except Exception as e:
            log.error("  Failed: %s", e)

    if all_features and fieldnames:
        FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURES_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(all_features)
        log.info("Saved %d patients → %s", len(all_features), FEATURES_CSV)


if __name__ == "__main__":
    main()
