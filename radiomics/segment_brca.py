#!/usr/bin/env python3
"""
Segment breast tumors using MAMA-MIA nnUNet v2 model.

For each patient: load post-contrast MRI → run MAMA-MIA → extract tumor features.

Usage:
    python radiomics/segment_brca.py [--max-patients N]
"""
import csv
import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import torch
from scipy import stats

# Patch torch.load before any nnUNet imports
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = config.RESULTS_DIR / "brca"
MASKS_DIR = RESULTS_DIR / "segmentation_masks"
FEATURES_CSV = RESULTS_DIR / "tumor_features.csv"

os.environ['nnUNet_results'] = str(PROJECT / 'models' / 'mama_mia' / 'nnUNet_results')

import pandas as pd
BRCA_METADATA = Path("/hpcwork/p0021834/datasets/TCGA-BRCA/metadata.csv")
BRCA_DOWNLOAD = Path("/hpcwork/p0021834/datasets/TCGA-BRCA/download")


def find_post_contrast_series(pid, metadata_df):
    """Find the best post-contrast MRI series for a patient."""
    patient = metadata_df[metadata_df['PatientID'] == pid]
    mr = patient[patient['Modality'] == 'MR']
    if mr.empty:
        return None

    # Priority: POST-CONTRAST > VIBRANT > largest MR series
    for pattern in ['POST', 'VIBRANT', 'post', 'vibrant']:
        match = mr[mr['SeriesDescription'].str.contains(pattern, case=False, na=False)]
        if not match.empty:
            return match.sort_values('ImageCount', ascending=False).iloc[0]

    # Fallback: largest MR series (likely DCE)
    return mr.sort_values('ImageCount', ascending=False).iloc[0]


def load_mri_volume(series_uid):
    """Load MRI volume from DICOM series."""
    dicom_dir = BRCA_DOWNLOAD / series_uid
    if not dicom_dir.exists():
        return None, None, None

    dcm_files = sorted(dicom_dir.glob('*.dcm'))
    if not dcm_files:
        return None, None, None

    slices = []
    for f in dcm_files:
        try:
            slices.append(pydicom.dcmread(str(f)))
        except Exception:
            continue
    if not slices:
        return None, None, None

    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        slices.sort(key=lambda s: int(getattr(s, 'InstanceNumber', 0)))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    ds = slices[0]
    ps = [float(x) for x in getattr(ds, 'PixelSpacing', [1.0, 1.0])]
    st = float(getattr(ds, 'SliceThickness', getattr(ds, 'SpacingBetweenSlices', 3.0)))
    spacing = np.array([ps[1], ps[0], st])  # x, y, z for NIfTI

    # Build affine
    iop = [float(x) for x in ds.ImageOrientationPatient]
    ipp = [float(x) for x in slices[0].ImagePositionPatient]
    row_cos, col_cos = np.array(iop[:3]), np.array(iop[3:])
    slice_cos = np.cross(row_cos, col_cos)
    affine = np.eye(4)
    affine[:3, 0] = row_cos * ps[1]
    affine[:3, 1] = col_cos * ps[0]
    affine[:3, 2] = slice_cos * st
    affine[:3, 3] = ipp

    return volume, spacing, affine


def extract_tumor_features(volume, prediction, spacing_xyz):
    """Extract features from tumor segmentation."""
    voxel_vol = spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]
    tumor_mask = prediction > 0
    n_tumor = int(np.sum(tumor_mask))

    features = {
        'tumor_volume_ml': n_tumor * voxel_vol / 1000,
        'tumor_n_voxels': n_tumor,
        'has_tumor': int(n_tumor > 10),
    }

    if n_tumor > 10:
        vals = volume.transpose(2, 1, 0)[tumor_mask].astype(float)
        features['tumor_intensity_mean'] = float(np.mean(vals))
        features['tumor_intensity_std'] = float(np.std(vals))
        features['tumor_intensity_skew'] = float(stats.skew(vals))
        features['tumor_heterogeneity'] = float(np.std(vals) / (abs(np.mean(vals)) + 1e-10))

        # Ruggedness
        padded = np.pad(tumor_mask.astype(np.uint8), 1, mode="constant")
        sa = sum(np.sum(np.abs(np.diff(padded, axis=ax))) *
                 np.prod([spacing_xyz[i] for i in range(3) if i != ax]) for ax in range(3))
        vol_mm3 = n_tumor * voxel_vol
        features['tumor_ruggedness'] = sa / (vol_mm3 ** (2/3)) if vol_mm3 > 0 else 0.0
    else:
        for k in ['tumor_intensity_mean', 'tumor_intensity_std', 'tumor_intensity_skew',
                   'tumor_heterogeneity', 'tumor_ruggedness']:
            features[k] = 0.0

    # Total breast volume (rough: all non-zero voxels)
    breast_voxels = np.sum(volume.transpose(2, 1, 0) > 0)
    features['tumor_breast_ratio'] = n_tumor / breast_voxels if breast_voxels > 0 else 0.0

    return features


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patients", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(BRCA_METADATA)

    # All MR patients
    mr_patients = sorted(metadata[metadata['Modality'] == 'MR']['PatientID'].unique())
    if args.max_patients:
        mr_patients = mr_patients[:args.max_patients]
    log.info("Processing %d breast MRI patients", len(mr_patients))

    # Resume
    existing = set()
    all_features = []
    if FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV)
        existing = set(df['patient_id'])
        all_features = df.to_dict('records')
        log.info("Already processed: %d", len(existing))

    # Load model once
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        device=torch.device('cuda', 0), verbose=False, verbose_preprocessing=False,
    )
    model_dir = str(PROJECT / 'models' / 'mama_mia' / 'nnUNet_results' /
                    'Dataset101_MAMAMMIA' / 'nnUNetTrainer__nnUNetPlans__3d_fullres')
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0, 1, 2, 3, 4))
    log.info("MAMA-MIA model loaded")

    fieldnames = None
    for i, pid in enumerate(mr_patients):
        if pid in existing:
            continue

        log.info("[%d/%d] %s", i + 1, len(mr_patients), pid)
        series_info = find_post_contrast_series(pid, metadata)
        if series_info is None:
            log.warning("  No MR series found")
            continue

        series_uid = series_info['SeriesInstanceUID']
        log.info("  Series: %s (%d images)", series_info['SeriesDescription'], series_info['ImageCount'])

        volume, spacing, affine = load_mri_volume(series_uid)
        if volume is None:
            log.warning("  Failed to load volume")
            continue

        log.info("  Volume: %s", volume.shape)

        # Center-crop large volumes
        if volume.shape[0] > 300:
            n = volume.shape[0]
            start = (n - 300) // 2
            volume = volume[start:start + 300]
            log.info("  Center-cropped %d → 300 slices", n)

        try:
            data = volume.transpose(2, 1, 0)[np.newaxis].astype(np.float32)  # (1, x, y, z)
            prediction = predictor.predict_single_npy_array(data, {'spacing': spacing}, None, None, False)

            # Save mask
            mask_dir = MASKS_DIR / pid
            mask_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine),
                     str(mask_dir / 'mama_mia_tumor.nii.gz'))

            # Extract features
            feats = extract_tumor_features(volume, prediction, spacing)
            feats['patient_id'] = pid
            feats['series_description'] = series_info['SeriesDescription']
            all_features.append(feats)

            if fieldnames is None:
                fieldnames = list(feats.keys())

            n_tumor = np.sum(prediction > 0)
            log.info("  Tumor: %d voxels (%.1f mL)", n_tumor, feats['tumor_volume_ml'])

            # Checkpoint
            if len(all_features) % 10 == 0 and fieldnames:
                with open(FEATURES_CSV, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    w.writeheader()
                    w.writerows(all_features)
                log.info("  Checkpoint: %d patients", len(all_features))

        except Exception as e:
            log.error("  Failed: %s", e)

    # Final save
    if all_features and fieldnames:
        with open(FEATURES_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(all_features)
        log.info("Saved %d patients → %s", len(all_features), FEATURES_CSV)


if __name__ == '__main__':
    main()
