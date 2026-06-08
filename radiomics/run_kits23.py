#!/usr/bin/env python3
"""
Run KiTS23 nnU-Net v1 tumor segmentation on TCGA-KIRC CT volumes.

Segments 3 classes: kidney (1), tumor (2), cyst (3).
Saves predictions as NIfTI files alongside TotalSegmentator masks.

Usage:
    python radiomics/run_kits23.py [--max-patients N]
"""
import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataloader import TCGAKIRCDataset, get_pixel_spacing

log = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path("models/nnunet/results")
MASKS_DIR = config.RESULTS_DIR / "segmentation_masks"

# nnU-Net v1 environment variables
os.environ["RESULTS_FOLDER"] = str(MODEL_DIR)
os.environ["nnUNet_raw_data_base"] = str(MODEL_DIR.parent / "raw")
os.environ["nnUNet_preprocessed"] = str(MODEL_DIR.parent / "preprocessed")

# Fix PyTorch 2.6+ weights_only=True incompatibility with nnU-Net v1 checkpoints
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load


def load_volume_sitk(dicom_dir: Path):
    """Load volume using SimpleITK — matches the ordering used for segmentation."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_names)
    try:
        sitk_img = reader.Execute()
    except RuntimeError as e:
        if "Zero-valued spacing" not in str(e):
            raise
        log.warning("  Zero z-spacing detected, reconstructing with fallback spacing")
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(dicom_names[0])
        file_reader.ReadImageInformation()
        xy_spacing = [float(s) for s in file_reader.GetMetaData("0028|0030").split("\\")]
        z_positions = []
        for fn in dicom_names:
            file_reader.SetFileName(fn)
            file_reader.ReadImageInformation()
            pos = [float(x) for x in file_reader.GetMetaData("0020|0032").split("\\")]
            z_positions.append(pos[2])
        z_positions.sort()
        if len(z_positions) > 1:
            diffs = [z_positions[i+1] - z_positions[i] for i in range(len(z_positions)-1)]
            z_spacing = max(abs(d) for d in diffs if d != 0) if any(d != 0 for d in diffs) else 1.0
        else:
            z_spacing = 1.0
        slices = []
        for fn in dicom_names:
            s = sitk.ReadImage(fn)
            slices.append(sitk.GetArrayFromImage(s)[0])
        volume = np.stack(slices, axis=0)
        return volume, (z_spacing, xy_spacing[1], xy_spacing[0])
    volume = sitk.GetArrayFromImage(sitk_img)  # (z, y, x)
    spacing = sitk_img.GetSpacing()  # (x, y, z)
    return volume, (spacing[2], spacing[1], spacing[0])  # return as (z, y, x)


def dicom_to_nifti(dicom_dir: Path, volume: np.ndarray = None, spacing: tuple = None) -> nib.Nifti1Image:
    """Convert DICOM series to NIfTI using SimpleITK (handles all orientation edge cases)."""
    import SimpleITK as sitk
    import tempfile
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_names)
    try:
        sitk_img = reader.Execute()
    except RuntimeError as e:
        if "Zero-valued spacing" not in str(e):
            raise
        # Fallback: read slice-by-slice and compute spacing manually
        log.warning("  Zero z-spacing detected, reconstructing with fallback spacing")
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(dicom_names[0])
        file_reader.ReadImageInformation()
        xy_spacing = [float(s) for s in file_reader.GetMetaData("0028|0030").split("\\")]
        # Compute z-spacing from slice positions
        z_positions = []
        for fn in dicom_names:
            file_reader.SetFileName(fn)
            file_reader.ReadImageInformation()
            pos = [float(x) for x in file_reader.GetMetaData("0020|0032").split("\\")]
            z_positions.append(pos[2])
        z_positions.sort()
        if len(z_positions) > 1:
            diffs = [z_positions[i+1] - z_positions[i] for i in range(len(z_positions)-1)]
            z_spacing = max(abs(d) for d in diffs if d != 0) if any(d != 0 for d in diffs) else 1.0
        else:
            z_spacing = 1.0
        log.warning("  Using fallback z-spacing: %.4f", z_spacing)
        # Read slices individually and stack into a 3D volume
        slices = []
        for fn in dicom_names:
            s = sitk.ReadImage(fn)
            slices.append(sitk.GetArrayFromImage(s)[0])  # 2D array per slice
        vol_arr = np.stack(slices, axis=0)  # (Z, Y, X)
        sitk_img = sitk.GetImageFromArray(vol_arr)
        # Get orientation from first slice
        file_reader.SetFileName(dicom_names[0])
        file_reader.ReadImageInformation()
        origin = [float(x) for x in file_reader.GetMetaData("0020|0032").split("\\")]
        sitk_img.SetOrigin(origin)
        sitk_img.SetSpacing((xy_spacing[0], xy_spacing[1], z_spacing))
    tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    sitk.WriteImage(sitk_img, tmp.name)
    nifti = nib.load(tmp.name)
    # Force data into memory before deleting temp file (nibabel uses lazy loading)
    nifti = nib.Nifti1Image(np.asarray(nifti.dataobj), nifti.affine, nifti.header)
    Path(tmp.name).unlink()
    return nifti


def run_kits23_inference(input_nifti: Path, output_dir: Path):
    """Run nnU-Net v1 inference with KiTS23 weights."""
    from nnunet.inference.predict import predict_from_folder

    # nnU-Net v1 expects input in a folder with _0000.nii.gz naming
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare input
        in_dir = Path(tmpdir) / "input"
        in_dir.mkdir()
        out_dir = Path(tmpdir) / "output"
        out_dir.mkdir()

        # nnU-Net v1 naming: case_0000.nii.gz
        import shutil
        shutil.copy(str(input_nifti), str(in_dir / "case_0000.nii.gz"))

        model_folder = str(MODEL_DIR / "3d_fullres" / "Task779_Kidneys_KIRC" / "nnUNetTrainerV2__nnUNetPlansv2.1")

        predict_from_folder(
            model=model_folder,
            input_folder=str(in_dir),
            output_folder=str(out_dir),
            folds=(0, 1, 2, 3, 4),  # 5-fold ensemble
            save_npz=False,
            num_threads_preprocessing=4,
            num_threads_nifti_save=2,
            lowres_segmentations=None,
            part_id=0,
            num_parts=1,
            tta=False,  # no test-time augmentation for speed
            overwrite_existing=True,
        )

        # Copy result
        pred_file = out_dir / "case.nii.gz"
        if pred_file.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(pred_file), str(output_dir / "kits23_segmentation.nii.gz"))
            return True

    return False


def extract_tumor_features(volume: np.ndarray, seg_data: np.ndarray,
                            spacing: tuple) -> dict:
    """Extract features from KiTS23 segmentation (1=kidney, 2=tumor, 3=cyst)."""
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    features = {}

    for label, name in [(1, 'kits_kidney'), (2, 'kits_tumor'), (3, 'kits_cyst')]:
        mask = seg_data == label
        n_voxels = int(np.sum(mask))
        features[f'{name}_volume_ml'] = n_voxels * voxel_vol / 1000.0
        features[f'{name}_n_voxels'] = n_voxels

        if n_voxels > 10:
            vals = volume[mask].astype(float)
            features[f'{name}_hu_mean'] = float(np.mean(vals))
            features[f'{name}_hu_std'] = float(np.std(vals))
            features[f'{name}_hu_skew'] = float(__import__('scipy').stats.skew(vals))
            features[f'{name}_hu_p10'] = float(np.percentile(vals, 10))
            features[f'{name}_hu_p90'] = float(np.percentile(vals, 90))
            features[f'{name}_heterogeneity'] = float(np.std(vals) / (abs(np.mean(vals)) + 1e-10))
            features[f'{name}_necrotic_frac'] = float(np.mean(vals < 20))
            features[f'{name}_enhancing_frac'] = float(np.mean(vals > 150))

            # Surface ruggedness
            padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
            sa = 0.0
            for axis in range(3):
                diff = np.diff(padded, axis=axis)
                face = np.prod([spacing[i] for i in range(3) if i != axis])
                sa += np.sum(np.abs(diff)) * face
            vol_mm3 = n_voxels * voxel_vol
            features[f'{name}_ruggedness'] = sa / (vol_mm3 ** (2/3)) if vol_mm3 > 0 else 0.0
        else:
            for f in ['hu_mean', 'hu_std', 'hu_skew', 'hu_p10', 'hu_p90',
                       'heterogeneity', 'necrotic_frac', 'enhancing_frac', 'ruggedness']:
                features[f'{name}_{f}'] = 0.0

    # Derived
    kidney_vol = features['kits_kidney_volume_ml'] + features['kits_tumor_volume_ml']
    features['tumor_kidney_ratio'] = (features['kits_tumor_volume_ml'] / kidney_vol
                                       if kidney_vol > 0 else 0.0)
    features['cyst_kidney_ratio'] = (features['kits_cyst_volume_ml'] / kidney_vol
                                      if kidney_vol > 0 else 0.0)
    features['has_tumor'] = int(features['kits_tumor_n_voxels'] > 10)

    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--max-slices", type=int, default=300,
                        help="Skip volumes with more slices than this (OOM protection)")
    parser.add_argument("--output-csv", type=str,
                        default=str(config.RESULTS_DIR / "radiomics" / "kits23_features.csv"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dataset = TCGAKIRCDataset(require_both=True, load_sequences=False)
    patients = dataset.patients
    if args.max_patients:
        patients = patients[:args.max_patients]

    # Resume: check existing
    output_csv = Path(args.output_csv)
    existing = set()
    all_features = []
    if output_csv.exists():
        import pandas as pd
        df_existing = pd.read_csv(output_csv)
        existing = set(df_existing['patient_id'])
        all_features = df_existing.to_dict('records')
        log.info("Already processed: %d patients", len(existing))

    fieldnames = None
    for i, patient in enumerate(patients):
        pid = patient.patient_id
        if pid in existing:
            continue

        seg_file = MASKS_DIR / pid / "kits23_segmentation.nii.gz"

        log.info("[%d/%d] %s", i + 1, len(patients), pid)
        series = patient.ct_series[0]
        try:
            volume, spacing = load_volume_sitk(Path(series['dicom_dir']))
        except Exception as e:
            log.error("  Volume load failed for %s: %s", pid, e)
            continue
        if volume is None:
            continue
        full_nz = volume.shape[0]

        # Center-crop large volumes to avoid OOM during nnU-Net resampling
        if volume.shape[0] > args.max_slices:
            n = volume.shape[0]
            start = (n - args.max_slices) // 2
            volume = volume[start:start + args.max_slices]
            log.info("  Center-cropped %d → %d slices", n, args.max_slices)

        # Run segmentation if not cached
        if not seg_file.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                nifti_path = Path(tmpdir) / "input.nii.gz"
                try:
                    nifti_img = dicom_to_nifti(Path(series['dicom_dir']))
                except Exception as e:
                    log.error("  DICOM conversion failed for %s: %s", pid, e)
                    continue
                nib.save(nifti_img, str(nifti_path))

                log.info("  Running KiTS23 segmentation...")
                try:
                    success = run_kits23_inference(nifti_path, MASKS_DIR / pid)
                    if not success:
                        log.warning("  KiTS23 failed for %s", pid)
                        continue
                except Exception as e:
                    log.error("  KiTS23 error for %s: %s", pid, e)
                    continue

        # Load segmentation and extract features
        try:
            seg_img = nib.load(str(seg_file))
            seg_data = seg_img.get_fdata().transpose(2, 1, 0).astype(int)
            # Center-crop segmentation to match volume if it was cropped
            if seg_data.shape[0] != volume.shape[0] and seg_data.shape[0] == full_nz:
                start = (full_nz - args.max_slices) // 2
                seg_data = seg_data[start:start + args.max_slices]
                log.info("  Center-cropped segmentation %d → %d slices", full_nz, seg_data.shape[0])
            if seg_data.shape != volume.shape:
                log.warning("  Shape mismatch: seg=%s vol=%s", seg_data.shape, volume.shape)
                continue
        except Exception as e:
            log.error("  Failed to load segmentation: %s", e)
            continue

        feats = extract_tumor_features(volume, seg_data, spacing)
        feats['patient_id'] = pid
        all_features.append(feats)

        if fieldnames is None:
            fieldnames = list(feats.keys())

        # Checkpoint every 10
        if len(all_features) % 10 == 0:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            import csv
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_features)
            log.info("  Checkpoint: %d patients", len(all_features))

    # Final save
    if all_features and fieldnames:
        import csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_features)
        log.info("Saved %d patients to %s", len(all_features), output_csv)


if __name__ == "__main__":
    main()
