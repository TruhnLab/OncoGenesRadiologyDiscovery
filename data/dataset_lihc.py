#!/usr/bin/env python3
"""
TCGA-LIHC (Liver Hepatocellular Carcinoma) dataset loader.

Imaging: 97 patients, CT (75) + MR (40), from TCIA.
Genomics: Downloaded from GDC (MAF files + clinical).
Clinical: Downloaded from GDC.

Usage:
    from data.dataset_lihc import TCGALIHCDataset
    dataset = TCGALIHCDataset()
"""
import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ──
DATASET_ROOT = Path("/hpcwork/p0021834/datasets/TCGA-LIHC")
DICOM_ROOT = DATASET_ROOT / "download" / "doiJNLP-TCGA-LIHC-01-30-2017" / "TCGA-LIHC"
IMAGING_METADATA = DATASET_ROOT / "download" / "doiJNLP-TCGA-LIHC-01-30-2017" / "metadata.csv"

# Genomic/clinical data (to be downloaded via GDC)
DATA_DIR = Path(__file__).resolve().parent
GENOMIC_DIR = DATA_DIR / "genomic_lihc"
MAF_DIR = GENOMIC_DIR / "maf_files"
CLINICAL_CSV = GENOMIC_DIR / "clinical.csv"
REFERENCE_GENOME = DATA_DIR / "reference" / "hg38.fa"

# GDC project
GDC_PROJECT = "TCGA-LIHC"

# Liver-specific KIRC-equivalent driver genes
LIHC_DRIVER_GENES = {
    "TP53", "CTNNB1", "AXIN1", "ARID1A", "ARID2", "ALB", "APOB",
    "BAP1", "BRD7", "CDKN2A", "NFE2L2", "KEAP1", "RPS6KA3", "RB1",
    "PIK3CA", "TERT",
}


@dataclass
class Mutation:
    hugo_symbol: str
    chromosome: str
    start_position: int
    end_position: int
    variant_classification: str
    variant_type: str
    reference_allele: str
    tumor_allele: str


@dataclass
class PatientRecord:
    patient_id: str
    clinical: dict = field(default_factory=dict)
    ct_series: list = field(default_factory=list)
    mutations: list = field(default_factory=list)

    @property
    def has_imaging(self):
        return len(self.ct_series) > 0

    @property
    def has_genomic(self):
        return len(self.mutations) > 0

    @property
    def has_both(self):
        return self.has_imaging and self.has_genomic

    @property
    def n_mutations(self):
        return len(self.mutations)


def load_imaging_metadata(modality="CT"):
    """Load TCIA imaging metadata, filter to specified modality."""
    if not IMAGING_METADATA.exists():
        log.warning("Imaging metadata not found: %s", IMAGING_METADATA)
        return {}

    df = pd.read_csv(IMAGING_METADATA)
    df = df[df["Modality"] == modality]

    # Build per-patient series list
    patient_series = {}
    for _, row in df.iterrows():
        pid = row["Subject ID"]
        file_loc = row["File Location"]
        # Resolve path relative to dataset root
        if file_loc.startswith("./"):
            dicom_dir = DICOM_ROOT.parent / file_loc[2:]
        else:
            dicom_dir = Path(file_loc)

        series_info = {
            "series_uid": row["Series UID"],
            "study_uid": row["Study UID"],
            "dicom_dir": str(dicom_dir),
            "n_images": int(row["Number of Images"]),
            "series_description": str(row.get("Series Description", "")),
            "modality": row["Modality"],
        }

        if pid not in patient_series:
            patient_series[pid] = []
        patient_series[pid].append(series_info)

    # Sort each patient's series: prefer liver-specific protocols, then by n_images
    def series_priority(s):
        desc = s["series_description"].lower()
        # Prefer portal venous / arterial phase (best for liver segmentation)
        if any(kw in desc for kw in ["portal", "p.venous", "venous pha"]):
            return (0, -s["n_images"])
        if any(kw in desc for kw in ["arterial", "art. phase", "late art"]):
            return (1, -s["n_images"])
        if any(kw in desc for kw in ["liver", "hepatic", "abd"]):
            return (2, -s["n_images"])
        # Avoid scouts, localizers, reformats
        if any(kw in desc for kw in ["scout", "localizer", "topogram", "reformat", "cor ", "sag "]):
            return (9, -s["n_images"])
        return (5, -s["n_images"])

    for pid in patient_series:
        patient_series[pid].sort(key=series_priority)

    return patient_series


def load_clinical_data():
    """Load clinical CSV if available."""
    if not CLINICAL_CSV.exists():
        log.warning("Clinical data not found: %s (run download_genomic_lihc.py first)", CLINICAL_CSV)
        return {}
    df = pd.read_csv(CLINICAL_CSV)
    return {row["patient_id"]: row.to_dict() for _, row in df.iterrows()}


def load_mutations(patient_id):
    """Load mutations from MAF file for a patient."""
    maf_file = MAF_DIR / f"{patient_id}.maf.gz"
    if not maf_file.exists():
        return []

    mutations = []
    with gzip.open(maf_file, "rt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith("Hugo"):
                header = line.strip().split("\t")
                continue
            if header is None:
                continue
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            mutations.append(Mutation(
                hugo_symbol=row.get("Hugo_Symbol", ""),
                chromosome=row.get("Chromosome", ""),
                start_position=int(row.get("Start_Position", 0)),
                end_position=int(row.get("End_Position", 0)),
                variant_classification=row.get("Variant_Classification", ""),
                variant_type=row.get("Variant_Type", ""),
                reference_allele=row.get("Reference_Allele", ""),
                tumor_allele=row.get("Tumor_Seq_Allele2", ""),
            ))
    return mutations


def load_dicom_volume(dicom_dir, max_slices=0):
    """Load a 3D volume from a DICOM series directory."""
    import pydicom
    dicom_dir = Path(dicom_dir)
    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        return None

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return None

    # Sort by position
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))

    if max_slices > 0 and len(slices) > max_slices:
        start = (len(slices) - max_slices) // 2
        slices = slices[start:start + max_slices]

    volume = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope", 1))
        intercept = float(getattr(s, "RescaleIntercept", 0))
        arr = arr * slope + intercept
        volume.append(arr)

    return np.stack(volume, axis=0)


def get_pixel_spacing(dicom_dir):
    """Get voxel spacing (z, y, x) in mm from DICOM headers."""
    import pydicom
    dicom_dir = Path(dicom_dir)
    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        return None

    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    ps = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    st = float(getattr(ds, "SliceThickness", getattr(ds, "SpacingBetweenSlices", 5.0)))
    return (st, ps[0], ps[1])


class TCGALIHCDataset:
    """TCGA-LIHC dataset combining imaging, genomics, and clinical data."""

    def __init__(self, require_both=False, modality="CT", load_sequences=False):
        log.info("Loading TCGA-LIHC dataset...")

        self.imaging = load_imaging_metadata(modality=modality)
        self.clinical = load_clinical_data()

        # Build patient records
        all_pids = set(self.imaging.keys()) | set(self.clinical.keys())

        # Add patients with MAF files
        if MAF_DIR.exists():
            for f in MAF_DIR.glob("*.maf.gz"):
                all_pids.add(f.stem.replace(".maf", ""))

        self.patients = []
        for pid in sorted(all_pids):
            record = PatientRecord(
                patient_id=pid,
                clinical=self.clinical.get(pid, {}),
                ct_series=self.imaging.get(pid, []),
                mutations=load_mutations(pid),
            )
            if require_both and not record.has_both:
                continue
            self.patients.append(record)

        n_img = sum(1 for p in self.patients if p.has_imaging)
        n_gen = sum(1 for p in self.patients if p.has_genomic)
        n_both = sum(1 for p in self.patients if p.has_both)
        log.info("LIHC dataset: %d patients (%d imaging, %d genomic, %d both)",
                 len(self.patients), n_img, n_gen, n_both)

    def __len__(self):
        return len(self.patients)
