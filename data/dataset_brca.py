#!/usr/bin/env python3
"""
TCGA-BRCA (Breast Invasive Carcinoma) dataset loader.

Imaging: Mammography (MG) + MR from TCIA (needs download from manifest).
Genomics: Downloaded from GDC (MAF files + clinical).
Clinical: Downloaded from GDC.

Usage:
    from data.dataset_brca import TCGABRCADataset
    dataset = TCGABRCADataset()
"""
import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ──
DATASET_ROOT = Path("/hpcwork/p0021834/datasets/TCGA-BRCA")
# Imaging (downloaded via tcia_utils — flat series UID directories)
DOWNLOAD_DIR = DATASET_ROOT / "download"
IMAGING_METADATA = DATASET_ROOT / "metadata.csv"  # saved by tcia_utils getSeries()

# Genomic/clinical data (to be downloaded via GDC)
DATA_DIR = Path(__file__).resolve().parent
GENOMIC_DIR = DATA_DIR / "genomic_brca"
MAF_DIR = GENOMIC_DIR / "maf_files"
CLINICAL_CSV = GENOMIC_DIR / "clinical.csv"

# GDC project
GDC_PROJECT = "TCGA-BRCA"

# Breast cancer driver genes
BRCA_DRIVER_GENES = {
    "TP53", "PIK3CA", "CDH1", "GATA3", "MAP3K1", "MLL3", "PTEN",
    "AKT1", "CBFB", "MAP2K4", "RUNX1", "TBX3", "NCOR1", "CTCF",
    "SF3B1", "CDKN1B", "RB1", "ERBB2", "BRCA1", "BRCA2",
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
    imaging_series: list = field(default_factory=list)
    mutations: list = field(default_factory=list)

    @property
    def has_imaging(self):
        return len(self.imaging_series) > 0

    @property
    def has_genomic(self):
        return len(self.mutations) > 0

    @property
    def has_both(self):
        return self.has_imaging and self.has_genomic

    @property
    def n_mutations(self):
        return len(self.mutations)


def load_imaging_metadata(modality=None):
    """Load TCIA imaging metadata (tcia_utils format with SeriesInstanceUID dirs)."""
    if not IMAGING_METADATA.exists():
        log.warning("Imaging metadata not found: %s", IMAGING_METADATA)
        return {}

    df = pd.read_csv(IMAGING_METADATA)
    if modality:
        df = df[df["Modality"] == modality]

    patient_series = {}
    for _, row in df.iterrows():
        pid = row["PatientID"]
        series_uid = row["SeriesInstanceUID"]
        # tcia_utils downloads into flat dirs named by SeriesInstanceUID
        dicom_dir = DOWNLOAD_DIR / series_uid

        if not dicom_dir.exists():
            continue

        series_info = {
            "series_uid": series_uid,
            "study_uid": row.get("StudyInstanceUID", ""),
            "dicom_dir": str(dicom_dir),
            "n_images": int(row.get("ImageCount", 0)),
            "series_description": str(row.get("SeriesDescription", "")),
            "modality": row["Modality"],
        }
        if pid not in patient_series:
            patient_series[pid] = []
        patient_series[pid].append(series_info)

    for pid in patient_series:
        patient_series[pid].sort(key=lambda s: s["n_images"], reverse=True)

    return patient_series


def load_clinical_data():
    """Load clinical CSV if available."""
    if not CLINICAL_CSV.exists():
        return {}
    df = pd.read_csv(CLINICAL_CSV)
    return {row["patient_id"]: row.to_dict() for _, row in df.iterrows()}


def load_mutations(patient_id):
    """Load mutations from MAF file."""
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


class TCGABRCADataset:
    """TCGA-BRCA dataset combining imaging, genomics, and clinical data."""

    def __init__(self, require_both=False, modality=None, load_sequences=False):
        log.info("Loading TCGA-BRCA dataset...")

        self.imaging = load_imaging_metadata(modality=modality)
        self.clinical = load_clinical_data()

        all_pids = set(self.imaging.keys()) | set(self.clinical.keys())
        if MAF_DIR.exists():
            for f in MAF_DIR.glob("*.maf.gz"):
                all_pids.add(f.stem.replace(".maf", ""))

        self.patients = []
        for pid in sorted(all_pids):
            record = PatientRecord(
                patient_id=pid,
                clinical=self.clinical.get(pid, {}),
                imaging_series=self.imaging.get(pid, []),
                mutations=load_mutations(pid),
            )
            if require_both and not record.has_both:
                continue
            self.patients.append(record)

        n_img = sum(1 for p in self.patients if p.has_imaging)
        n_gen = sum(1 for p in self.patients if p.has_genomic)
        n_both = sum(1 for p in self.patients if p.has_both)
        log.info("BRCA dataset: %d patients (%d imaging, %d genomic, %d both)",
                 len(self.patients), n_img, n_gen, n_both)

    def __len__(self):
        return len(self.patients)
