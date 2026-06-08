#!/usr/bin/env python3
"""
Unified dataloader that links DICOM imaging, somatic mutations (DNA sequences),
and clinical metadata per patient for the TCGA-KIRC cohort.

Provides:
  - TCGAKIRCDataset: PyTorch-compatible dataset
  - PatientRecord: dataclass holding all modalities for one patient
  - Helper functions for loading individual modalities
"""
import csv
import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    from pyfaidx import Fasta
except ImportError:
    Fasta = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """A single somatic mutation from a MAF file."""
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
    """All available data for a single TCGA-KIRC patient."""
    patient_id: str

    # Clinical
    clinical: dict = field(default_factory=dict)

    # Imaging: list of series, each series is a dict with metadata + path
    ct_series: list = field(default_factory=list)

    # Genomic: list of somatic mutations
    mutations: list = field(default_factory=list)

    # Derived: DNA sequences around mutations (populated by build_mutation_sequences)
    dna_sequences: list = field(default_factory=list)

    @property
    def has_imaging(self) -> bool:
        return len(self.ct_series) > 0

    @property
    def has_genomic(self) -> bool:
        return len(self.mutations) > 0

    @property
    def has_both(self) -> bool:
        return self.has_imaging and self.has_genomic

    @property
    def n_mutations(self) -> int:
        return len(self.mutations)


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_clinical_data() -> dict[str, dict]:
    """Load clinical CSV into a dict keyed by patient_id."""
    if not config.CLINICAL_CSV.exists():
        log.warning("Clinical CSV not found at %s", config.CLINICAL_CSV)
        return {}
    records = {}
    with open(config.CLINICAL_CSV) as f:
        for row in csv.DictReader(f):
            records[row["patient_id"]] = dict(row)
    return records


def load_imaging_metadata() -> dict[str, list[dict]]:
    """
    Load imaging metadata and return a dict mapping patient_id -> list of CT series info.
    Each entry has: series_uid, study_description, series_description, n_images, dicom_dir.
    We filter to CT modality only (for radiomics consistency).
    """
    if not config.IMAGING_METADATA.exists():
        log.warning("Imaging metadata not found at %s", config.IMAGING_METADATA)
        return {}

    patient_series = {}
    with open(config.IMAGING_METADATA) as f:
        for row in csv.DictReader(f):
            if row["Modality"] != "CT":
                continue
            pid = row["Subject ID"]
            # Resolve DICOM directory path
            rel_path = row["File Location"].lstrip("./")
            dicom_dir = config.IMAGING_DATASET / "download" / "TCIA_TCGA-KIRC_09-16-2015" / rel_path

            entry = {
                "series_uid": row["Series UID"],
                "study_description": row.get("Study Description", ""),
                "series_description": row.get("Series Description", ""),
                "n_images": int(row.get("Number of Images", 0)),
                "dicom_dir": str(dicom_dir),
            }
            patient_series.setdefault(pid, []).append(entry)

    # Sort series by number of images (descending) — larger series are more useful
    for pid in patient_series:
        patient_series[pid].sort(key=lambda s: s["n_images"], reverse=True)

    return patient_series


def load_mutations(patient_id: str) -> list[Mutation]:
    """Load somatic mutations from a patient's MAF file."""
    maf_path = config.MAF_DIR / f"{patient_id}.maf.gz"
    if not maf_path.exists():
        return []

    mutations = []
    text = gzip.open(maf_path).read().decode("utf-8")
    lines = [l for l in text.split("\n") if l and not l.startswith("#")]
    if len(lines) < 2:
        return []

    header = lines[0].split("\t")
    col_idx = {name: i for i, name in enumerate(header)}

    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < len(header):
            continue
        try:
            mutations.append(Mutation(
                hugo_symbol=cols[col_idx["Hugo_Symbol"]],
                chromosome=cols[col_idx["Chromosome"]],
                start_position=int(cols[col_idx["Start_Position"]]),
                end_position=int(cols[col_idx["End_Position"]]),
                variant_classification=cols[col_idx["Variant_Classification"]],
                variant_type=cols[col_idx["Variant_Type"]],
                reference_allele=cols[col_idx["Reference_Allele"]],
                tumor_allele=cols[col_idx["Tumor_Seq_Allele2"]],
            ))
        except (ValueError, KeyError):
            continue

    return mutations


def load_dicom_volume(dicom_dir: str, max_slices: int = 0) -> Optional[np.ndarray]:
    """
    Load a DICOM series from a directory into a 3D numpy array (slices, H, W).
    Values are in Hounsfield Units (HU).
    Returns None if loading fails.
    """
    if pydicom is None:
        raise ImportError("pydicom is required: pip install pydicom")

    dicom_path = Path(dicom_dir)
    if not dicom_path.exists():
        log.warning("DICOM directory not found: %s", dicom_dir)
        return None

    # Read all DICOM files
    dcm_files = sorted(dicom_path.glob("*.dcm"))
    if not dcm_files:
        # Try without extension
        dcm_files = sorted(f for f in dicom_path.iterdir() if f.is_file())
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

    # Sort by ImagePositionPatient (z-axis) or InstanceNumber
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except (AttributeError, ValueError):
            pass

    if max_slices and len(slices) > max_slices:
        # Take center slices
        start = (len(slices) - max_slices) // 2
        slices = slices[start : start + max_slices]

    # Convert to HU
    volume = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        intercept = getattr(s, "RescaleIntercept", 0)
        slope = getattr(s, "RescaleSlope", 1)
        arr = arr * slope + intercept
        volume.append(arr)

    return np.stack(volume, axis=0)


def get_pixel_spacing(dicom_dir: str) -> Optional[tuple[float, float, float]]:
    """Get (slice_spacing, pixel_row, pixel_col) in mm from first DICOM file."""
    if pydicom is None:
        return None
    dicom_path = Path(dicom_dir)
    dcm_files = sorted(dicom_path.glob("*.dcm"))
    if not dcm_files:
        return None
    try:
        ds = pydicom.dcmread(str(dcm_files[0]))
        px = ds.PixelSpacing
        st = getattr(ds, "SliceThickness", getattr(ds, "SpacingBetweenSlices", px[0]))
        return (float(st), float(px[0]), float(px[1]))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DNA sequence construction from mutations + reference genome
# ---------------------------------------------------------------------------

def build_mutation_sequences(
    mutations: list[Mutation],
    context_bp: int = config.MUTATION_CONTEXT_BP,
    reference_path: Path = config.REFERENCE_GENOME,
) -> list[dict]:
    """
    For each mutation, extract reference DNA context and apply the mutation.
    Returns list of dicts with: gene, chrom, position, ref_seq, mut_seq, variant_class.

    Each sequence is (2 * context_bp) nucleotides centered on the mutation,
    with the tumor allele substituted in mut_seq.
    """
    if Fasta is None:
        raise ImportError("pyfaidx is required: pip install pyfaidx")

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference genome not found at {reference_path}. "
            "Run: python data/download_genomic.py"
        )

    ref = Fasta(str(reference_path))

    sequences = []
    for mut in mutations:
        chrom = mut.chromosome
        # pyfaidx uses 0-based indexing, MAF uses 1-based
        center = mut.start_position - 1
        start = max(0, center - context_bp)
        end = center + context_bp

        # Get chromosome name (MAF uses 'chr1', hg38 uses 'chr1')
        chrom_key = chrom if chrom in ref else f"chr{chrom}"
        if chrom_key not in ref:
            continue

        chrom_len = len(ref[chrom_key])
        end = min(end, chrom_len)
        if end - start < context_bp:
            continue

        ref_seq = str(ref[chrom_key][start:end]).upper()

        # Apply mutation: substitute tumor allele at the mutation position
        mut_offset = center - start
        ref_allele = mut.reference_allele
        tumor_allele = mut.tumor_allele

        if tumor_allele == "-" or ref_allele == "-":
            # Indel: just use reference sequence (Evo2 will see the structural difference)
            mut_seq = ref_seq
        else:
            # SNV or MNV: substitute
            mut_seq = ref_seq[:mut_offset] + tumor_allele + ref_seq[mut_offset + len(ref_allele):]

        # Validate: only keep sequences with valid nucleotides
        valid_chars = set("ACGTN")
        if not all(c in valid_chars for c in ref_seq):
            continue

        sequences.append({
            "gene": mut.hugo_symbol,
            "chrom": chrom,
            "position": mut.start_position,
            "ref_seq": ref_seq,
            "mut_seq": mut_seq,
            "variant_class": mut.variant_classification,
        })

    return sequences


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class TCGAKIRCDataset:
    """
    Dataset that loads and links all modalities for TCGA-KIRC patients.

    Usage:
        dataset = TCGAKIRCDataset(require_both=True)
        for patient in dataset:
            print(patient.patient_id, patient.n_mutations, len(patient.ct_series))
    """

    def __init__(
        self,
        require_both: bool = True,
        load_sequences: bool = False,
        ct_modality_only: bool = True,
    ):
        """
        Args:
            require_both: If True, only include patients with BOTH imaging and genomic data.
            load_sequences: If True, also construct DNA sequences from mutations + reference.
            ct_modality_only: If True, only include CT series (not MR).
        """
        log.info("Loading TCGA-KIRC dataset...")

        clinical_data = load_clinical_data()
        imaging_data = load_imaging_metadata()

        # Get all patient IDs from imaging metadata
        all_patient_ids = sorted(set(clinical_data.keys()) | set(imaging_data.keys()))

        self.patients: list[PatientRecord] = []
        for pid in all_patient_ids:
            record = PatientRecord(
                patient_id=pid,
                clinical=clinical_data.get(pid, {}),
                ct_series=imaging_data.get(pid, []),
                mutations=load_mutations(pid),
            )

            if load_sequences and record.has_genomic:
                try:
                    record.dna_sequences = build_mutation_sequences(record.mutations)
                except (ImportError, FileNotFoundError) as e:
                    log.warning("Cannot build sequences for %s: %s", pid, e)

            if require_both and not record.has_both:
                continue

            self.patients.append(record)

        n_img = sum(1 for p in self.patients if p.has_imaging)
        n_gen = sum(1 for p in self.patients if p.has_genomic)
        n_both = sum(1 for p in self.patients if p.has_both)
        log.info(
            "Dataset loaded: %d patients total (%d with imaging, %d with genomic, %d with both)",
            len(self.patients), n_img, n_gen, n_both,
        )

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> PatientRecord:
        return self.patients[idx]

    def __iter__(self):
        return iter(self.patients)

    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        for p in self.patients:
            if p.patient_id == patient_id:
                return p
        return None

    def patient_ids(self) -> list[str]:
        return [p.patient_id for p in self.patients]

    def summary(self) -> dict:
        """Return summary statistics about the dataset."""
        mutations_per_patient = [p.n_mutations for p in self.patients if p.has_genomic]
        series_per_patient = [len(p.ct_series) for p in self.patients if p.has_imaging]
        return {
            "n_patients": len(self.patients),
            "n_with_imaging": sum(1 for p in self.patients if p.has_imaging),
            "n_with_genomic": sum(1 for p in self.patients if p.has_genomic),
            "n_with_both": sum(1 for p in self.patients if p.has_both),
            "mutations_per_patient_mean": np.mean(mutations_per_patient) if mutations_per_patient else 0,
            "mutations_per_patient_median": np.median(mutations_per_patient) if mutations_per_patient else 0,
            "ct_series_per_patient_mean": np.mean(series_per_patient) if series_per_patient else 0,
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import json

    dataset = TCGAKIRCDataset(require_both=True, load_sequences=False)
    print(json.dumps(dataset.summary(), indent=2))

    # Show a sample patient
    if dataset.patients:
        p = dataset.patients[0]
        print(f"\nSample patient: {p.patient_id}")
        print(f"  Clinical: {p.clinical}")
        print(f"  CT series: {len(p.ct_series)}, top series: {p.ct_series[0]['n_images']} slices")
        print(f"  Mutations: {p.n_mutations}")
        if p.mutations:
            m = p.mutations[0]
            print(f"    First mutation: {m.hugo_symbol} {m.chromosome}:{m.start_position} "
                  f"{m.reference_allele}>{m.tumor_allele} ({m.variant_classification})")
