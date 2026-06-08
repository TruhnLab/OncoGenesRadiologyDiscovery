#!/usr/bin/env python3
"""
Download genomic (MAF) and clinical data from GDC for TCGA-KIRC patients
that have matching imaging data in the local dataset.

Usage:
    python data/download_genomic.py [--skip-maf] [--skip-clinical] [--skip-reference]

Downloads:
  1. Masked Somatic Mutation MAF files (open access) from GDC
  2. Clinical metadata (demographics, diagnosis, staging) via GDC API
  3. hg38 reference genome for mutation context extraction
"""
import argparse
import csv
import gzip
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gdc_post(endpoint: str, payload: dict, timeout: int = 60) -> dict:
    """POST JSON to a GDC API endpoint and return parsed response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 2:
                log.warning("GDC request failed (%s), retrying in 5s...", e)
                time.sleep(5)
            else:
                raise


def get_imaging_patient_ids() -> list[str]:
    """Read patient IDs from the imaging metadata CSV."""
    patients = set()
    with open(config.IMAGING_METADATA) as f:
        reader = csv.DictReader(f)
        for row in reader:
            patients.add(row["Subject ID"])
    return sorted(patients)


# ---------------------------------------------------------------------------
# 1. Download MAF files
# ---------------------------------------------------------------------------

def fetch_maf_file_manifest(patient_ids: list[str]) -> list[dict]:
    """Query GDC for open-access Masked Somatic Mutation MAF files for our patients."""
    log.info("Querying GDC for MAF files matching %d imaging patients...", len(patient_ids))

    all_hits = []
    # GDC API has a limit on filter size, so batch patient IDs
    batch_size = 50
    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i : i + batch_size]
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": config.TCGA_PROJECT,
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.submitter_id",
                        "value": batch,
                    },
                },
                {
                    "op": "=",
                    "content": {
                        "field": "data_type",
                        "value": "Masked Somatic Mutation",
                    },
                },
                {
                    "op": "=",
                    "content": {"field": "data_format", "value": "maf"},
                },
                {
                    "op": "=",
                    "content": {"field": "access", "value": "open"},
                },
            ],
        }
        payload = {
            "filters": filters,
            "fields": "file_id,file_name,file_size,cases.submitter_id",
            "size": 1000,
        }
        resp = gdc_post(config.GDC_FILES_ENDPOINT, payload)
        hits = resp["data"]["hits"]
        all_hits.extend(hits)
        log.info(
            "  Batch %d-%d: found %d MAF files",
            i, min(i + batch_size, len(patient_ids)), len(hits),
        )

    # Deduplicate by file_id (a patient may appear in multiple batches if near boundary)
    seen = set()
    unique = []
    for h in all_hits:
        if h["file_id"] not in seen:
            seen.add(h["file_id"])
            unique.append(h)

    # Build patient -> file mapping
    manifest = []
    for h in unique:
        patient_id = h["cases"][0]["submitter_id"]
        manifest.append({
            "patient_id": patient_id,
            "file_id": h["file_id"],
            "file_name": h["file_name"],
            "file_size": h["file_size"],
        })

    log.info("Total unique MAF files found: %d", len(manifest))
    return manifest


def download_single_maf(entry: dict, out_dir: Path) -> str:
    """Download a single MAF file from GDC. Returns the patient_id on success."""
    patient_id = entry["patient_id"]
    file_id = entry["file_id"]
    out_path = out_dir / f"{patient_id}.maf.gz"

    if out_path.exists() and out_path.stat().st_size > 0:
        return patient_id  # already downloaded

    url = f"{config.GDC_DATA_ENDPOINT}/{file_id}"
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(url, timeout=120)
            data = resp.read()
            out_path.write_bytes(data)
            return patient_id
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                log.error("Failed to download MAF for %s: %s", patient_id, e)
                return None


def download_maf_files(patient_ids: list[str]):
    """Download all MAF files for the given patients."""
    config.MAF_DIR.mkdir(parents=True, exist_ok=True)

    manifest = fetch_maf_file_manifest(patient_ids)

    # Save manifest for reference
    config.GENOMIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PATIENT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    # Check which patients have MAF data
    patients_with_maf = {e["patient_id"] for e in manifest}
    patients_without_maf = set(patient_ids) - patients_with_maf
    if patients_without_maf:
        log.warning(
            "%d imaging patients have NO MAF data on GDC: %s",
            len(patients_without_maf),
            sorted(patients_without_maf)[:10],
        )

    # Download in parallel
    log.info("Downloading %d MAF files...", len(manifest))
    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(download_single_maf, entry, config.MAF_DIR): entry
            for entry in manifest
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                downloaded += 1
            else:
                failed += 1
            if (downloaded + failed) % 50 == 0:
                log.info("  Progress: %d/%d downloaded", downloaded, len(manifest))

    log.info("MAF download complete: %d succeeded, %d failed", downloaded, failed)


# ---------------------------------------------------------------------------
# 2. Download clinical data
# ---------------------------------------------------------------------------

CLINICAL_FIELDS = [
    "submitter_id",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.tumor_grade",
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.age_at_diagnosis",
    "diagnoses.tissue_or_organ_of_origin",
]


def download_clinical_data(patient_ids: list[str]):
    """Download clinical metadata for all patients via GDC cases API."""
    config.GENOMIC_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading clinical data for %d patients...", len(patient_ids))

    all_cases = []
    batch_size = 100
    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i : i + batch_size]
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "project.project_id",
                        "value": config.TCGA_PROJECT,
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "submitter_id",
                        "value": batch,
                    },
                },
            ],
        }
        payload = {
            "filters": filters,
            "fields": ",".join(CLINICAL_FIELDS),
            "size": 200,
        }
        resp = gdc_post(config.GDC_CASES_ENDPOINT, payload)
        all_cases.extend(resp["data"]["hits"])
        log.info("  Batch %d-%d: %d cases", i, i + len(batch), len(resp["data"]["hits"]))

    # Flatten nested clinical data into rows
    rows = []
    for case in all_cases:
        patient_id = case.get("submitter_id", "")
        demo = case.get("demographic", {}) or {}

        # Pick the KIRC diagnosis (morphology 8310/3 = clear cell adenocarcinoma)
        diagnoses = case.get("diagnoses", []) or []
        dx = {}
        for d in diagnoses:
            if d.get("morphology") == "8310/3" or d.get("ajcc_pathologic_stage"):
                dx = d
                break
        if not dx and diagnoses:
            dx = diagnoses[0]

        rows.append({
            "patient_id": patient_id,
            "gender": demo.get("gender", ""),
            "race": demo.get("race", ""),
            "ethnicity": demo.get("ethnicity", ""),
            "vital_status": demo.get("vital_status", ""),
            "days_to_death": demo.get("days_to_death", ""),
            "age_at_index": demo.get("age_at_index", ""),
            "ajcc_stage": dx.get("ajcc_pathologic_stage", ""),
            "ajcc_t": dx.get("ajcc_pathologic_t", ""),
            "ajcc_n": dx.get("ajcc_pathologic_n", ""),
            "ajcc_m": dx.get("ajcc_pathologic_m", ""),
            "tumor_grade": dx.get("tumor_grade", ""),
            "primary_diagnosis": dx.get("primary_diagnosis", ""),
            "morphology": dx.get("morphology", ""),
            "days_to_last_follow_up": dx.get("days_to_last_follow_up", ""),
            "age_at_diagnosis": dx.get("age_at_diagnosis", ""),
            "tissue_or_organ": dx.get("tissue_or_organ_of_origin", ""),
        })

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(config.CLINICAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Clinical data saved to %s (%d patients)", config.CLINICAL_CSV, len(rows))


# ---------------------------------------------------------------------------
# 3. Download reference genome (for mutation context extraction)
# ---------------------------------------------------------------------------

REFERENCE_URLS = {
    # UCSC hg38 - primary assembly chromosomes
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
}


def download_reference_genome():
    """Download hg38 reference genome for extracting mutation context sequences."""
    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    if config.REFERENCE_GENOME.exists():
        log.info("Reference genome already exists at %s", config.REFERENCE_GENOME)
        return

    gz_path = config.REFERENCE_DIR / "hg38.fa.gz"
    if not gz_path.exists():
        url = REFERENCE_URLS["hg38"]
        log.info("Downloading hg38 reference genome (~900MB compressed)...")
        log.info("  URL: %s", url)
        log.info("  This may take 10-30 minutes depending on network speed.")

        # Use urllib with progress
        resp = urllib.request.urlopen(url, timeout=600)
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(gz_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * chunk_size) == 0:
                    pct = downloaded / total * 100
                    log.info("  Downloaded %.0f%% (%d MB / %d MB)", pct, downloaded // 1e6, total // 1e6)

        log.info("  Download complete: %d MB", gz_path.stat().st_size // 1e6)

    # Decompress
    log.info("Decompressing reference genome (this takes a few minutes)...")
    import shutil
    with gzip.open(gz_path, "rb") as f_in, open(config.REFERENCE_GENOME, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
    log.info("Reference genome ready at %s", config.REFERENCE_GENOME)

    # Index with samtools if available, otherwise with pyfaidx
    _index_reference()


def _index_reference():
    """Create .fai index for the reference genome."""
    fai_path = Path(str(config.REFERENCE_GENOME) + ".fai")
    if fai_path.exists():
        log.info("Reference index already exists.")
        return

    log.info("Indexing reference genome...")
    # Try samtools first
    ret = os.system(f"samtools faidx {config.REFERENCE_GENOME} 2>/dev/null")
    if ret == 0 and fai_path.exists():
        log.info("Indexed with samtools.")
        return

    # Fallback to pyfaidx
    try:
        from pyfaidx import Faidx
        Faidx(str(config.REFERENCE_GENOME))
        log.info("Indexed with pyfaidx.")
    except ImportError:
        log.warning(
            "Neither samtools nor pyfaidx available. "
            "Install pyfaidx (`pip install pyfaidx`) to index the reference genome. "
            "The pipeline will attempt to index on first use."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-maf", action="store_true", help="Skip MAF download")
    parser.add_argument("--skip-clinical", action="store_true", help="Skip clinical data")
    parser.add_argument("--skip-reference", action="store_true", help="Skip reference genome")
    args = parser.parse_args()

    patient_ids = get_imaging_patient_ids()
    log.info("Found %d patients with imaging data.", len(patient_ids))

    if not args.skip_maf:
        download_maf_files(patient_ids)

    if not args.skip_clinical:
        download_clinical_data(patient_ids)

    if not args.skip_reference:
        download_reference_genome()

    # Summary
    log.info("=" * 60)
    log.info("Download summary:")
    if config.MAF_DIR.exists():
        n_maf = len(list(config.MAF_DIR.glob("*.maf.gz")))
        log.info("  MAF files: %d", n_maf)
    if config.CLINICAL_CSV.exists():
        with open(config.CLINICAL_CSV) as f:
            n_clin = sum(1 for _ in f) - 1
        log.info("  Clinical records: %d", n_clin)
    if config.REFERENCE_GENOME.exists():
        log.info("  Reference genome: %s (%.1f GB)", config.REFERENCE_GENOME,
                 config.REFERENCE_GENOME.stat().st_size / 1e9)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
