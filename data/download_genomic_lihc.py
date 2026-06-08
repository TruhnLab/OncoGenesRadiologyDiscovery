#!/usr/bin/env python3
"""
Download genomic (MAF) and clinical data from GDC for TCGA-LIHC patients.

Usage:
    python data/download_genomic_lihc.py
"""
import csv
import gzip
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"
GDC_CASES = "https://api.gdc.cancer.gov/cases"
PROJECT = "TCGA-LIHC"

DATA_DIR = Path(__file__).resolve().parent
GENOMIC_DIR = DATA_DIR / "genomic_lihc"
MAF_DIR = GENOMIC_DIR / "maf_files"
CLINICAL_CSV = GENOMIC_DIR / "clinical.csv"
MANIFEST_JSON = GENOMIC_DIR / "patient_manifest.json"

IMAGING_METADATA = Path("/hpcwork/p0021834/datasets/TCGA-LIHC/download/doiJNLP-TCGA-LIHC-01-30-2017/metadata.csv")


def gdc_post(endpoint, payload, timeout=60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                raise


def get_target_patients():
    """Return patient IDs to download MAFs for.

    Uses the clinical CSV (all TCGA-LIHC patients) so that every patient with
    clinical annotations gets mutation data — not just the imaging subset.
    Falls back to imaging metadata if clinical CSV doesn't exist yet.
    """
    if CLINICAL_CSV.exists():
        import pandas as pd
        df = pd.read_csv(CLINICAL_CSV)
        log.info("Using clinical CSV: %d patients", len(df))
        return sorted(df['patient_id'].unique().tolist())
    if IMAGING_METADATA.exists():
        patients = set()
        with open(IMAGING_METADATA) as f:
            for row in csv.DictReader(f):
                patients.add(row["Subject ID"])
        log.info("Using imaging metadata: %d patients", len(patients))
        return sorted(patients)
    log.warning("No clinical or imaging metadata — downloading for ALL LIHC patients")
    return None


def download_maf_files(patient_ids):
    MAF_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Querying GDC for MAF files (%s, %d patients)...", PROJECT, len(patient_ids) if patient_ids else "all")

    all_hits = []
    ids = patient_ids or []
    # If no patient_ids, query all LIHC
    batch_size = 50
    if not ids:
        filters = {"op": "and", "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": PROJECT}},
            {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
            {"op": "=", "content": {"field": "data_format", "value": "maf"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ]}
        resp = gdc_post(GDC_FILES, {"filters": filters, "fields": "file_id,file_name,file_size,cases.submitter_id", "size": 5000})
        all_hits = resp["data"]["hits"]
    else:
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            filters = {"op": "and", "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": PROJECT}},
                {"op": "in", "content": {"field": "cases.submitter_id", "value": batch}},
                {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
                {"op": "=", "content": {"field": "data_format", "value": "maf"}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ]}
            resp = gdc_post(GDC_FILES, {"filters": filters, "fields": "file_id,file_name,file_size,cases.submitter_id", "size": 1000})
            all_hits.extend(resp["data"]["hits"])

    # Deduplicate
    seen = set()
    manifest = []
    for h in all_hits:
        if h["file_id"] in seen:
            continue
        seen.add(h["file_id"])
        manifest.append({
            "patient_id": h["cases"][0]["submitter_id"],
            "file_id": h["file_id"],
            "file_name": h["file_name"],
            "file_size": h["file_size"],
        })

    log.info("Found %d MAF files", len(manifest))
    GENOMIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2)

    # Download
    downloaded, failed = 0, 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        def dl(entry):
            pid = entry["patient_id"]
            out = MAF_DIR / f"{pid}.maf.gz"
            if out.exists() and out.stat().st_size > 0:
                return pid
            url = f"{GDC_DATA}/{entry['file_id']}"
            try:
                out.write_bytes(urllib.request.urlopen(url, timeout=120).read())
                return pid
            except Exception as e:
                log.error("Failed: %s: %s", pid, e)
                return None

        for fut in as_completed([pool.submit(dl, e) for e in manifest]):
            if fut.result():
                downloaded += 1
            else:
                failed += 1
            if (downloaded + failed) % 20 == 0:
                log.info("  %d/%d downloaded", downloaded, len(manifest))

    log.info("MAF: %d downloaded, %d failed", downloaded, failed)


def download_clinical(patient_ids):
    GENOMIC_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading clinical data...")

    # Query ALL LIHC cases
    filters = {"op": "=", "content": {"field": "project.project_id", "value": PROJECT}}
    fields = [
        "submitter_id", "demographic.gender", "demographic.race", "demographic.ethnicity",
        "demographic.vital_status", "demographic.days_to_death", "demographic.age_at_index",
        "diagnoses.ajcc_pathologic_stage", "diagnoses.ajcc_pathologic_t",
        "diagnoses.ajcc_pathologic_n", "diagnoses.ajcc_pathologic_m",
        "diagnoses.tumor_grade", "diagnoses.primary_diagnosis", "diagnoses.morphology",
        "diagnoses.days_to_last_follow_up", "diagnoses.age_at_diagnosis",
        "diagnoses.tissue_or_organ_of_origin",
    ]
    resp = gdc_post(GDC_CASES, {"filters": filters, "fields": ",".join(fields), "size": 1000})
    cases = resp["data"]["hits"]
    log.info("Got %d cases from GDC", len(cases))

    rows = []
    for c in cases:
        demo = c.get("demographic", {})
        diag = c.get("diagnoses", [{}])[0] if c.get("diagnoses") else {}
        rows.append({
            "patient_id": c["submitter_id"],
            "gender": demo.get("gender", ""),
            "race": demo.get("race", ""),
            "ethnicity": demo.get("ethnicity", ""),
            "vital_status": demo.get("vital_status", ""),
            "days_to_death": demo.get("days_to_death", ""),
            "age_at_index": demo.get("age_at_index", ""),
            "ajcc_stage": diag.get("ajcc_pathologic_stage", ""),
            "ajcc_t": diag.get("ajcc_pathologic_t", ""),
            "ajcc_n": diag.get("ajcc_pathologic_n", ""),
            "ajcc_m": diag.get("ajcc_pathologic_m", ""),
            "tumor_grade": diag.get("tumor_grade", ""),
            "primary_diagnosis": diag.get("primary_diagnosis", ""),
            "morphology": diag.get("morphology", ""),
            "days_to_last_follow_up": diag.get("days_to_last_follow_up", ""),
            "age_at_diagnosis": diag.get("age_at_diagnosis", ""),
            "tissue_or_organ": diag.get("tissue_or_organ_of_origin", ""),
        })

    with open(CLINICAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Clinical data saved: %d patients → %s", len(rows), CLINICAL_CSV)


if __name__ == "__main__":
    # Download clinical first (for all TCGA-LIHC), then use those patient IDs for MAFs.
    download_clinical(None)
    patient_ids = get_target_patients()
    download_maf_files(patient_ids)
    log.info("Done!")
