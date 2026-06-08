#!/usr/bin/env python3
"""
Batch-score all somatic mutations with Evo2 delta log-likelihood.

Uses Evo2's built-in score_sequences() method (same as the official BRCA1
variant effect prediction notebook). For each mutation:
  - Extracts a context window centered on the mutation from hg38
  - Scores both the reference and mutated sequence
  - delta_ll = score(mutated) - score(reference)
  - Negative delta = mutation is disruptive

Default context: 4096bp each side = 8192bp total (official recommendation).
Resume-friendly: skips patients already in the output CSV.

Usage:
    python genomics/mutation_scoring.py [--max-patients N] [--context-bp 4096]
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataloader import TCGAKIRCDataset, build_mutation_sequences

log = logging.getLogger(__name__)

# Default scoring context: 4096bp each side = 8192bp total window
# (matches the official Evo2 BRCA1 VEP notebook which uses 8192bp)
DEFAULT_SCORING_CONTEXT_BP = 4096


def load_scored_patients(output_csv: Path) -> set[str]:
    """Load patient IDs already scored (for resume)."""
    if not output_csv.exists():
        return set()
    scored = set()
    with open(output_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scored.add(row["patient_id"])
    return scored


def score_patient(model, patient, context_bp: int, max_mutations: int = 0,
                   batch_size: int = 4) -> list[dict]:
    """Score all mutations for one patient using model.score_sequences().

    Args:
        batch_size: Number of sequences scored in parallel on the GPU.
            Higher = faster but more VRAM.  H100 80 GB fits batch_size=8
            comfortably at 8192 bp context.  Default 4 is conservative.
    """
    sequences = build_mutation_sequences(
        patient.mutations, context_bp=context_bp
    )
    if not sequences:
        return []

    if max_mutations > 0:
        sequences = sequences[:max_mutations]

    # Batch all ref and mut sequences for efficient scoring
    ref_seqs = [s["ref_seq"] for s in sequences]
    mut_seqs = [s["mut_seq"] for s in sequences]

    ref_scores = model.score_sequences(ref_seqs, batch_size=batch_size)
    mut_scores = model.score_sequences(mut_seqs, batch_size=batch_size)

    results = []
    for seq_info, ref_ll, mut_ll in zip(sequences, ref_scores, mut_scores):
        gene = seq_info["gene"]
        vc = seq_info["variant_class"]

        if gene in config.KIRC_DRIVER_GENES and vc in config.FUNCTIONAL_MUTATIONS:
            category = "driver"
        elif vc in config.FUNCTIONAL_MUTATIONS:
            category = "functional"
        elif vc in config.SILENT_MUTATIONS:
            category = "silent"
        else:
            category = "other"

        results.append({
            "patient_id": patient.patient_id,
            "gene": gene,
            "variant_class": vc,
            "category": category,
            "chrom": seq_info["chrom"],
            "position": seq_info["position"],
            "ref_ll": ref_ll,
            "mut_ll": mut_ll,
            "delta_ll": mut_ll - ref_ll,
        })

    return results


FIELDNAMES = [
    "patient_id", "gene", "variant_class", "category",
    "chrom", "position", "ref_ll", "mut_ll", "delta_ll",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patients", type=int, default=0,
                        help="Limit number of patients (0 = all)")
    parser.add_argument("--max-mutations", type=int, default=0,
                        help="Limit mutations per patient (0 = all)")
    parser.add_argument("--context-bp", type=int, default=DEFAULT_SCORING_CONTEXT_BP,
                        help="Context bp each side of mutation (default: 4096 = 8192bp total)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="GPU batch size for sequence scoring (default: 4, H100 fits 8)")
    parser.add_argument("--output", type=str, default=str(config.MUTATION_SCORES_CSV))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("Scoring context: %dbp each side = %dbp total window",
             args.context_bp, args.context_bp * 2)

    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    scored = load_scored_patients(output_csv)
    log.info("Already scored: %d patients", len(scored))

    # Load dataset
    dataset = TCGAKIRCDataset(require_both=False, load_sequences=False)
    patients = [p for p in dataset.patients if p.n_mutations > 0 and p.patient_id not in scored]
    if args.max_patients > 0:
        patients = patients[:args.max_patients]
    log.info("Patients to score: %d", len(patients))

    if not patients:
        log.info("Nothing to do.")
        return

    # Load Evo2
    from evo2 import Evo2
    log.info("Loading Evo2 model: %s", config.EVO2_MODEL)
    model = Evo2(config.EVO2_MODEL)

    # Open CSV in append mode
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, patient in enumerate(patients):
            log.info("[%d/%d] Scoring %s (%d mutations)...",
                     i + 1, len(patients), patient.patient_id, patient.n_mutations)
            try:
                results = score_patient(model, patient, args.context_bp, args.max_mutations,
                                       batch_size=args.batch_size)
                writer.writerows(results)
                f.flush()
                n_deleterious = sum(1 for r in results if r["delta_ll"] < 0)
                log.info("  Scored %d mutations, %d deleterious (%.0f%%)",
                         len(results), n_deleterious,
                         100 * n_deleterious / len(results) if results else 0)
            except Exception as e:
                log.error("  Failed for %s: %s", patient.patient_id, e)

    log.info("Done. Results saved to %s", output_csv)


if __name__ == "__main__":
    main()
