#!/usr/bin/env python3
"""
One-command driver: recreate Figure 1 (rediscovery) and Figure 2 (discovery).

Fast path (default): draws both figures from the intermediate tables already
shipped under results/ — no GPU, no reference genome, no network.

    python make_figures.py
    python make_figures.py --regenerate-tables   # rebuild discovery_wide_table.csv first

With --regenerate-tables, the TMB-residualized per-gene-metric x imaging sweep
(discovery_wide_table.csv) is rebuilt from the shipped mutation-score and
radiomic-feature CSVs before drawing. This reproduces the table byte-for-byte;
it does NOT re-run Evo2 scoring or radiomic segmentation (see README for the
full-from-raw pipeline).

Outputs: results/final_results/paper/fig1_rediscovery.{png,pdf}
         results/final_results/paper/fig2_discovery.{png,pdf}
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate-tables", action="store_true",
                    help="Rebuild discovery_wide_table.csv (and the KEGG-group / "
                         "ClinVar annotation tables) from the shipped CSVs first.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    py = sys.executable

    if args.regenerate_tables:
        for cmd in (
            [py, "analysis/comprehensive_results.py"],   # -> discovery_wide_table.csv
            [py, "-m", "analysis.test_kegg_groups"],     # -> kegg_groups_*.csv (Fig 2b)
            [py, "-m", "analysis.test_clinvar_novel"],   # -> clinvar_novel_table.csv (Fig 2e)
        ):
            logging.info("Running: %s", " ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)

    subprocess.run([py, "analysis/publication_figures.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
