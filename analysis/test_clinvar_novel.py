"""ClinVar profile of novel (non-OncoKB) FDR-significant imaging-correlating
genes.

Question: are these novel discoveries already in ClinVar (i.e. associated
with at least one curated germline disease variant)? If yes, the genes are
not biological dark matter — they have prior medical-genetics evidence of
phenotype-relevance even if not classified as cancer drivers.

Sources (NCBI ClinVar FTP, both small):
  - gene_specific_summary.txt   — per-gene ClinVar allele/submission counts
  - gene_condition_source_id    — per-gene disease-condition associations

Procedure:
  1. Download / cache the two ClinVar tables (≈3 MB total).
  2. Per cohort, split tested genes into:
       OncoKB-known FDR-sig, novel FDR-sig, non-sig tested.
  3. Look up ClinVar variant count + distinct condition count for each gene.
  4. Report:
       - Per-gene table (novel hits sorted by ClinVar variant count).
       - Fraction of each group with any ClinVar entry / disease association.
       - Fisher's exact: novel-sig vs non-sig (one-sided).

Output:
  results/final_results/exploration/clinvar_novel_table.csv
  results/final_results/exploration/clinvar_novel_summary.csv
  text summary printed to stdout.

Usage:
  python -m analysis.test_clinvar_novel
"""
from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

CLINVAR_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar"
# `gene_specific_summary.txt` lives under /tab_delimited/; `gene_condition_source_id`
# is at the FTP root.
GENE_SUMMARY_URL = f"{CLINVAR_BASE}/tab_delimited/gene_specific_summary.txt"
GENE_CONDITION_URL = f"{CLINVAR_BASE}/gene_condition_source_id"
CACHE_DIR = Path("data/clinvar")


def _download_if_missing(url, dest, force=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    log.info("Downloading %s → %s", url, dest)
    with urllib.request.urlopen(url, timeout=180) as r, dest.open("wb") as f:
        f.write(r.read())
    return dest


def load_clinvar(force=False):
    """Returns (gene_summary_df, gene_condition_df). Both indexed by gene
    symbol where possible. Empty frames if download fails."""
    gs_path = CACHE_DIR / "gene_specific_summary.txt"
    gc_path = CACHE_DIR / "gene_condition_source_id"
    try:
        _download_if_missing(GENE_SUMMARY_URL, gs_path, force=force)
        _download_if_missing(GENE_CONDITION_URL, gc_path, force=force)
    except Exception as e:                             # noqa: BLE001
        log.warning("ClinVar download failed (%s) — continuing with cache "
                    "if present", e)

    gs = pd.DataFrame()
    if gs_path.exists():
        # Header row begins with '#' so we read it manually. Schema (2026):
        # Symbol  GeneID  Total_submissions  Total_alleles
        # Submissions_reporting_this_gene  Alleles_reported_Pathogenic_Likely_pathogenic
        # Gene_MIM_number  Number_uncertain  Number_with_conflicts
        cols = ["Symbol", "GeneID", "Total_submissions", "Total_alleles",
                 "Submissions_reporting_this_gene",
                 "Alleles_reported_Pathogenic_Likely_pathogenic",
                 "Gene_MIM_number", "Number_uncertain", "Number_with_conflicts"]
        gs = pd.read_csv(gs_path, sep="\t", comment="#", names=cols,
                          header=None, low_memory=False)
        for col in ("Total_submissions", "Total_alleles",
                    "Submissions_reporting_this_gene",
                    "Alleles_reported_Pathogenic_Likely_pathogenic",
                    "Number_uncertain", "Number_with_conflicts"):
            gs[col] = pd.to_numeric(gs[col], errors="coerce")
        gs = gs.drop_duplicates("Symbol")
    gc = pd.DataFrame()
    if gc_path.exists():
        # File is tab-separated with a header row starting "#GeneID".
        # Schema (2026): #GeneID  AssociatedGenes  RelatedGenes  ConceptID
        # DiseaseName  SourceName  SourceID  DiseaseMIM  LastUpdated.
        # A gene symbol can appear in either AssociatedGenes or RelatedGenes
        # (occasionally both); we expand into a long table keyed by GeneSymbol.
        raw = pd.read_csv(gc_path, sep="\t", low_memory=False)
        raw.columns = [c.lstrip("#") for c in raw.columns]
        if "AssociatedGenes" in raw.columns and "RelatedGenes" in raw.columns:
            ag = raw[raw["AssociatedGenes"].notna() &
                       (raw["AssociatedGenes"] != "")].copy()
            ag["GeneSymbol"] = ag["AssociatedGenes"]
            rg = raw[raw["RelatedGenes"].notna() &
                       (raw["RelatedGenes"] != "")].copy()
            rg["GeneSymbol"] = rg["RelatedGenes"]
            gc = pd.concat([ag, rg], ignore_index=True)
        elif "GeneSymbol" in raw.columns:
            gc = raw
    return gs, gc


def per_gene_clinvar(genes, gs, gc):
    """For each gene symbol, return ClinVar variant count + distinct
    condition count (NaN if not present in ClinVar)."""
    rows = []
    gs_idx = gs.set_index("Symbol") if not gs.empty else None
    gc_grp = (gc.groupby("GeneSymbol")["DiseaseName"].nunique()
              if (not gc.empty and "GeneSymbol" in gc.columns
                  and "DiseaseName" in gc.columns)
              else None)
    for g in genes:
        n_subs = n_alleles = n_conditions = np.nan
        n_path = np.nan
        in_clinvar = False
        if gs_idx is not None and g in gs_idx.index:
            row = gs_idx.loc[g]
            n_subs    = float(row["Total_submissions"]) if pd.notna(row["Total_submissions"]) else np.nan
            n_alleles = float(row["Total_alleles"])     if pd.notna(row["Total_alleles"])     else np.nan
            n_path    = float(row["Alleles_reported_Pathogenic_Likely_pathogenic"]) \
                          if pd.notna(row["Alleles_reported_Pathogenic_Likely_pathogenic"]) else np.nan
            in_clinvar = (n_alleles or 0) > 0
        if gc_grp is not None and g in gc_grp.index:
            n_conditions = int(gc_grp.loc[g])
        rows.append({"gene": g,
                     "clinvar_total_submissions": n_subs,
                     "clinvar_total_alleles": n_alleles,
                     "clinvar_path_lp_alleles": n_path,
                     "clinvar_n_conditions": n_conditions,
                     "in_clinvar": bool(in_clinvar)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-table", default="results/final_results/discovery_wide_table.csv")
    parser.add_argument("--out-dir",    default="results/final_results/exploration")
    parser.add_argument("--hit-fdr",    type=float, default=0.05)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading ClinVar tables (cached at %s) ...", CACHE_DIR)
    gs, gc = load_clinvar(force=args.force_download)
    if gs.empty and gc.empty:
        log.error("No ClinVar data available — aborting")
        return
    log.info("  gene_specific_summary: %d genes", len(gs))
    log.info("  gene_condition: %d gene-condition rows", len(gc))

    wide = pd.read_csv(args.wide_table)
    df_best = (wide.sort_values("p_res")
                    .drop_duplicates(["cohort", "gene"]).reset_index(drop=True))

    cohorts = ["KIRC", "LIHC", "BRCA"]
    long_rows = []
    summary_rows = []
    for c in cohorts:
        sub = df_best[df_best["cohort"] == c]
        if sub.empty:
            continue
        hits     = sub[sub["fdr_full"] <  args.hit_fdr]
        non      = sub[sub["fdr_full"] >= args.hit_fdr]
        groups = {
            "OncoKB-FDR-sig":  hits[hits["CGC"]]["gene"].tolist(),
            "novel-FDR-sig":   hits[~hits["CGC"]]["gene"].tolist(),
            "non-sig-tested":  non["gene"].tolist(),
        }
        for grp_name, gene_list in groups.items():
            if not gene_list:
                continue
            tab = per_gene_clinvar(gene_list, gs, gc)
            tab.insert(0, "group",  grp_name)
            tab.insert(0, "cohort", c)
            long_rows.append(tab)

            n = len(tab)
            n_in    = int(tab["in_clinvar"].sum())
            n_disease = int((tab["clinvar_n_conditions"].fillna(0) >= 1).sum())
            mean_alleles = float(tab["clinvar_total_alleles"]
                                   .fillna(0).mean())
            summary_rows.append({"cohort": c, "group": grp_name,
                                  "n_genes": n,
                                  "n_in_clinvar": n_in,
                                  "frac_in_clinvar": n_in / n,
                                  "n_with_disease": n_disease,
                                  "frac_with_disease": n_disease / n,
                                  "mean_clinvar_alleles": mean_alleles})

    if not long_rows:
        log.warning("No genes to query — exiting")
        return
    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_csv(out_dir / "clinvar_novel_table.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "clinvar_novel_summary.csv", index=False)

    print("\n" + "=" * 78)
    print(f"ClinVar profile of FDR-significant imaging genes "
          f"(fdr_full < {args.hit_fdr})")
    print("=" * 78)
    for c in cohorts:
        sub = summary_df[summary_df["cohort"] == c]
        if sub.empty:
            continue
        print(f"\n  {c}")
        print(f"    {'group':<18} {'n':>4} {'in_ClinVar':>11} {'%in':>6} "
              f"{'with_disease':>13} {'%dis':>6} {'mean_alleles':>13}")
        for _, r in sub.iterrows():
            print(f"    {r['group']:<18} {int(r['n_genes']):>4d} "
                  f"{int(r['n_in_clinvar']):>11d} "
                  f"{r['frac_in_clinvar']*100:>5.0f}% "
                  f"{int(r['n_with_disease']):>13d} "
                  f"{r['frac_with_disease']*100:>5.0f}% "
                  f"{r['mean_clinvar_alleles']:>13.1f}")
        # Fisher: novel-FDR-sig vs non-sig-tested for "in ClinVar"
        if {"novel-FDR-sig", "non-sig-tested"}.issubset(set(sub["group"])):
            n_row = sub[sub["group"] == "novel-FDR-sig"].iloc[0]
            x_row = sub[sub["group"] == "non-sig-tested"].iloc[0]
            n_in, n_tot = int(n_row["n_in_clinvar"]), int(n_row["n_genes"])
            x_in, x_tot = int(x_row["n_in_clinvar"]), int(x_row["n_genes"])
            try:
                _, p = stats.fisher_exact(
                    [[n_in, n_tot - n_in],
                     [x_in, x_tot - x_in]],
                    alternative="greater")
            except Exception:                              # noqa: BLE001
                p = float("nan")
            fold = ((n_in / n_tot) / (x_in / x_tot)
                     if x_in else float("inf"))
            star = ("***" if p < 1e-3 else "**" if p < 1e-2
                    else "*" if p < 5e-2 else "ns")
            print(f"    Fisher (novel vs non-sig, in_ClinVar):  "
                  f"fold={fold:.2f}×, $p$={p:.3g} {star}")

    # Per-cohort: list novel genes with ClinVar entries (most-burdened first)
    print("\n--- Novel FDR-sig genes with ClinVar entries ---")
    for c in cohorts:
        nv = long_df[(long_df["cohort"] == c) &
                       (long_df["group"] == "novel-FDR-sig") &
                       (long_df["in_clinvar"])]
        if nv.empty:
            print(f"\n  {c}: no novel FDR-sig genes in ClinVar")
            continue
        nv = nv.sort_values("clinvar_total_alleles", ascending=False)
        print(f"\n  {c}: {len(nv)} novel FDR-sig genes in ClinVar")
        print(f"    {'gene':<10} {'alleles':>8} {'path/LP':>8} "
              f"{'submissions':>11} {'conditions':>10}")
        for _, r in nv.iterrows():
            n_cond = (int(r['clinvar_n_conditions'])
                       if pd.notna(r['clinvar_n_conditions']) else 0)
            n_path = (int(r['clinvar_path_lp_alleles'])
                       if pd.notna(r['clinvar_path_lp_alleles']) else 0)
            print(f"    {r['gene']:<10} {int(r['clinvar_total_alleles']):>8d} "
                  f"{n_path:>8d} "
                  f"{int(r['clinvar_total_submissions']):>11d} "
                  f"{n_cond:>10d}")

    print(f"\nWritten: {out_dir / 'clinvar_novel_table.csv'}")
    print(f"         {out_dir / 'clinvar_novel_summary.csv'}")


if __name__ == "__main__":
    main()
