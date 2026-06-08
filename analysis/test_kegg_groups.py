"""KEGG functional-group membership of the FDR-significant imaging-correlating
genes — no enrichment test, just "which group does each hit fall into?"

For each cohort, we take the FDR-significant genes from
discovery_wide_table.csv (panel-validated and non-OncoKB, pooled), and check
each gene's membership in a small curated set of high-level KEGG functional
groups (calcium, MAPK, PI3K-Akt, etc.).

Output: per cohort, a table showing
  - which groups contain at least one FDR-significant gene
  - which genes belong to each group
  - genes that fall outside every group ("ungrouped" — interesting candidates)

This is the question "are there groups of genes affected?" answered
descriptively, not as a hypergeometric test.

Usage:
  python -m analysis.test_kegg_groups
  python -m analysis.test_kegg_groups --hit-fdr 0.10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)


# ─── KEGG functional groups ────────────────────────────────────────────────
# Each group is a high-level functional bucket; the value is the list of
# KEGG hsa IDs whose member genes belong to that group. A gene belongs to a
# group if it's in ANY of the group's pathways.
KEGG_GROUPS = {
    "Calcium signaling":          ["hsa04020"],
    "MAPK / Ras / RTK":           ["hsa04010", "hsa04014", "hsa04015",
                                    "hsa04012", "hsa04068"],
    "PI3K-Akt / mTOR":            ["hsa04151", "hsa04150"],
    "Wnt / Notch / Hedgehog":     ["hsa04310", "hsa04330", "hsa04340"],
    "Hormone (estrogen / progest.)": ["hsa04915", "hsa04914"],
    "HIF-1 / VEGF":               ["hsa04066", "hsa04370"],
    "TGF-β":                      ["hsa04350"],
    "DNA damage / repair":        ["hsa03460", "hsa03430", "hsa03440",
                                    "hsa03450", "hsa03420", "hsa03410"],
    "Cell cycle / apoptosis / p53": ["hsa04110", "hsa04115", "hsa04210"],
    "Ubiquitin / proteasome":     ["hsa04120"],
    "Adhesion / cytoskeleton":    ["hsa04510", "hsa04520", "hsa04530",
                                    "hsa04810"],
}


# ─── KEGG cache loader (inlined; avoids scipy.stats import side-effect) ────

KEGG_CACHE_PATH = Path("data/kegg_pathways.json")
KEGG_REST_URL = "https://rest.kegg.jp/get/{}"


def _parse_kegg_gene_section(text):
    in_genes = False
    symbols = []
    for line in text.splitlines():
        if line.startswith("GENE"):
            in_genes = True
            line = line[len("GENE"):]
        elif in_genes and (not line.startswith(" ") and not line.startswith("\t")):
            in_genes = False
        if not in_genes:
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        symbols.append(parts[1].split(";", 1)[0].strip())
    return symbols


def fetch_kegg_pathway_genes(pathway_id, cache_path=KEGG_CACHE_PATH):
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}
    if pathway_id in cache:
        return set(cache[pathway_id])
    log.info("  Fetching KEGG pathway %s", pathway_id)
    with urllib.request.urlopen(KEGG_REST_URL.format(pathway_id), timeout=30) as resp:
        text = resp.read().decode("utf-8")
    symbols = sorted(set(_parse_kegg_gene_section(text)))
    if not symbols:
        return set()
    cache[pathway_id] = symbols
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return set(symbols)


def bh_fdr(p_values):
    n = len(p_values)
    if n == 0:
        return np.array([])
    p = np.asarray(p_values, dtype=float)
    idx = np.argsort(p)
    sp = p[idx]
    adj = np.empty(n)
    adj[idx[-1]] = sp[-1]
    for i in range(n - 2, -1, -1):
        adj[idx[i]] = min(adj[idx[i + 1]], sp[i] * n / (i + 1))
    return np.clip(adj, 0, 1)


# ─── Group resolution: build {group_name -> set of all member genes} ────

def resolve_groups():
    out = {}
    for name, pids in KEGG_GROUPS.items():
        union = set()
        for pid in pids:
            try:
                union |= fetch_kegg_pathway_genes(pid)
            except Exception as e:                        # noqa: BLE001
                log.warning("  failed %s in group '%s': %s", pid, name, e)
        out[name] = union
    return out


# ─── Main ────────────────────────────────────────────────────────────────

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-table", default="results/final_results/discovery_wide_table.csv")
    parser.add_argument("--out-dir",    default="results/final_results/exploration")
    parser.add_argument("--hit-fdr",    type=float, default=0.05,
                        help="FDR cutoff to define 'hit' (default 0.05)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    wide = pd.read_csv(args.wide_table)
    df_best = (wide.sort_values("p_res")
                    .drop_duplicates(["cohort", "gene"]).reset_index(drop=True))

    log.info("Resolving KEGG group → gene-set map ...")
    groups = resolve_groups()
    log.info("  %d groups resolved", len(groups))

    lines = []
    lines.append("=" * 78)
    lines.append("KEGG functional-group membership — FDR-significant hits per cohort")
    lines.append(f"Hit cutoff: FDR_unique < {args.hit_fdr}")
    lines.append(f"Functional groups tested: {len(groups)}")
    lines.append("=" * 78)

    long_rows = []                      # for the gene × group CSV
    summary_rows = []                   # for the per-cohort × group counts

    for cohort in ["KIRC", "LIHC", "BRCA"]:
        sub = df_best[df_best["cohort"] == cohort]
        hits = sub[sub["fdr_full"] < args.hit_fdr].copy()
        if hits.empty:
            lines.append(f"\n{cohort}: 0 FDR-significant genes — skipped")
            continue
        hits["origin"] = np.where(hits["CGC"], "panel-validated", "non-OncoKB")
        all_genes = set(hits["gene"].unique())
        lines.append(f"\n{cohort} — {len(all_genes)} FDR-significant genes "
                     f"(panel-validated: {(hits['origin']=='panel-validated').sum()}; "
                     f"non-OncoKB: {(hits['origin']=='non-OncoKB').sum()})")
        lines.append("-" * 78)

        # Per-gene group memberships
        for _, r in hits.iterrows():
            for gname, gset in groups.items():
                if r["gene"] in gset:
                    long_rows.append({"cohort": cohort, "gene": r["gene"],
                                       "origin": r["origin"], "group": gname})

        # Per-group counts within this cohort
        for gname, gset in groups.items():
            members = all_genes & gset
            if not members:
                continue
            n_pv = sum(1 for g in members
                       if (hits[hits["gene"] == g]["origin"].iloc[0] == "panel-validated"))
            n_nv = len(members) - n_pv
            summary_rows.append({"cohort": cohort, "group": gname,
                                  "n_total": len(members),
                                  "n_panel_validated": n_pv,
                                  "n_non_oncokb": n_nv,
                                  "genes": ",".join(sorted(members))})
            lines.append(f"  {gname:<32} {len(members):>3} "
                         f"(pv={n_pv}, novel={n_nv})  "
                         f"{', '.join(sorted(members))}")

        # Genes that hit no group at all
        in_any = set()
        for gset in groups.values():
            in_any |= (all_genes & gset)
        ungrouped = all_genes - in_any
        if ungrouped:
            lines.append(f"  {'(ungrouped — outside all KEGG groups)':<32} "
                         f"{len(ungrouped):>3}        {', '.join(sorted(ungrouped))}")

    # Save outputs
    if long_rows:
        long_df = pd.DataFrame(long_rows)
        long_df.to_csv(out_dir / "kegg_groups_gene_table.csv", index=False)
    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        sum_df.to_csv(out_dir / "kegg_groups_summary.csv", index=False)

    txt = "\n".join(lines)
    print(txt)
    (out_dir / "kegg_groups_summary.txt").write_text(txt + "\n")
    print(f"\nWritten:\n  {out_dir / 'kegg_groups_summary.txt'}\n  "
          f"{out_dir / 'kegg_groups_summary.csv'}\n  "
          f"{out_dir / 'kegg_groups_gene_table.csv'}")


if __name__ == "__main__":
    run()
