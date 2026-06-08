#!/usr/bin/env python3
"""
Publication-grade figures (Nature-tier style) for the gene-radiology discovery.

Two figures, each built from the TMB-residualized per-gene-metric x imaging
sweep (results/final_results/discovery_wide_table.csv):

  Figure 1 — REDISCOVERY of OncoKB-known cancer drivers
    Per-cohort top-10 panel-validated (OncoKB/CGC) genes whose Evo2 |ΔLL|
    severity correlates with a radiomic feature (TMB-adjusted partial
    Spearman), plus an OncoKB-enrichment panel. Shows the pipeline recovers
    established cancer genes.

  Figure 2 — DISCOVERY of novel imaging-correlating genes
    Per-cohort top-10 non-OncoKB genes (a, c, d), KEGG functional-group
    composition of the FDR-significant hits (b), external ClinVar + OMIM
    disease-burden lookup for the novel KIRC genes (e), and cross-metric
    concordance across all reported genes (f).

Style: sans-serif, no title chrome, consistent colour palette, units in axis
labels, vector-friendly DPI.

Outputs (PNG + PDF):
  results/final_results/paper/fig1_rediscovery.{png,pdf}
  results/final_results/paper/fig2_discovery.{png,pdf}

Usage:
    python analysis/publication_figures.py
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.comprehensive_results import (
    DATASETS, load_dataset, bh_fdr, pretty_label,
)

log = logging.getLogger(__name__)


# ─── Style ──────────────────────────────────────────────────────────────────
# Three-tier significance palette (uniform across all figures):
#   ns      = grey   (FDR≥0.05 and p≥0.05)
#   nominal = blue   (nominal p<0.05 only, did not survive FDR)
#   fdr     = violet (FDR-corrected significance)
# Heatmap cell intensity is shaded within each category by |r|, using the
# sequential "Greys"/"Blues"/"Purples" colormaps clamped to the upper end.
PALETTE = {
    "ns":          "#9ca3af",   # grey
    "nominal":     "#1d4ed8",   # blue
    "fdr":         "#7c3aed",   # violet
    "tmb":         "#e67e22",   # TMB-confounded (kept distinct for Fig 3 cliffs)
    # Backward-compat aliases (kept for any caller still using old keys)
    "novel":       "#7c3aed",
    "cgc":         "#7c3aed",
    "border":      "#1d4ed8",
    "carrier":     "#7c3aed",
    "non_carrier": "#9ca3af",
    "div_cmap":    "RdBu_r",    # legacy — heatmaps now use category-shading
}


def _tint(hex_color, intensity):
    """Blend a base color toward white. intensity=0 → white, =1 → solid base."""
    import matplotlib.colors as mcolors
    base = np.array(mcolors.to_rgb(hex_color))
    rgb  = (1.0 - intensity) * np.ones(3) + intensity * base
    return tuple(rgb)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Render parameters for the nominal mark — bigger and bolder than asterisks.
def save_dual(fig, out_dir, basename):
    """Save both PNG and PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{basename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{basename}.pdf", bbox_inches="tight")


# ───────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Validation: per-pathway × imaging (TMB-residualized)
# ───────────────────────────────────────────────────────────────────────────

def _draw_cohort_bar_strip(fig, gs_cell, ds_key, tab, *, cohort_title=None):
    """Render one cohort's bar (r_res) inside the given gridspec cell.
    Shared by Fig 1 (rediscovery) and Fig 2-top.

    The per-carrier raw-data strip lives in a separate figure (Fig 1b / Fig 2b)
    so this panel stays compact.
    """
    ax = fig.add_subplot(gs_cell)

    tab = tab.sort_values("r_res").reset_index(drop=True)
    y_pos = np.arange(len(tab))
    colors = []
    for _, row in tab.iterrows():
        if row["fdr_full"] < 0.05:
            colors.append(PALETTE["fdr"])
        elif row["p_res"] < 0.05:
            colors.append(PALETTE["nominal"])
        else:
            colors.append(PALETTE["ns"])
    ax.barh(y_pos, tab["r_res"].to_numpy(), color=colors,
            edgecolor="white", linewidth=0.6, height=0.72)
    ax.axvline(0, color="0.4", linewidth=0.6)

    x_max = max(0.50, tab["r_res"].abs().max() * 1.08)

    # Column position for the inline labels: anchor right after the longest
    # positive bar (in data coords) so labels sit close to the bars rather
    # than at the panel's right margin. A small offset keeps them clear of
    # any bar tip.
    longest_pos = max(tab["r_res"].max(), 0.0)
    label_x = longest_pos + 0.04
    for i, (_, row) in enumerate(tab.iterrows()):
        metric_label = METRIC_DISPLAY.get(row["metric"], row["metric"])
        feat_short = img_label(row["imaging"])
        text_part = f"{metric_label} × {feat_short}"
        ax.annotate(text_part, xy=(label_x, i),
                    xycoords=("data", "data"),
                    va="center", ha="left", fontsize=9, color="0.15",
                    annotation_clip=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tab["gene"].tolist(), fontsize=11, weight="bold")
    ax.set_xlim(-x_max, x_max)
    ax.set_xlabel(r"Partial Spearman $r$  (TMB-adjusted)", fontsize=12)
    if cohort_title:
        ax.set_title(cohort_title, pad=6, weight="bold", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", which="both", length=2, labelsize=10)
    return ax


def _compute_oncokb_enrichment(cohorts, hit_fdr=0.05):
    """Per-cohort 2×2 OncoKB-known enrichment among FDR-sig vs non-sig genes.
    Returns dict cohort -> stats (or None if untestable).
    Fisher's exact one-sided 'greater' (hits enriched vs non-sig)."""
    wide_path = Path("results/final_results/discovery_wide_table.csv")
    if not wide_path.exists():
        return {}
    wide = pd.read_csv(wide_path)
    df_best = (wide.sort_values("p_res")
                    .drop_duplicates(["cohort", "gene"]).reset_index(drop=True))
    out = {}
    for c in cohorts:
        sub = df_best[df_best["cohort"] == c]
        hits = sub[sub["fdr_full"] <  hit_fdr]
        non  = sub[sub["fdr_full"] >= hit_fdr]
        ho = int(hits["CGC"].sum()); hn = int((~hits["CGC"]).sum())
        no = int(non["CGC"].sum());  nn = int((~non["CGC"]).sum())
        if (ho + hn) == 0 or (no + nn) == 0:
            out[c] = None
            continue
        f_hit = ho / (ho + hn)
        f_non = no / (no + nn) if (no + nn) else 0.0
        try:
            from scipy.stats import fisher_exact
            _, p = fisher_exact([[ho, hn], [no, nn]], alternative="greater")
        except Exception:                              # noqa: BLE001
            p = float("nan")
        fold = f_hit / f_non if f_non > 0 else float("inf")
        star = ("***" if p < 1e-3 else "**" if p < 1e-2
                else "*" if p < 5e-2 else "ns")
        out[c] = {"hits_total": ho + hn, "hits_oncokb": ho, "hits_pct": f_hit*100,
                  "non_total":  no + nn, "non_oncokb":  no, "non_pct":  f_non*100,
                  "fold": fold, "p": p, "star": star}
    return out


def _draw_oncokb_enrichment_panel(fig, gs_cell, stats_per_cohort):
    """Draw a single compact OncoKB-known enrichment panel.
    Currently only KIRC has FDR-sig hits, so we show its bars; LIHC/BRCA
    are noted as untestable beneath the axes."""
    ax = fig.add_subplot(gs_cell)
    s = stats_per_cohort.get("KIRC")
    if s is None:
        ax.text(0.5, 0.5, "no testable cohort", ha="center", va="center",
                fontsize=9, transform=ax.transAxes, color="0.5")
        ax.set_axis_off()
        return ax

    bars_x = [0, 1]
    heights = [s["hits_pct"], s["non_pct"]]
    colors  = [PALETTE["fdr"], "0.55"]
    ax.bar(bars_x, heights, color=colors, edgecolor="white",
           linewidth=0.6, width=0.62)
    for x, h, fr_str in zip(bars_x, heights, [
            f"$n$={s['hits_oncokb']}/{s['hits_total']}",
            f"$n$={s['non_oncokb']}/{s['non_total']}"]):
        ax.text(x, h + 4.5, f"{h:.1f}%", ha="center", va="bottom",
                fontsize=13, weight="bold", color="0.15")
        ax.text(x, h + 1.0, fr_str, ha="center", va="bottom",
                fontsize=10, color="0.35")
    bracket_y = max(heights) * 1.32 + 6
    tick_h = 1.5
    ax.plot([0, 0, 1, 1],
            [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
            color="0.25", linewidth=0.8, clip_on=False)
    label = f"{s['fold']:.2f}×    {s['star']}" if np.isfinite(s["p"]) else ""
    ax.text(0.5, bracket_y + 0.5, label, ha="center", va="bottom",
            fontsize=12, weight="bold",
            color="0.10" if s["star"] != "ns" else "0.45")
    ax.set_xticks(bars_x)
    ax.set_xticklabels(["FDR-sig\nhits", "non-sig\ntested"], fontsize=11)
    ax.set_ylim(0, max(50, bracket_y + 8))
    ax.set_ylabel("% OncoKB-known", fontsize=12)
    ax.set_title("OncoKB-known enrichment — KIRC",
                 pad=6, weight="bold", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", which="both", length=2, labelsize=10)
    return ax


def _draw_kegg_groups_panel(fig, gs_cell, cohort="KIRC", hit_fdr=0.05):
    """Compact stacked horizontal bar — KEGG functional-group membership of
    cohort's FDR-significant hits. Segments distinguish panel-validated
    (OncoKB-known, lighter purple) from novel (non-OncoKB, full purple).
    Ungrouped genes (outside all KEGG groups) are reported separately as
    a single annotation since they would dwarf the in-group bars."""
    ax = fig.add_subplot(gs_cell)

    summary_csv = Path("results/final_results/exploration/kegg_groups_summary.csv")
    gene_csv    = Path("results/final_results/exploration/kegg_groups_gene_table.csv")
    wide_path   = Path("results/final_results/discovery_wide_table.csv")
    if not (summary_csv.exists() and gene_csv.exists() and wide_path.exists()):
        ax.text(0.5, 0.5,
                "no KEGG group data\n(run analysis.test_kegg_groups)",
                ha="center", va="center", fontsize=9,
                transform=ax.transAxes, color="0.5")
        ax.set_axis_off()
        return ax

    sm = pd.read_csv(summary_csv)
    sub = sm[sm["cohort"] == cohort].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"no KEGG groups with hits in {cohort}",
                ha="center", va="center", fontsize=9,
                transform=ax.transAxes, color="0.5")
        ax.set_axis_off()
        return ax
    sub = sub.sort_values("n_non_oncokb", ascending=True)

    # Compute ungrouped split panel-validated vs novel — for the annotation.
    wide = pd.read_csv(wide_path)
    df_best = (wide.sort_values("p_res")
                    .drop_duplicates(["cohort", "gene"]).reset_index(drop=True))
    hits = df_best[(df_best["cohort"] == cohort) &
                    (df_best["fdr_full"] < hit_fdr)]
    hit_genes = set(hits["gene"])
    pv_genes  = set(hits[hits["CGC"]]["gene"])
    nv_genes  = hit_genes - pv_genes
    g_table = pd.read_csv(gene_csv)
    grouped = set(g_table[g_table["cohort"] == cohort]["gene"])
    ungrouped = hit_genes - grouped
    n_un_pv = len(ungrouped & pv_genes)
    n_un_nv = len(ungrouped & nv_genes)

    labels = sub["group"].tolist()
    pv = sub["n_panel_validated"].to_numpy()
    nv = sub["n_non_oncokb"].to_numpy()
    y = np.arange(len(labels))

    color_pv = _tint(PALETTE["fdr"], 0.40)     # lighter purple
    color_nv = PALETTE["fdr"]                  # full purple
    ax.barh(y, pv, color=color_pv, height=0.72,
            edgecolor="white", linewidth=0.5,
            label="OncoKB-known")
    ax.barh(y, nv, left=pv, color=color_nv, height=0.72,
            edgecolor="white", linewidth=0.5,
            label="novel\n(non-OncoKB)")
    for i, (p_, n_) in enumerate(zip(pv, nv)):
        tot = int(p_ + n_)
        if tot > 0:
            ax.text(tot + 0.20, i, f"{tot}", va="center", ha="left",
                    fontsize=10, color="0.20", weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("# FDR-sig genes", fontsize=11)
    ax.set_title(f"KEGG functional groups ({cohort})",
                 pad=18, weight="bold", fontsize=12)
    max_x = max((pv + nv).max() * 1.15, 7)
    ax.set_xlim(0, max_x)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", which="both", length=2, labelsize=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False,
              handlelength=1.3, handletextpad=0.5,
              borderaxespad=0.3, labelspacing=0.4)
    # Ungrouped summary just below the title (inside the figure area).
    ax.text(0.5, 1.02,
            f"+ {n_un_pv + n_un_nv} ungrouped genes "
            f"({n_un_pv} OncoKB-known, {n_un_nv} novel)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="0.40", style="italic")
    return ax


def _load_oncokb_known_top10(cohorts):
    """Reusable: load discovery_wide_table.csv and return per-cohort
    top-10 OncoKB-known (CGC=True) gene tables (best metric per gene,
    BH-FDR within full per-cohort gene family)."""
    data = {}
    wide_path = Path("results/final_results/discovery_wide_table.csv")
    if not wide_path.exists():
        log.warning("discovery_wide_table.csv missing — figure will be empty")
        return data
    wide = pd.read_csv(wide_path)
    df_best = (wide.sort_values("p_res")
                   .drop_duplicates(subset=["cohort", "gene"], keep="first")
                   .reset_index(drop=True))
    for ds_key in cohorts:
        sub = df_best[(df_best["cohort"] == ds_key) & (df_best["CGC"])].copy()
        if sub.empty:
            continue
        data[ds_key] = sub.sort_values("p_res").head(10).reset_index(drop=True)
    return data


def figure1_rediscovery(out_dir):
    """FIG 1 — Rediscovery of OncoKB-known cancer drivers.

    2×2 layout:
      a (top-left)     KIRC top-10 OncoKB-known partial-Spearman bars
      b (top-right)    OncoKB-known enrichment in FDR-sig hits (Fisher's exact)
      c (bottom-left)  LIHC top-10 OncoKB-known bars
      d (bottom-right) BRCA top-10 OncoKB-known bars

    Per-gene carrier-vs-non-carrier raw-data view moved to figS_raw_data.
    """
    cohorts = ["KIRC", "LIHC", "BRCA"]
    log.info("Computing Figure 1 (rediscovery, 2×2 layout)...")
    data = _load_oncokb_known_top10(cohorts)
    if not data:
        log.warning("No OncoKB-known data for Fig 1 — skipping")
        return
    onc_stats = _compute_oncokb_enrichment(cohorts)

    cohort_titles = {"KIRC": "TCGA-KIRC (Kidney)",
                     "LIHC": "TCGA-LIHC (Liver)",
                     "BRCA": "TCGA-BRCA (Breast)"}
    n_gen_max = max(len(data[c]) for c in cohorts if c in data)
    cell_h = max(3.6, 0.36 * n_gen_max + 1.0)
    fig_w = 14.0
    fig_h = cell_h * 2 + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.32,
                          width_ratios=[1, 1],
                          left=0.05, right=0.985, top=0.97, bottom=0.10)

    # Cell positions: a=[0,0]=KIRC, b=[0,1]=OncoKB, c=[1,0]=LIHC, d=[1,1]=BRCA
    cell_map = {"KIRC": gs[0, 0], "LIHC": gs[1, 0], "BRCA": gs[1, 1]}
    panel_label_map = {"KIRC": "a", "LIHC": "c", "BRCA": "d"}
    label_axes = {}
    for ds_key, gs_cell in cell_map.items():
        if ds_key not in data:
            ax_empty = fig.add_subplot(gs_cell)
            ax_empty.set_title(cohort_titles[ds_key], pad=6,
                                weight="bold", fontsize=12, color="0.55")
            ax_empty.set_xticks([]); ax_empty.set_yticks([])
            for sp in ("top", "right", "left", "bottom"):
                ax_empty.spines[sp].set_visible(False)
            label_axes[panel_label_map[ds_key]] = ax_empty
            continue
        ax = _draw_cohort_bar_strip(fig, gs_cell, ds_key, data[ds_key],
                                     cohort_title=cohort_titles[ds_key])
        label_axes[panel_label_map[ds_key]] = ax

    ax_b = _draw_oncokb_enrichment_panel(fig, gs[0, 1], onc_stats)
    label_axes["b"] = ax_b

    for letter, ax in label_axes.items():
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes,
                fontsize=22, weight="bold", va="top", ha="left")

    legend_handles = [
        Patch(facecolor=PALETTE["fdr"],     label="FDR<0.05  (corrected)"),
        Patch(facecolor=PALETTE["nominal"], label="$p$<0.05  (nominal only)"),
        Patch(facecolor=PALETTE["ns"],      label="n.s."),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=11)

    save_dual(fig, out_dir, "fig1_rediscovery")
    plt.close(fig)

    for ds_key, tab in data.items():
        tab.to_csv(out_dir / f"fig1_top10_oncokb_known_{ds_key}.csv", index=False)


# Compact LaTeX-rendered labels for figure ticks/annotations. Internal feature
# names → short symbols. Spelt out in each figure's footer.
IMAGING_ABBR = {
    # Unified subscripts:
    #   T  = tumor (the tumor / lesion mask, regardless of cohort terminology)
    #   K  = kidney  (host organ, KIRC)
    #   Lv = liver   (host organ, LIHC)
    #   B  = breast  (host organ, BRCA)
    # Volumes
    "kits_tumor_volume_ml":  r"$V_\mathrm{T}$",
    "tumor_volume_ml":       r"$V_\mathrm{T}$",
    "lesion_volume_ml":      r"$V_\mathrm{T}$",
    "liver_volume_ml":       r"$V_\mathrm{Lv}$",
    "log_tumor_volume":      r"$\log V_\mathrm{T}$",
    "log_lesion_volume":     r"$\log V_\mathrm{T}$",
    "log_liver_volume":      r"$\log V_\mathrm{Lv}$",
    "log_tumor_volume_brca": r"$\log V_\mathrm{T}$",
    # Tumor / host-organ volume ratios
    "tumor_kidney_ratio":    r"$V_\mathrm{T}/V_\mathrm{K}$",
    "lesion_liver_ratio":    r"$V_\mathrm{T}/V_\mathrm{Lv}$",
    "tumor_breast_ratio":    r"$V_\mathrm{T}/V_\mathrm{B}$",
    # Texture / heterogeneity (CV of intra-tumor intensity)
    "kits_tumor_heterogeneity": r"$H$",
    "tumor_heterogeneity":      r"$H$",
    "lesion_heterogeneity":     r"$H$",
    # Necrosis (tumor mask)
    "kits_tumor_necrotic_frac": r"$f_n$",
    "lesion_necrotic_frac":     r"$f_n$",
    # Whole-liver HU (LIHC only)
    "liver_hu_mean":  r"$\overline{HU}_\mathrm{Lv}$",
    "liver_hu_std":   r"$\sigma_{HU,\mathrm{Lv}}$",
    "liver_hu_cv":    r"$CV_{HU,\mathrm{Lv}}$",
    # Tumor MRI intensity (BRCA only)
    "tumor_intensity_mean": r"$\overline{I}_\mathrm{T}$",
    "tumor_intensity_std":  r"$\sigma_{I,\mathrm{T}}$",
    "tumor_intensity_cv":   r"$CV_{I,\mathrm{T}}$",
}


def img_label(feat):
    """Compact LaTeX label for an imaging feature; falls back to pretty_label."""
    return IMAGING_ABBR.get(feat, pretty_label(feat, "short"))


# Concise labels for the 8 per-gene severity summaries — the 4 axes spanned:
#   magnitude  : max/mean/sum |ΔLL|
#   direction  : min ΔLL, max ΔLL, top-1 ΔLL (signed)
#   count      : n_mut
#   dispersion : std |ΔLL|
METRIC_DISPLAY = {
    "max_abs_dll":  r"max$|\Delta\mathrm{ll}|$",
    "mean_abs_dll": r"$\mu|\Delta\mathrm{ll}|$",
    "sum_abs_dll":  r"$\Sigma|\Delta\mathrm{ll}|$",
    "std_abs_dll":  r"$\sigma|\Delta\mathrm{ll}|$",
    "min_dll":      r"min$\,\Delta\mathrm{ll}$",
    "max_dll":      r"max$\,\Delta\mathrm{ll}$",
    "n_mutations":  r"$N_\mathrm{mut}$",
    "top1_signed":  r"$\Delta\mathrm{ll}_1$",
}


# ───────────────────────────────────────────────────────────────────────────
# FIGURE 2 (legacy) — non-pathway × COSMIC SBS heatmap. Currently NOT called
# from main(); retained as supplementary code in case the pathway-aggregate ×
# SBS view is needed later. Active Fig 2 is figure2_discovery (per-gene novel
# imaging-correlating discoveries); active Fig 3 is figure3_etiology_novel
# (carrier-vs-non SBS exposure for the novel-gene panel).
# ───────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Single-gene functional annotation + ABCA13 cross-cohort
# ───────────────────────────────────────────────────────────────────────────

def _draw_clinvar_novel_panel(fig, gs_cell, cohort="KIRC", hit_fdr=0.05,
                                top_n=10):
    """Horizontal-bar panel showing the same top-N novel (non-OncoKB) genes
    listed in Fig 2 panel a (KIRC), in the SAME order, annotated with
    external ClinVar information: pathogenic / likely-pathogenic allele
    count + the gene's primary curated disease association.

    Reads from results/final_results/exploration/clinvar_novel_table.csv
    (produced by analysis.test_clinvar_novel) and from
    data/clinvar/gene_condition_source_id (cached by the same script).
    Falls back to a "no data" placeholder if either is missing."""
    ax = fig.add_subplot(gs_cell)
    novel_csv = Path("results/final_results/exploration/clinvar_novel_table.csv")
    cond_path = Path("data/clinvar/gene_condition_source_id")
    if not (novel_csv.exists() and cond_path.exists()):
        ax.text(0.5, 0.5,
                "no ClinVar data\n(run analysis.test_clinvar_novel)",
                ha="center", va="center", fontsize=9,
                transform=ax.transAxes, color="0.5")
        ax.set_axis_off()
        return ax

    # Mirror Fig 2a: same top-N novel genes in the same r_res-ascending order.
    fig2a = _load_non_oncokb_top10([cohort]).get(cohort)
    if fig2a is None or fig2a.empty:
        ax.text(0.5, 0.5, f"no novel-FDR-sig genes in {cohort}",
                ha="center", va="center", fontsize=9,
                transform=ax.transAxes, color="0.5")
        ax.set_axis_off()
        return ax
    fig2a = (fig2a.head(top_n)
                  .sort_values("r_res", ascending=True)
                  .reset_index(drop=True))

    cv = pd.read_csv(novel_csv)
    cv = (cv[(cv["cohort"] == cohort) & (cv["group"] == "novel-FDR-sig")]
            .set_index("gene"))
    nv = fig2a.merge(
        cv[["clinvar_path_lp_alleles", "clinvar_total_alleles",
             "clinvar_n_conditions"]],
        left_on="gene", right_index=True, how="left")
    nv["clinvar_path_lp_alleles"] = nv["clinvar_path_lp_alleles"].fillna(0)

    # Disease lookup: prefer "AssociatedGenes" rows, fall back to RelatedGenes.
    cond_raw = pd.read_csv(cond_path, sep="\t", low_memory=False)
    cond_raw.columns = [c.lstrip("#") for c in cond_raw.columns]
    pieces = []
    if "AssociatedGenes" in cond_raw.columns:
        ag = cond_raw[cond_raw["AssociatedGenes"].astype(str).ne("")]
        ag = ag.assign(GeneSymbol=ag["AssociatedGenes"], _kind="associated")
        pieces.append(ag)
    if "RelatedGenes" in cond_raw.columns:
        rg = cond_raw[cond_raw["RelatedGenes"].astype(str).ne("")]
        rg = rg.assign(GeneSymbol=rg["RelatedGenes"], _kind="related")
        pieces.append(rg)
    cond = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    def _conditions(gene, omim_only):
        """Distinct disease names for a gene from gene_condition_source_id.
        omim_only=True keeps only rows carrying an OMIM phenotype MIM (the
        ClinVar/MedGen → OMIM mapping; OMIM proper is licence-gated). Returns
        names with associated rows first, shortest-name primary."""
        if cond.empty:
            return []
        sub = cond[cond["GeneSymbol"] == gene].copy()
        if omim_only:
            sub = sub[sub["DiseaseMIM"].notna()].copy()
            sub["mim"] = (sub["DiseaseMIM"].astype(str).str.split(".").str[0]
                                           .str.strip())
            sub = sub[sub["mim"].str.fullmatch(r"\d+")]
        if sub.empty:
            return []
        sub = sub.sort_values("_kind", key=lambda s: s.map(
            {"associated": 0, "related": 1}).fillna(2))
        names, seen = [], set()
        for _, r in sub.iterrows():
            nm = str(r["DiseaseName"]).strip()
            key = r["mim"] if omim_only else nm
            if not nm or key in seen:
                continue
            seen.add(key)
            names.append(nm)
        return names

    clinvar_dx = {g: _conditions(g, omim_only=False) for g in nv["gene"]}
    omim_dx = {g: _conditions(g, omim_only=True) for g in nv["gene"]}

    y = np.arange(len(nv))
    cv_counts = nv["clinvar_path_lp_alleles"].astype(float).to_numpy()
    om_counts = np.array([len(omim_dx[g]) for g in nv["gene"]], float)
    # Stronger but desaturated (dusty/grayish) fills so the bars read clearly;
    # darker matching shades for text/axes.
    CV_FILL, CV_TXT = "#7c6f9c", "#564b73"   # dusty muted purple — ClinVar
    OM_FILL, OM_TXT = "#5f9b85", "#3c6f5c"   # dusty muted teal   — OMIM
    h = 0.38                     # two half-height bars stacked within each row

    # ClinVar P/LP alleles on the primary (bottom) x-axis; OMIM phenotype count
    # on a twin (top) x-axis with its own scale (counts differ by ~3 orders of
    # magnitude). ~1.6× headroom lets the bars use the left ~60% of the panel,
    # leaving a narrower right column for the split pathology labels.
    ax.barh(y - 0.205, cv_counts, height=h, color=CV_FILL,
            edgecolor="white", linewidth=0.5, zorder=3)
    cv_max = max(cv_counts.max(), 1.0)
    cv_xlim = cv_max * 1.6
    ax.set_xlim(0, cv_xlim)

    ax2 = ax.twiny()
    ax2.barh(y + 0.205, om_counts, height=h, color=OM_FILL,
             edgecolor="white", linewidth=0.5, zorder=3)
    om_max = max(om_counts.max(), 1.0)
    ax2.set_xlim(0, om_max * 1.6)

    def _short(name, k=40):
        return name if len(name) <= k else name[:k - 1].rstrip() + "…"

    # Per-bar annotations: ClinVar bar (lower half-row) gets its count + primary
    # ClinVar condition (lavender); OMIM bar (upper half-row) gets its count +
    # primary OMIM phenotype (sage). "+N" flags further pathologies. Names sit
    # in a right-hand column and may spill into the inter-panel gap.
    x_name = cv_xlim * 0.78
    for i, gene in enumerate(nv["gene"].tolist()):
        cdx, odx = clinvar_dx.get(gene, []), omim_dx.get(gene, [])
        if cv_counts[i] > 0:
            ax.text(cv_counts[i] + cv_max * 0.02, y[i] - 0.205,
                    f"{int(cv_counts[i]):,}", va="center", ha="left",
                    fontsize=6.5, color=CV_TXT, weight="bold")
        if cdx:
            extra = f"  +{len(cdx) - 1}" if len(cdx) > 1 else ""
            ax.text(x_name, y[i] - 0.205, _short(sorted(cdx, key=len)[0]) + extra,
                    va="center", ha="left", fontsize=6.3, color=CV_TXT,
                    style="italic", clip_on=False)
        if om_counts[i] > 0:
            ax2.text(om_counts[i] + om_max * 0.04, y[i] + 0.205,
                     f"{int(om_counts[i])}", va="center", ha="left",
                     fontsize=6.5, color=OM_TXT, weight="bold")
        if odx:
            extra = f"  +{len(odx) - 1}" if len(odx) > 1 else ""
            ax.text(x_name, y[i] + 0.205, _short(sorted(odx, key=len)[0]) + extra,
                    va="center", ha="left", fontsize=6.3, color=OM_TXT,
                    style="italic", clip_on=False)
        if not cdx and not odx:
            ax.text(x_name, i, "—", va="center", ha="left", fontsize=6.5,
                    color="0.6")

    ax.set_yticks(y)
    ax.set_yticklabels(nv["gene"].tolist(), fontsize=10, weight="bold")
    ax.set_ylim(-0.6, len(nv) - 0.4)
    ax2.set_ylim(ax.get_ylim())
    # ClinVar axis (bottom, lavender) and OMIM axis (top, sage) colour-matched.
    ax.set_xlabel("# P/LP ClinVar variants", fontsize=9, color=CV_TXT)
    ax2.set_xlabel("# OMIM phenotypes", fontsize=9, color=OM_TXT, labelpad=4)
    ax.tick_params(axis="x", colors=CV_TXT, labelsize=8, length=2)
    ax2.tick_params(axis="x", colors=OM_TXT, labelsize=8, length=2)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis="y", length=2)
    for sp in ("right",):
        ax.spines[sp].set_visible(False)
        ax2.spines[sp].set_visible(False)
    ax.spines["top"].set_color(OM_TXT)
    ax.spines["bottom"].set_color(CV_TXT)
    ax.set_title("ClinVar + OMIM disease burden — top-10 novel KIRC genes",
                 pad=12, weight="bold", fontsize=12)
    ax.legend(handles=[Patch(facecolor=CV_FILL, label="ClinVar P/LP alleles"),
                       Patch(facecolor=OM_FILL, label="OMIM phenotypes")],
              loc="upper right", fontsize=7.5, framealpha=0.9)
    return ax


def _load_non_oncokb_top10(cohorts):
    """Reusable: per-cohort top-10 non-OncoKB (CGC=False) genes."""
    data = {}
    wide_path = Path("results/final_results/discovery_wide_table.csv")
    if not wide_path.exists():
        log.warning("discovery_wide_table.csv missing — figure will be empty")
        return data
    wide = pd.read_csv(wide_path)
    df_best = (wide.sort_values("p_res")
                   .drop_duplicates(subset=["cohort", "gene"], keep="first")
                   .reset_index(drop=True))
    for ds_key in cohorts:
        sub = df_best[(df_best["cohort"] == ds_key) & (~df_best["CGC"])].copy()
        if sub.empty:
            continue
        data[ds_key] = sub.sort_values("p_res").head(10).reset_index(drop=True)
    return data


def figure2_discovery(out_dir):
    """FIG 2 — DISCOVERY of non-OncoKB imaging-correlating genes.

    Layout:
      a (row1-left)    KIRC top-10 non-OncoKB partial-Spearman bars
      b (row1-right)   KEGG functional-group composition of FDR-sig hits
      c (row2-left)    LIHC top-10 non-OncoKB bars
      d (row2-right)   BRCA — typically empty (no FDR-sig non-OncoKB hits)
      e (row3-left)    top-10 novel KIRC genes × ClinVar P/LP allele counts
      f (row3-right)   cross-metric concordance for all Fig 1/2 genes
                       (absolute |r|, one colour per metric)

    Per-gene carrier-vs-non-carrier raw-data view moved to figS_raw_data.
    """
    cohorts = ["KIRC", "LIHC", "BRCA"]
    log.info("Computing Figure 2 (discovery, 2×2 layout)...")
    data = _load_non_oncokb_top10(cohorts)
    if not data:
        log.warning("No non-OncoKB data — skipping Fig 2")
        return

    cohort_titles = {"KIRC": "TCGA-KIRC (Kidney)",
                     "LIHC": "TCGA-LIHC (Liver)",
                     "BRCA": "TCGA-BRCA (Breast)"}
    n_gen_max = max(len(data[c]) for c in cohorts if c in data)
    cell_h = max(3.6, 0.36 * n_gen_max + 1.0)
    panel_e_h = 0.40 * 10 + 1.4          # height for top-10 ClinVar bars
    fig_w = 16.0
    fig_h = cell_h * 2 + panel_e_h + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    # 3 rows: top-2 = the 2×2 cohort/KEGG grid; row 3 = panels e (ClinVar)
    # and f (per-gene metric concordance, absolute |r|) side by side.
    gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.32,
                          width_ratios=[1, 1],
                          height_ratios=[cell_h, cell_h, panel_e_h],
                          left=0.05, right=0.985, top=0.97, bottom=0.07)

    cell_map = {"KIRC": gs[0, 0], "LIHC": gs[1, 0], "BRCA": gs[1, 1]}
    panel_label_map = {"KIRC": "a", "LIHC": "c", "BRCA": "d"}
    label_axes = {}
    for ds_key, gs_cell in cell_map.items():
        if ds_key not in data:
            ax_empty = fig.add_subplot(gs_cell)
            ax_empty.set_title(cohort_titles[ds_key], pad=6,
                                weight="bold", fontsize=12, color="0.55")
            ax_empty.text(0.5, 0.45,
                          "0 non-OncoKB FDR-sig hits",
                          ha="center", va="center", fontsize=10,
                          transform=ax_empty.transAxes, color="0.55",
                          style="italic")
            ax_empty.set_xticks([]); ax_empty.set_yticks([])
            for sp in ("top", "right", "left", "bottom"):
                ax_empty.spines[sp].set_visible(False)
            label_axes[panel_label_map[ds_key]] = ax_empty
            continue
        ax = _draw_cohort_bar_strip(fig, gs_cell, ds_key, data[ds_key],
                                     cohort_title=cohort_titles[ds_key])
        label_axes[panel_label_map[ds_key]] = ax

    # Panel b — KEGG functional-group composition of FDR-sig hits (KIRC).
    ax_b = _draw_kegg_groups_panel(fig, gs[0, 1], cohort="KIRC")
    label_axes["b"] = ax_b

    # Panel e — top-10 novel KIRC genes ranked by ClinVar pathogenic /
    # likely-pathogenic allele count, with primary disease annotation.
    ax_e = _draw_clinvar_novel_panel(fig, gs[2, 0], cohort="KIRC", top_n=10)
    label_axes["e"] = ax_e

    # Panel f — cross-metric concordance for ALL Fig 1/2 genes, ABSOLUTE |r|
    # only (one column per gene, one colour per metric). The sign-flip mirror
    # collapses, so the seven severity metrics overlap at one height (magnitude
    # agreement) and mutation count (n_mut) is the lone divergent metric.
    from analysis.gene_metric_concordance import (
        compute_concordance, draw_gene_axis_scatter)
    _, _conc_long, _conc_summ = compute_concordance()
    ax_f = fig.add_subplot(gs[2, 1])
    draw_gene_axis_scatter(ax_f, _conc_long, _conc_summ, show_title=False,
                           absolute=True, xlabels=True, cohort_labels=True,
                           point_s=16, label_fs=5.5, cohort_fs=9,
                           metric_labels=METRIC_DISPLAY, legend=True,
                           legend_loc="lower center", legend_anchor=(0.5, 1.01),
                           legend_ncol=4, legend_fs=8)
    # Title above the (2-row) colour legend; pad clears the legend block.
    ax_f.set_title("Cross-metric concordance (all genes)",
                   pad=40, weight="bold", fontsize=12)
    label_axes["f"] = ax_f

    # Nudge the whole bottom row (e + f) down so panel f's title + legend clear
    # the x-axis label of panel d directly above; moving both keeps e/f aligned.
    for _ax in (ax_e, ax_f):
        _p = _ax.get_position()
        _ax.set_position([_p.x0, _p.y0 - 0.012, _p.width, _p.height])

    for letter, ax in label_axes.items():
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes,
                fontsize=22, weight="bold", va="top", ha="left")

    legend_handles = [
        Patch(facecolor=PALETTE["fdr"],     label="FDR<0.05  (corrected)"),
        Patch(facecolor=PALETTE["nominal"], label="$p$<0.05  (nominal only)"),
        Patch(facecolor=PALETTE["ns"],      label="n.s."),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=11)

    save_dual(fig, out_dir, "fig2_discovery")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Mutational etiology of the novel imaging-correlating genes
# ───────────────────────────────────────────────────────────────────────────

# ── KEGG functional-group buckets used in Fig 3 (mirror analysis/test_kegg_groups.py)
# ───────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Wide gene-metric × imaging discovery (volcano + top hits)
# ───────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str,
                        default="results/final_results/paper")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure1_rediscovery(out_dir)
    log.info("Wrote Fig 1 (rediscovery) -> %s/fig1_rediscovery.{png,pdf}", out_dir)
    figure2_discovery(out_dir)
    log.info("Wrote Fig 2 (discovery)   -> %s/fig2_discovery.{png,pdf}", out_dir)


if __name__ == "__main__":
    main()
