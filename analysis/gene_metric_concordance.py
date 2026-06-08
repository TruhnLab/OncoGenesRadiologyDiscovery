#!/usr/bin/env python3
"""
Cross-metric concordance for the Fig 1 (rediscovery) and Fig 2 (discovery) genes.

Motivation (reviewer question):
  Figures 1 and 2 report each gene with a SINGLE best-performing per-gene Evo2
  severity metric (smallest TMB-residualized partial-Spearman p against one
  imaging feature). That raises the obvious question: where does each gene land
  in the OTHER severity metrics? Do the metrics point in the same direction, or
  is the headline association carried by a single metric? Single-metric hits
  (especially count-driven or dispersion-only) deserve explicit discussion.

What this does:
  For every gene shown in Fig 1 (top-10 OncoKB-known per cohort) and Fig 2
  (top-10 non-OncoKB per cohort), it pulls the partial-Spearman r of ALL eight
  per-gene metrics against THAT gene's headline imaging feature, straight from
  results/final_results/discovery_wide_table.csv (no recomputation — same
  numbers as the figures).

  The eight metrics split into three conceptual blocks:
    magnitude severity : max_abs_dll, mean_abs_dll, sum_abs_dll, std_abs_dll
    signed direction   : min_dll, max_dll, top1_signed
    burden             : n_mutations

  Sign convention: for a carrier with a single damaging variant (ΔLL<0),
  abs_dll = -ΔLL, so the three signed metrics are exact sign-flips of the
  magnitude metrics. To ask "do the severity scores point the same way" we
  therefore put everything on a common "severity ↑" axis by negating the signed
  metrics (severity_aligned_r). n_mutations is a count, not a severity score, so
  it is reported separately and never folded into the severity-concordance.

Outputs (results/final_results/exploration/):
  gene_metric_concordance_long.csv     one row per (gene, metric)
  gene_metric_concordance_summary.csv  one row per gene + classification
  gene_metric_concordance.txt          human-readable per-cohort report
  figS_gene_metric_concordance.{png,pdf}  two heatmaps (rediscovery / discovery)
  gene_metric_concordance_discussion.md   auto-drafted discussion paragraph
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

WIDE_PATH = Path("results/final_results/discovery_wide_table.csv")
OUT_DIR = Path("results/final_results/exploration")

MAGNITUDE = ["max_abs_dll", "mean_abs_dll", "sum_abs_dll", "std_abs_dll"]
SIGNED = ["min_dll", "max_dll", "top1_signed"]
COUNT = ["n_mutations"]
METRIC_ORDER = MAGNITUDE + SIGNED + COUNT
# Severity metrics = everything except the raw count. The signed block is
# flipped onto the "more severe → larger value" axis.
SEVERITY_METRICS = MAGNITUDE + SIGNED
SIGN_FLIP = set(SIGNED)              # negate r to align onto severity↑ axis
COHORTS = ["KIRC", "LIHC", "BRCA"]
P_SIG = 0.05                         # nominal p_res threshold for "stands out"

# Compact codes for imaging features (genes within a cohort may differ).
IMG_SHORT = {
    "kits_tumor_volume_ml": "Vtum", "tumor_kidney_ratio": "T/K",
    "kits_tumor_heterogeneity": "het", "kits_tumor_necrotic_frac": "necr",
    "lesion_volume_ml": "Vles", "lesion_liver_ratio": "L/Liv",
    "lesion_heterogeneity": "het", "lesion_necrotic_frac": "necr",
    "liver_hu_mean": "HUm", "liver_hu_std": "HUsd", "liver_volume_ml": "Vliv",
    "tumor_volume_ml": "Vtum", "tumor_breast_ratio": "T/B",
    "tumor_heterogeneity": "het", "tumor_intensity_mean": "Imean",
    "tumor_intensity_std": "Istd",
}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 / Fig 2 gene selection (must match analysis/publication_figures.py)
# ─────────────────────────────────────────────────────────────────────────────

def select_headline_genes(wide):
    """Reproduce the per-cohort top-10 gene tables behind Fig 1 and Fig 2.

    Best metric per (cohort, gene) = smallest p_res; then top-10 by p_res,
    split on OncoKB/CGC membership (Fig 1 = CGC, Fig 2 = non-CGC).
    Returns a DataFrame with one row per headline gene plus a 'figure' tag.
    """
    best = (wide.sort_values("p_res")
                .drop_duplicates(subset=["cohort", "gene"], keep="first")
                .reset_index(drop=True))
    out = []
    for fig_name, is_cgc in [("Fig1_rediscovery", True), ("Fig2_discovery", False)]:
        for ck in COHORTS:
            sub = best[(best.cohort == ck) & (best.CGC == is_cgc)]
            sub = sub.sort_values("p_res").head(10)
            for rank, (_, r) in enumerate(sub.iterrows(), 1):
                out.append({"figure": fig_name, "cohort": ck, "rank": rank,
                            "gene": r.gene, "headline_metric": r.metric,
                            "imaging": r.imaging, "headline_r": r.r_res,
                            "headline_p": r.p_res,
                            "headline_fdr_best": r.fdr_best_per_gene,
                            "CGC": r.CGC})
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-metric profile + concordance classification
# ─────────────────────────────────────────────────────────────────────────────

def metric_profile(wide, cohort, gene, imaging):
    """r_res / p_res / FDR for all 8 metrics of one gene vs its headline imaging."""
    sub = wide[(wide.cohort == cohort) & (wide.gene == gene)
               & (wide.imaging == imaging)].set_index("metric")
    prof = {}
    for m in METRIC_ORDER:
        if m in sub.index:
            row = sub.loc[m]
            prof[m] = {"r": float(row.r_res), "p": float(row.p_res),
                       "fdr": float(row.fdr_per_gene),
                       "aligned": -float(row.r_res) if m in SIGN_FLIP
                                  else float(row.r_res)}
        else:
            prof[m] = None
    return prof


def classify(prof, headline_metric):
    """Summarise robustness of a gene's headline association across metrics.

    Reference direction is the headline metric's own severity-aligned sign (the
    direction the figure reports); for count-headline genes the reference is the
    majority severity-aligned direction instead, so we can ask whether the
    per-mutation severity agrees with the count signal. Returns concordance
    fields + a single classification label.
    """
    avail = {m: v for m, v in prof.items() if v is not None}
    sev_avail = {m: v for m, v in avail.items() if m in SEVERITY_METRICS}

    n_sig = sum(v["p"] < P_SIG for v in avail.values())
    n_sig_sev = sum(v["p"] < P_SIG for v in sev_avail.values())
    n_mag = [m for m in MAGNITUDE if m in avail]
    n_sig_mag = sum(avail[m]["p"] < P_SIG for m in n_mag)
    central = [m for m in ("max_abs_dll", "mean_abs_dll", "sum_abs_dll") if m in avail]
    n_sig_central = sum(avail[m]["p"] < P_SIG for m in central)

    # severity reference direction: majority sign of the MAGNITUDE |ΔLL| block
    # (max/mean/sum/std) — these are the primary severity scores; the signed
    # metrics are sign-flip derivatives and so are excluded from the reference.
    # Prefer significant magnitude metrics; fall back to all available magnitude.
    def _majority_sign(items):
        if not items:
            return None
        s = np.sign(np.median([v["aligned"] for v in items]))
        return int(s) if s != 0 else 1
    mag_avail = [avail[m] for m in MAGNITUDE if m in avail]
    mag_sig = [v for v in mag_avail if v["p"] < P_SIG]
    sev_majority = _majority_sign(mag_sig) or _majority_sign(mag_avail)

    # reference = headline direction (severity headline) or severity majority (count)
    if headline_metric in COUNT or headline_metric not in avail:
        ref = sev_majority
    else:
        ref = int(np.sign(avail[headline_metric]["aligned"])) or 1

    # concordance: fraction of OTHER available / significant severity metrics
    # whose severity-aligned sign agrees with the reference direction.
    others = [m for m in sev_avail if m != headline_metric]
    others_sig = [m for m in others if sev_avail[m]["p"] < P_SIG]
    conc_all = (np.mean([np.sign(sev_avail[m]["aligned"]) == ref for m in others])
                if others and ref is not None else np.nan)
    conc_sig = (np.mean([np.sign(sev_avail[m]["aligned"]) == ref for m in others_sig])
                if others_sig and ref is not None else np.nan)

    # n_mutations behaviour relative to the severity direction
    nm = avail.get("n_mutations")
    nm_sig = nm is not None and nm["p"] < P_SIG
    nm_concordant = (nm is not None and sev_majority is not None
                     and int(np.sign(nm["aligned"])) == sev_majority)

    # ── classification ────────────────────────────────────────────────────
    if n_sig == 0:
        # headline metric itself does not reach p<0.05 (e.g. the BRCA panels,
        # which the figures show ranked by p with no FDR-significant hits).
        label = "n.s. (no metric p<0.05)"
    elif headline_metric == "n_mutations":
        if n_sig_sev == 0:
            label = "count-only"            # no per-mutation severity signal
        elif not nm_concordant:
            label = "count-driven (severity discordant)"
        else:
            label = "count + severity concordant"
    elif headline_metric == "std_abs_dll" and n_sig_central == 0:
        label = "dispersion-only (std)"
    elif n_sig_sev <= 1:
        # only the headline metric (or nothing else) reaches significance.
        label = "single-metric"
    elif not np.isnan(conc_sig) and conc_sig >= 0.5:
        # the MAJORITY of the other significant severity metrics agree in
        # severity-aligned direction with the headline metric.
        label = "severity-concordant (robust)"
    else:
        # ≥2 severity metrics significant but they disagree in direction.
        label = "mixed (divergent metrics)"

    # redundancy flag: single-mutation gene → magnitude metrics collapse
    single_mut = ("std_abs_dll" not in avail
                  and all(m in avail for m in ("max_abs_dll", "sum_abs_dll"))
                  and abs(avail["max_abs_dll"]["r"] - avail["sum_abs_dll"]["r"]) < 0.02)

    return {
        "n_metrics_available": len(avail),
        "n_sig": n_sig,
        "n_sig_severity": n_sig_sev,
        "n_sig_magnitude": n_sig_mag,
        "ref_sign": (None if ref is None else int(ref)),
        "concordance_other_severity": (np.nan if np.isnan(conc_all) else round(conc_all, 3)),
        "concordance_sig_severity": (np.nan if np.isnan(conc_sig) else round(conc_sig, 3)),
        "n_mutations_sig": bool(nm_sig),
        "n_mutations_concordant_with_severity": bool(nm_concordant),
        "single_mutation_collapse": bool(single_mut),
        "classification": label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_heatmaps(long_df, summary_df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "figure.dpi": 200})
    col_label = {"max_abs_dll": "max|ΔLL|", "mean_abs_dll": "mean|ΔLL|",
                 "sum_abs_dll": "sum|ΔLL|", "std_abs_dll": "std|ΔLL|",
                 "min_dll": "min ΔLL", "max_dll": "max ΔLL",
                 "top1_signed": "top1 ΔLL", "n_mutations": "n_mut"}

    figures = ["Fig1_rediscovery", "Fig2_discovery"]
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 12),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})

    for ax, fig_name in zip(axes, figures):
        sub = summary_df[summary_df.figure == fig_name].copy()
        sub = sub.sort_values(["cohort", "rank"]).reset_index(drop=True)
        rows = [f"{r.gene}  ({r.cohort})" for _, r in sub.iterrows()]
        M = np.full((len(sub), len(METRIC_ORDER)), np.nan)
        Pn = np.zeros_like(M)
        for i, (_, g) in enumerate(sub.iterrows()):
            for j, m in enumerate(METRIC_ORDER):
                rec = long_df[(long_df.figure == fig_name)
                              & (long_df.cohort == g.cohort)
                              & (long_df.gene == g.gene)
                              & (long_df.metric == m)]
                if len(rec):
                    M[i, j] = rec.r_res.iloc[0]
                    Pn[i, j] = rec.p_res.iloc[0]
        im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r",
                       vmin=-0.5, vmax=0.5, aspect="auto")
        im.cmap.set_bad("0.85")
        for i in range(len(sub)):
            for j, m in enumerate(METRIC_ORDER):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "·", ha="center", va="center", color="0.5")
                    continue
                star = "*" if Pn[i, j] < P_SIG else ""
                ax.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center",
                        fontsize=6.2,
                        color="white" if abs(M[i, j]) > 0.3 else "black")
        # box the headline metric cell
        for i, (_, g) in enumerate(sub.iterrows()):
            j = METRIC_ORDER.index(g.headline_metric)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="black", lw=1.8))
        # block separators (magnitude | signed | count)
        for x in (len(MAGNITUDE) - 0.5, len(MAGNITUDE) + len(SIGNED) - 0.5):
            ax.axvline(x, color="0.3", lw=1.2)
        # cohort separators
        prev = None
        for i, (_, g) in enumerate(sub.iterrows()):
            if prev is not None and g.cohort != prev:
                ax.axhline(i - 0.5, color="0.3", lw=1.2)
            prev = g.cohort
        ax.set_xticks(range(len(METRIC_ORDER)))
        ax.set_xticklabels([col_label[m] for m in METRIC_ORDER], rotation=45,
                           ha="right")
        ax.set_yticks(range(len(sub)))
        # append classification tag to each row label
        ylabels = []
        tag = {"severity-concordant (robust)": "robust",
               "count-driven (severity discordant)": "count-driven",
               "count-only": "count-only",
               "count + severity concordant": "count+sev",
               "dispersion-only (std)": "std-only",
               "single-metric": "single",
               "mixed (divergent metrics)": "divergent",
               "n.s. (no metric p<0.05)": "n.s."}
        for _, g in sub.iterrows():
            ylabels.append(f"{g.gene} ({g.cohort}·{IMG_SHORT.get(g.imaging, '?')}) "
                           f"· {tag.get(g.classification, '')}")
        ax.set_yticklabels(ylabels, fontsize=6.5)
        ax.set_title(f"{fig_name.replace('_', ' — ')}\n"
                     f"(□ headline metric · * p<0.05)", fontsize=9)

    cbar = fig.colorbar(im, ax=axes, shrink=0.4, pad=0.02)
    cbar.set_label("TMB-residualized partial Spearman r\n"
                   "(vs each gene's headline imaging feature)")
    fig.suptitle("Cross-metric concordance for Fig 1 / Fig 2 genes — "
                 "magnitude |ΔLL|  |  signed ΔLL  |  count\n"
                 "(signed metrics are sign-flips of |ΔLL| for single-variant carriers; "
                 "row tag = robustness class)",
                 fontsize=9.5, y=0.998)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figS_gene_metric_concordance.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def make_concordance_scatter(long_df, out_dir):
    """One-glance summary: per-mutation severity (mean |ΔLL|) vs mutation count.

    Each gene is one point: x = mean_abs_dll partial r, y = n_mutations partial r,
    both vs the gene's headline imaging feature. Same-sign (positive-diagonal)
    quadrants = the severity and the count agree; off-diagonal quadrants =
    count-driven discordance. Marker filled if BOTH metrics are nominally
    significant. Discordant genes with ≥1 significant metric are labelled.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    pts = []
    for (fig, co, g), grp in long_df.groupby(["figure", "cohort", "gene"]):
        sev = grp[grp.metric == "mean_abs_dll"]
        cnt = grp[grp.metric == "n_mutations"]
        if sev.empty or cnt.empty or pd.isna(sev.r_res.iloc[0]) or pd.isna(cnt.r_res.iloc[0]):
            continue
        pts.append({"gene": g, "cohort": co,
                    "sev": sev.r_res.iloc[0], "p_sev": sev.p_res.iloc[0],
                    "cnt": cnt.r_res.iloc[0], "p_cnt": cnt.p_res.iloc[0]})
    d = pd.DataFrame(pts)
    d["concordant"] = np.sign(d.sev) == np.sign(d.cnt)
    d["both_sig"] = (d.p_sev < P_SIG) & (d.p_cnt < P_SIG)
    d["any_sig"] = (d.p_sev < P_SIG) | (d.p_cnt < P_SIG)

    rho, prho = stats.spearmanr(d.sev, d.cnt)
    n = len(d)
    n_conc = int(d.concordant.sum())
    n_conc_sig = int((d.concordant & d.both_sig).sum())
    n_bothsig = int(d.both_sig.sum())

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    lim = 0.55
    # quadrant shading: concordant (same sign) green, discordant red
    ax.axhspan(0, lim, xmin=0.5, xmax=1.0, color="#2a9d8f", alpha=0.06)
    ax.axhspan(-lim, 0, xmin=0.0, xmax=0.5, color="#2a9d8f", alpha=0.06)
    ax.axhspan(0, lim, xmin=0.0, xmax=0.5, color="#e76f51", alpha=0.06)
    ax.axhspan(-lim, 0, xmin=0.5, xmax=1.0, color="#e76f51", alpha=0.06)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.4", lw=0.8)

    for _, r in d.iterrows():
        col = "#2a9d8f" if r.concordant else "#e76f51"
        ax.scatter(r.sev, r.cnt, s=55,
                   facecolor=col if r.both_sig else "white",
                   edgecolor=col, linewidth=1.6, zorder=3)
        # label discordant genes that reach significance in at least one metric
        if (not r.concordant) and r.any_sig:
            ax.annotate(f"{r.gene}", (r.sev, r.cnt), fontsize=7,
                        xytext=(4, 3), textcoords="offset points",
                        color="#b3361f")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("per-mutation severity:  mean |ΔLL|  partial-Spearman r")
    ax.set_ylabel("mutation burden:  n_mutations  partial-Spearman r")
    ax.set_title("Severity vs count concordance across all Fig 1/2 genes\n"
                 "(r vs each gene's headline imaging feature)", fontsize=10)
    # quadrant captions (placed in the correct quadrants)
    ax.text(0.97, 0.97, "concordant\n(same sign)", transform=ax.transAxes,
            ha="right", va="top", color="#1d6f63", fontsize=8, style="italic")
    ax.text(0.03, 0.62, "discordant\n(severity ↓, count ↑)", transform=ax.transAxes,
            ha="left", va="top", color="#b3361f", fontsize=8, style="italic")
    ax.text(0.97, 0.06, "discordant\n(severity ↑, count ↓)", transform=ax.transAxes,
            ha="right", va="bottom", color="#b3361f", fontsize=8, style="italic")
    # legend (upper left) + stats box (lower left, in the clear corner)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", ls="", mfc="#555", mec="#555", ms=8,
               label="both p<0.05"),
        Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=8,
               label="not both sig."),
        Line2D([0], [0], marker="o", ls="", mfc="#2a9d8f", mec="#2a9d8f", ms=8,
               label="same sign"),
        Line2D([0], [0], marker="o", ls="", mfc="#e76f51", mec="#e76f51", ms=8,
               label="opposite sign"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.text(0.03, 0.03,
            f"sign-concordant: {n_conc}/{n} ({n_conc/n:.0%})\n"
            f"  of both-sig: {n_conc_sig}/{n_bothsig}\n"
            f"Spearman(sev, count) = {rho:+.2f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figS_severity_vs_count_scatter.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    log.info("severity-vs-count: %d/%d sign-concordant, rho=%.2f", n_conc, n, rho)


def make_metric_profiles(long_df, summary_df, out_dir):
    """Heatmap → scatter/parallel-coordinates view.

    x = the 8 metrics (magnitude | signed | count); y = partial-Spearman r vs
    each gene's headline imaging feature; one colour per gene. Points are joined
    per gene so a flat profile reads as cross-metric agreement and a zig-zag as
    divergence (the magnitude vs signed blocks mirror because the signed metrics
    are sign-flips for single-variant carriers). Star = the headline metric;
    filled marker = nominal p<0.05, hollow = n.s. Faceted figure × cohort so the
    per-panel gene count stays ≤10 and colours remain distinguishable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "figure.dpi": 200})
    short = {"max_abs_dll": "max|ΔLL|", "mean_abs_dll": "mean|ΔLL|",
             "sum_abs_dll": "sum|ΔLL|", "std_abs_dll": "std|ΔLL|",
             "min_dll": "min ΔLL", "max_dll": "max ΔLL",
             "top1_signed": "top1 ΔLL", "n_mutations": "n_mut"}
    figs = ["Fig1_rediscovery", "Fig2_discovery"]
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2), sharey=True)
    fig.subplots_adjust(wspace=0.45)
    for ri, fg in enumerate(figs):
        for ci, co in enumerate(COHORTS):
            ax = axes[ri][ci]
            genes = summary_df[(summary_df.figure == fg)
                               & (summary_df.cohort == co)].sort_values("rank")
            handles = []
            for gi, (_, grow) in enumerate(genes.iterrows()):
                sub = long_df[(long_df.figure == fg) & (long_df.cohort == co)
                              & (long_df.gene == grow.gene)].set_index("metric")
                color = cmap(gi % 10)
                xs, ys = [], []
                for j, m in enumerate(METRIC_ORDER):
                    if m in sub.index and pd.notna(sub.loc[m, "r_res"]):
                        xs.append(j); ys.append(sub.loc[m, "r_res"])
                ax.plot(xs, ys, "-", color=color, alpha=0.4, lw=1.0, zorder=1)
                for j, m in enumerate(METRIC_ORDER):
                    if m not in sub.index or pd.isna(sub.loc[m, "r_res"]):
                        continue
                    sig = sub.loc[m, "p_res"] < P_SIG
                    is_h = (m == grow.headline_metric)
                    ax.scatter(j, sub.loc[m, "r_res"],
                               s=110 if is_h else 34,
                               marker="*" if is_h else "o",
                               facecolor=color if sig else "white",
                               edgecolor=color, linewidth=1.2, zorder=3)
                handles.append(Line2D([0], [0], marker="o", ls="", mfc=color,
                                      mec=color, ms=6, label=grow.gene))
            ax.axhline(0, color="0.4", lw=0.8)
            for x in (len(MAGNITUDE) - 0.5, len(MAGNITUDE) + len(SIGNED) - 0.5):
                ax.axvline(x, color="0.7", lw=1, ls="--")
            ax.set_xticks(range(len(METRIC_ORDER)))
            ax.set_xticklabels([short[m] for m in METRIC_ORDER], rotation=45,
                               ha="right", fontsize=7)
            ax.set_title(f"{co} — {fg.split('_')[1]}", fontsize=9, weight="bold")
            ax.set_ylim(-0.55, 0.55)
            if ci == 0:
                ax.set_ylabel("partial-Spearman r\n(vs headline imaging)",
                              fontsize=8)
            ax.legend(handles=handles, fontsize=6.5, ncol=1, loc="center left",
                      bbox_to_anchor=(1.0, 0.5), framealpha=0.9,
                      handletextpad=0.2, borderpad=0.3)
    fig.suptitle("Per-gene correlation profile across the 8 metrics "
                 "(★ = headline metric · filled = p<0.05 · "
                 "magnitude |ΔLL| │ signed ΔLL │ count)\n"
                 "flat profile = metrics agree; magnitude vs signed blocks mirror "
                 "(sign-flip for single-variant carriers); n_mut is a count, not a "
                 "severity score", fontsize=9.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figS_metric_profiles.{ext}", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figS_metric_profiles.{png,pdf}")


def make_metric_profiles_collapsed(long_df, summary_df, out_dir):
    """All 59 genes in ONE panel: metric on x, severity-ORIENTED r on y.

    Two transforms make a single readable panel:
      1. signed metrics (min/max ΔLL, top-1) are flipped onto the common
         "severity ↑" axis (severity_aligned_r);
      2. each gene is then oriented so its own |ΔLL| severity direction is
         positive (multiply the whole profile by the sign of the gene's
         magnitude-block median).
    After this, a cross-metric-concordant gene sits in the upper half across all
    seven severity metrics; the bold mean line stays flat and positive there but
    visibly drops at n_mutations, pulled down by the count-driven genes whose
    count points opposite to their severity. Colour = cohort, style = figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                         "figure.dpi": 200})
    short = {"max_abs_dll": "max|ΔLL|", "mean_abs_dll": "mean|ΔLL|",
             "sum_abs_dll": "sum|ΔLL|", "std_abs_dll": "std|ΔLL|",
             "min_dll": "min ΔLL", "max_dll": "max ΔLL",
             "top1_signed": "top1 ΔLL", "n_mutations": "n_mut"}
    cohort_col = {"KIRC": "#1f77b4", "LIHC": "#d62728", "BRCA": "#2ca02c"}
    style = {"Fig1_rediscovery": "-", "Fig2_discovery": "--"}

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    oriented = {m: [] for m in METRIC_ORDER}      # collect for mean/IQR overlay
    for _, g in summary_df.iterrows():
        sub = long_df[(long_df.figure == g.figure) & (long_df.cohort == g.cohort)
                      & (long_df.gene == g.gene)].set_index("metric")
        mags = [sub.loc[m, "severity_aligned_r"] for m in MAGNITUDE
                if m in sub.index and pd.notna(sub.loc[m, "severity_aligned_r"])]
        s = np.sign(np.median(mags)) if mags else 1.0
        s = s if s != 0 else 1.0
        xs, ys = [], []
        for j, m in enumerate(METRIC_ORDER):
            if m in sub.index and pd.notna(sub.loc[m, "severity_aligned_r"]):
                val = s * sub.loc[m, "severity_aligned_r"]
                xs.append(j); ys.append(val)
                oriented[m].append(val)
        ax.plot(xs, ys, style[g.figure], color=cohort_col[g.cohort],
                alpha=0.40, lw=1.0, marker="o", ms=2.5, zorder=2)

    # bold mean ± IQR band of the oriented profiles
    xs_full = list(range(len(METRIC_ORDER)))
    mean = [np.mean(oriented[m]) if oriented[m] else np.nan for m in METRIC_ORDER]
    q1 = [np.percentile(oriented[m], 25) if oriented[m] else np.nan for m in METRIC_ORDER]
    q3 = [np.percentile(oriented[m], 75) if oriented[m] else np.nan for m in METRIC_ORDER]
    ax.fill_between(xs_full, q1, q3, color="0.5", alpha=0.25, zorder=4)
    ax.plot(xs_full, mean, color="black", lw=2.6, marker="s", ms=6, zorder=6,
            label="mean across genes (±IQR)")

    ax.axhline(0, color="0.4", lw=0.8)
    for x in (len(MAGNITUDE) - 0.5, len(MAGNITUDE) + len(SIGNED) - 0.5):
        ax.axvline(x, color="0.6", lw=1, ls="--")
    ax.text(1.5, 0.55, "magnitude |ΔLL|", ha="center", fontsize=8, color="0.3")
    ax.text(5.0, 0.55, "signed ΔLL (flipped)", ha="center", fontsize=8, color="0.3")
    ax.text(7.0, 0.55, "count", ha="center", fontsize=8, color="0.3")
    ax.set_xticks(range(len(METRIC_ORDER)))
    ax.set_xticklabels([short[m] for m in METRIC_ORDER], rotation=40, ha="right")
    ax.set_ylim(-0.55, 0.60)
    ax.set_ylabel("severity-oriented partial-Spearman r\n"
                  "(each gene flipped so its |ΔLL| severity is positive)")
    ax.set_title("Per-gene metric profiles — all 59 Fig 1/2 genes, three cohorts in one panel\n"
                 "upper-half & flat across the 7 severity metrics = concordant; "
                 "mean drops at n_mut = count-driven genes", fontsize=9.5)
    handles = [Line2D([0], [0], color=cohort_col[c], lw=2, label=c) for c in COHORTS]
    handles += [Line2D([0], [0], color="0.3", lw=1.2, ls="-", label="rediscovery (Fig 1)"),
                Line2D([0], [0], color="0.3", lw=1.2, ls="--", label="discovery (Fig 2)"),
                Line2D([0], [0], color="black", lw=2.6, marker="s", ms=6,
                       label="mean across genes (±IQR)")]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figS_metric_profiles_collapsed.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figS_metric_profiles_collapsed.{png,pdf}")


# Colour per metric: blues = magnitude |ΔLL|, oranges = signed ΔLL, black = count.
GENE_METRIC_COL = {
    "max_abs_dll": "#08519c", "mean_abs_dll": "#3182bd",
    "sum_abs_dll": "#6baed6", "std_abs_dll": "#9ecae1",
    "min_dll": "#a63603", "max_dll": "#e6550d", "top1_signed": "#fd8d3c",
    "n_mutations": "#000000",
}
GENE_METRIC_LBL = {"max_abs_dll": "max|ΔLL|", "mean_abs_dll": "mean|ΔLL|",
                   "sum_abs_dll": "sum|ΔLL|", "std_abs_dll": "std|ΔLL|",
                   "min_dll": "min ΔLL", "max_dll": "max ΔLL",
                   "top1_signed": "top1 ΔLL", "n_mutations": "n_mut"}


def draw_gene_axis_scatter(ax, long_df, summary_df, *, show_title=True,
                           ylabel=True, absolute=False, xlabels=True,
                           cohort_labels=True, label_fs=5.6, cohort_fs=11,
                           point_s=26, metric_labels=None, legend=True,
                           legend_loc="lower center", legend_anchor=(0.5, -0.30),
                           legend_ncol=8, legend_fs=8):
    """Draw the per-gene metric-correlation scatter onto a given axes.

    x = all Fig 1/2 genes (one column each, grouped by cohort then figure),
    y = TMB-residualized partial-Spearman r vs the gene's headline imaging
    feature, one colour per metric (blues = magnitude |ΔLL|, oranges = signed
    ΔLL, black = n_mut). Filled = nominal p<0.05, hollow = n.s. A concordant
    gene shows the blues stacked together (oranges mirror, being sign-flips);
    count-driven genes are where the black n_mut point sits opposite the blues.

    absolute=True plots |r| instead of signed r: the sign-flip mirror collapses,
    so the seven severity metrics overlap at one height (magnitude agreement)
    and only n_mut can sit apart. Reused by the standalone supplementary figure
    and by Fig 2 panel f.
    """
    from matplotlib.lines import Line2D

    fig_order = {"Fig1_rediscovery": 0, "Fig2_discovery": 1}
    co_order = {c: i for i, c in enumerate(COHORTS)}
    rows = summary_df.copy()
    rows["_co"] = rows.cohort.map(co_order)
    rows["_fg"] = rows.figure.map(fig_order)
    rows = rows.sort_values(["_co", "_fg", "rank"]).reset_index(drop=True)
    n = len(rows)
    y_top = 0.55 if absolute else 0.62
    lbl_y = (0.50 if absolute else 0.57)

    for xi, (_, g) in enumerate(rows.iterrows()):
        sub = long_df[(long_df.figure == g.figure) & (long_df.cohort == g.cohort)
                      & (long_df.gene == g.gene)].set_index("metric")
        for m in METRIC_ORDER:
            if m not in sub.index or pd.isna(sub.loc[m, "r_res"]):
                continue
            sig = sub.loc[m, "p_res"] < P_SIG
            yval = abs(sub.loc[m, "r_res"]) if absolute else sub.loc[m, "r_res"]
            ax.scatter(xi, yval, s=point_s, zorder=3,
                       facecolor=GENE_METRIC_COL[m] if sig else "none",
                       edgecolor=GENE_METRIC_COL[m], linewidth=1.0, alpha=0.9)

    if not absolute:
        ax.axhline(0, color="0.4", lw=0.9)
    bounds = {}
    for ck in COHORTS:
        idx = rows.index[rows.cohort == ck]
        if len(idx):
            bounds[ck] = (idx.min(), idx.max())
    for ck, (lo, hi) in bounds.items():
        if lo > 0:
            ax.axvline(lo - 0.5, color="0.2", lw=1.6)
        if cohort_labels:
            ax.text((lo + hi) / 2, lbl_y, ck, ha="center", va="bottom",
                    fontsize=cohort_fs, weight="bold")
    for ck, (lo, hi) in bounds.items():
        f2 = rows.index[(rows.cohort == ck) & (rows.figure == "Fig2_discovery")]
        if len(f2):
            ax.axvline(f2.min() - 0.5, color="0.7", lw=0.8, ls=":")

    ax.set_xticks(range(n))
    if xlabels:
        ax.set_xticklabels(rows.gene, rotation=90, fontsize=label_fs)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim((0, y_top) if absolute else (-0.55, y_top))
    if ylabel:
        ax.set_ylabel("|partial r|" if absolute
                      else "partial-Spearman r\n(vs headline imaging)")
    if show_title:
        sub = ("|r|: severity metrics overlap (magnitude agreement) · "
               "n_mut the divergent metric" if absolute else
               "blues = magnitude |ΔLL| (stack = concordant) · "
               "oranges = signed ΔLL (mirror) · black = n_mut")
        ax.set_title("Per-gene metric correlations — all 59 Fig 1/2 genes "
                     "(one column per gene, one colour per metric)\n"
                     f"{sub} · filled = p<0.05 · dotted line = "
                     "rediscovery|discovery split", fontsize=9)
    if legend:
        lbls = metric_labels or GENE_METRIC_LBL
        handles = [Line2D([0], [0], marker="o", ls="", mfc=GENE_METRIC_COL[m],
                          mec=GENE_METRIC_COL[m], ms=7,
                          label=lbls.get(m, GENE_METRIC_LBL[m]))
                   for m in METRIC_ORDER]
        ax.legend(handles=handles, fontsize=legend_fs, loc=legend_loc,
                  ncol=legend_ncol, framealpha=0.9, columnspacing=1.0,
                  handletextpad=0.2, bbox_to_anchor=legend_anchor)
    return ax


def make_gene_axis_scatter(long_df, summary_df, out_dir):
    """Standalone supplementary version: a single strip of absolute |r| per
    gene (matches Fig 2 panel f)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                         "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(18, 6.0))
    draw_gene_axis_scatter(ax, long_df, summary_df, show_title=True,
                           absolute=True, xlabels=True, cohort_labels=True,
                           legend=True, legend_loc="lower center",
                           legend_anchor=(0.5, -0.32), legend_ncol=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figS_gene_axis_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figS_gene_axis_scatter.{png,pdf}")


# ─────────────────────────────────────────────────────────────────────────────
# Report + discussion text
# ─────────────────────────────────────────────────────────────────────────────

def write_report(headline, long_df, summary_df, out_dir):
    lines = ["=" * 80,
             "CROSS-METRIC CONCORDANCE — Fig 1 (rediscovery) & Fig 2 (discovery)",
             "=" * 80,
             "Each gene is reported in the figures via ONE best metric (boxed).",
             "Here we show all 8 metrics vs that gene's headline imaging feature.",
             "  magnitude severity : max/mean/sum/std |ΔLL|",
             "  signed direction   : min/max ΔLL, top1 ΔLL  (sign-flip of |ΔLL|",
             "                       for single-variant carriers)",
             "  burden             : n_mutations (count, not severity)",
             "* = nominal p_res < 0.05.  '·' = metric undefined for that gene.",
             ""]
    for fig_name in ("Fig1_rediscovery", "Fig2_discovery"):
        lines.append("\n" + "#" * 80)
        lines.append(f"# {fig_name}")
        lines.append("#" * 80)
        for ck in COHORTS:
            sub = summary_df[(summary_df.figure == fig_name)
                             & (summary_df.cohort == ck)].sort_values("rank")
            if sub.empty:
                continue
            imgs = ", ".join(f"{IMG_SHORT.get(i, i)}={i}"
                             for i in sorted(sub.imaging.unique()))
            lines.append(f"\n--- {ck} --- (each gene vs its own headline "
                         f"imaging; codes: {imgs})")
            hdr = (f"  {'gene':9s} {'img':5s} {'headline':12s} "
                   + " ".join(f"{m.replace('_abs_dll','|').replace('_dll','').replace('top1_signed','top1')[:7]:>7s}"
                             for m in METRIC_ORDER)
                   + f"  {'classification':28s}")
            lines.append(hdr)
            for _, g in sub.iterrows():
                cells = []
                for m in METRIC_ORDER:
                    rec = long_df[(long_df.figure == fig_name)
                                  & (long_df.cohort == ck)
                                  & (long_df.gene == g.gene)
                                  & (long_df.metric == m)]
                    if len(rec):
                        s = "*" if rec.p_res.iloc[0] < P_SIG else " "
                        cells.append(f"{rec.r_res.iloc[0]:>+6.2f}{s}")
                    else:
                        cells.append(f"{'·':>7s}")
                conc = ("  —" if pd.isna(g.concordance_sig_severity)
                        else f"{g.concordance_sig_severity:.2f}")
                lines.append(f"  {g.gene:9s} {IMG_SHORT.get(g.imaging, g.imaging):5s} "
                             f"{g.headline_metric:12s} "
                             + " ".join(cells)
                             + f"  sig={int(g.n_sig)}/{int(g.n_metrics_available)} "
                             + f"conc={conc}  {g.classification:28s}")

    # tallies
    lines.append("\n" + "=" * 80)
    lines.append("CLASSIFICATION TALLY")
    lines.append("=" * 80)
    for fig_name in ("Fig1_rediscovery", "Fig2_discovery"):
        lines.append(f"\n{fig_name}:")
        vc = summary_df[summary_df.figure == fig_name]["classification"].value_counts()
        for k, v in vc.items():
            lines.append(f"  {v:2d}  {k}")
    (out_dir / "gene_metric_concordance.txt").write_text("\n".join(lines))
    log.info("wrote %s", out_dir / "gene_metric_concordance.txt")


def write_discussion(summary_df, out_dir):
    def names(fig, label):
        s = summary_df[(summary_df.figure == fig)
                       & (summary_df.classification == label)]
        return [f"{r.gene} ({r.cohort})" for _, r in s.iterrows()]

    robust1 = names("Fig1_rediscovery", "severity-concordant (robust)")
    robust2 = names("Fig2_discovery", "severity-concordant (robust)")
    cnt1 = (names("Fig1_rediscovery", "count-driven (severity discordant)")
            + names("Fig1_rediscovery", "count-only"))
    cnt2 = (names("Fig2_discovery", "count-driven (severity discordant)")
            + names("Fig2_discovery", "count-only"))
    std1 = names("Fig1_rediscovery", "dispersion-only (std)")
    std2 = names("Fig2_discovery", "dispersion-only (std)")
    single = (names("Fig1_rediscovery", "single-metric")
              + names("Fig2_discovery", "single-metric")
              + names("Fig1_rediscovery", "mixed (divergent metrics)")
              + names("Fig2_discovery", "mixed (divergent metrics)"))
    ns = (names("Fig1_rediscovery", "n.s. (no metric p<0.05)")
          + names("Fig2_discovery", "n.s. (no metric p<0.05)"))

    n_tot = len(summary_df)
    n_robust = len(robust1) + len(robust2)
    n_cnt = len(cnt1) + len(cnt2)

    md = f"""# Cross-metric concordance of the Fig 1 / Fig 2 gene hits

Each gene in Figures 1 and 2 is reported through a single best-performing
per-gene Evo2 metric. To gauge how much weight that single metric carries, we
re-examined every reported gene across all eight per-gene metrics, correlated
against that gene's headline imaging feature (TMB-residualized partial
Spearman; numbers taken directly from `discovery_wide_table.csv`). Because the
three signed metrics (min/max ΔLL, top-1 ΔLL) are exact sign-flips of the
magnitude |ΔLL| metrics for carriers with a single variant in the gene, we
placed all severity metrics on a common "more-severe → larger-value" axis
before assessing directional agreement; `n_mutations` was kept separate as a
mutation-count (burden) measure rather than a severity score.

**Most reported associations are directionally concordant across the severity
metrics** ({n_robust}/{n_tot} genes), i.e. the magnitude and (sign-aligned)
signed metrics agree and at least one corroborates the headline metric beyond
the single best one. In the rediscovery set these include
{", ".join(robust1[:8])}{" …" if len(robust1) > 8 else ""}; in the discovery
set, {", ".join(robust2[:8])}{" …" if len(robust2) > 8 else ""}. For these
genes the choice of metric is largely cosmetic — the radiogenomic association
is a property of the gene, not of the summary statistic.

**Three caveats warrant explicit discussion.** First, a subset of hits are
*count-driven*: their headline metric is `n_mutations` and the per-mutation
severity metrics are either non-significant or point the opposite way
({n_cnt} genes; rediscovery: {", ".join(cnt1) or "none"}; discovery:
{", ".join(cnt2) or "none"}). For these genes the imaging correlation reflects
recurrent mutation of the gene rather than the predicted functional impact of
those mutations, and should be interpreted as a burden signal. Second, a few
hits are *dispersion-only*, reaching significance solely through the
within-gene spread of severity (`std_abs_dll`) while the central-tendency
metrics are flat (rediscovery: {", ".join(std1) or "none"}; discovery:
{", ".join(std2) or "none"}); these are consistent with a
"one-severe-hit-among-passengers" pattern but rest on a single, noisier
statistic. Third, the remaining single-metric or directionally-divergent hits
({", ".join(single) or "none"}) either reach nominal significance in only one
severity metric, or have ≥2 significant metrics that disagree in direction, and
should be regarded as the least robust of the reported associations.

Separately, the genes shown in the BRCA panels
({", ".join(ns) or "none"}) do not reach nominal significance in *any* per-gene
metric; they appear only because the panels are ranked by uncorrected $p$ and,
as already noted in Results, no BRCA gene survives cohort-level FDR. They should
not be interpreted as concordant or discordant hits.

We note finally that for single-variant genes the eight metrics collapse to two
independent quantities (severity magnitude and mutation count), so apparent
agreement among the |ΔLL| metrics for such genes is not independent
corroboration; the signed metrics (min/max ΔLL, top-1 ΔLL) are exact sign-flips
of the magnitude metrics in that regime.
"""
    (out_dir / "gene_metric_concordance_discussion.md").write_text(md)
    log.info("wrote %s", out_dir / "gene_metric_concordance_discussion.md")


# ─────────────────────────────────────────────────────────────────────────────
def compute_concordance(wide_path=WIDE_PATH):
    """Build the concordance tables for the Fig 1/2 genes (no file output).

    Returns (headline, long_df, summary_df). Reused by main() and by
    publication_figures.py for Fig 2 panel f.
    """
    wide = pd.read_csv(wide_path)
    headline = select_headline_genes(wide)
    long_rows, summ_rows = [], []
    for _, g in headline.iterrows():
        prof = metric_profile(wide, g.cohort, g.gene, g.imaging)
        for m in METRIC_ORDER:
            v = prof[m]
            long_rows.append({
                "figure": g.figure, "cohort": g.cohort, "gene": g.gene,
                "imaging": g.imaging, "metric": m,
                "is_headline": (m == g.headline_metric),
                "r_res": None if v is None else v["r"],
                "p_res": None if v is None else v["p"],
                "fdr_per_gene": None if v is None else v["fdr"],
                "severity_aligned_r": None if v is None else v["aligned"],
            })
        c = classify(prof, g.headline_metric)
        summ_rows.append({**g.to_dict(), **c})
    return headline, pd.DataFrame(long_rows), pd.DataFrame(summ_rows)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if not WIDE_PATH.exists():
        raise FileNotFoundError(f"{WIDE_PATH} not found — run the discovery sweep first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headline, long_df, summary_df = compute_concordance(WIDE_PATH)
    log.info("headline genes: %d (Fig1=%d, Fig2=%d)", len(headline),
             (headline.figure == "Fig1_rediscovery").sum(),
             (headline.figure == "Fig2_discovery").sum())
    long_df.to_csv(OUT_DIR / "gene_metric_concordance_long.csv", index=False)
    summary_df.to_csv(OUT_DIR / "gene_metric_concordance_summary.csv", index=False)
    log.info("wrote long (%d rows) and summary (%d genes)",
             len(long_df), len(summary_df))

    write_report(headline, long_df, summary_df, OUT_DIR)
    write_discussion(summary_df, OUT_DIR)
    make_heatmaps(long_df, summary_df, OUT_DIR)
    make_concordance_scatter(long_df, OUT_DIR)
    make_metric_profiles(long_df, summary_df, OUT_DIR)
    make_metric_profiles_collapsed(long_df, summary_df, OUT_DIR)
    make_gene_axis_scatter(long_df, summary_df, OUT_DIR)

    # console digest
    print("\nClassification tally:")
    print(summary_df.groupby(["figure", "classification"]).size().to_string())


if __name__ == "__main__":
    main()
