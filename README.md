# OncoGenesRadiologyDiscovery

Associations between **somatic mutation severity** (scored zero-shot by the Evo2
genomic language model, |ΔLL|) and **tumor radiomic phenotype** across three TCGA
cohorts — cRCC (clear cell renal cell carcinoma, TCGA-KIRC), HCC (hepatocellular
carcinoma, TCGA-LIHC), and BC (breast cancer, TCGA-BRCA).

The code produces the two main-text figures from a single TMB-residualized
per-gene-metric × imaging correlation table (`discovery_wide_table.csv`):

- **Fig 1 — Rediscovery:** top OncoKB/CGC genes whose Evo2 severity correlates with imaging.
- **Fig 2 — Discovery:** novel imaging-correlating genes + KEGG functional groups + ClinVar/OMIM disease burden + cross-metric concordance.

- **Source repository:** <https://github.com/TruhnLab/OncoGenesRadiologyDiscovery>
- **License:** MIT (see [`LICENSE`](LICENSE))
- **Detailed description of code functionality / pseudocode:** see the **Methods**
  section of the manuscript.

---

## 1. System requirements

**Operating systems (tested)**
- Linux (Rocky Linux 9 / kernel 5.14). The figure step is pure Python and is also
  expected to run on macOS and Windows.

**Software dependencies**
- Python **3.10** (tested on 3.10.18).
- Python packages, pinned in [`requirements.txt`](requirements.txt) — the exact
  versions the figures were produced with (nearby versions should also work):
  - `numpy==2.2.6`, `pandas==2.3.3`, `scipy==1.15.3`, `scikit-learn==1.5.2`,
    `matplotlib==3.10.9`, `lifelines==0.30.0`
- Regenerating the mutation-score / radiomic-feature tables from **raw** data
  additionally needs `evo2`, `pyfaidx`, `nnunet`, `nnunetv2`,
  `totalsegmentator`, `SimpleITK`, `nibabel`, `pydicom` (listed as provenance-only
  in `requirements.txt`).

**Hardware**
- Recreating Figure 1 and Figure 2 (and the correlation table) needs **no special
  hardware** — any recent CPU with ≥2 GB RAM. Peak memory observed below.
- Only the upstream steps (Evo2 mutation scoring, tumor segmentation) require a
  CUDA GPU (development used an NVIDIA H100 / A100). They are **not** needed to
  reproduce the figures.

---

## 2. Installation guide

```bash
git clone https://github.com/TruhnLab/OncoGenesRadiologyDiscovery.git
cd OncoGenesRadiologyDiscovery
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

**Typical install time** on a normal desktop: **1–3 minutes** (dominated by
downloading the NumPy/SciPy/matplotlib wheels; no compilation).

---

## 3. Demo

The demo redraws both main-text figures from the intermediate correlation tables.

**Input data.** The intermediate CSVs live under `results/` and are **derived from
publicly available data** (TCGA-KIRC/-LIHC/-BRCA via the NCI GDC and TCIA; OncoKB,
ClinVar, and KEGG). They are not re-distributed in this Git repository. Obtain them
in either of two ways:
- **Quick demo:** download the derived intermediate tables from the manuscript's
  Data Availability archive (**⟨add DOI/URL⟩**) and unpack them into `results/`
  so the tree matches the paths in `config.py`, or
- **From scratch:** regenerate them from the public raw data with the full pipeline
  (see *Instructions for use / Reproduction*, GPU required).

**Run:**

```bash
python make_figures.py                     # fast path — draw figures from the tables
python make_figures.py --regenerate-tables # rebuild discovery_wide_table.csv first
```

**Expected output** (written to `results/final_results/paper/`):
- `fig1_rediscovery.png` / `.pdf`
- `fig2_discovery.png` / `.pdf`
- supporting per-cohort top-10 CSVs (`fig1_top10_oncokb_known_*.csv`)

**Expected run time** on a normal desktop (measured, single core):
- `python make_figures.py` — **~1 minute**, peak ~0.35 GB RAM.
- `python make_figures.py --regenerate-tables` — **~1 min 40 s**, peak ~1.4 GB RAM
  (rebuilds the correlation table byte-for-byte before drawing).

---

## 4. Instructions for use

### Run on your own data
1. Place inputs at the paths declared in `config.py` (genomics under `data/`,
   imaging path set in `config.py`).
2. Score mutation severity and extract radiomic features (GPU steps):
   ```bash
   python genomics/mutation_scoring.py            # Evo2 |ΔLL| per variant (hg38, evo2_7b)
   python radiomics/run_kits23.py                 # cRCC (KIRC) tumor radiomics
   python radiomics/segment_lihc.py               # HCC  (LIHC) liver/lesion radiomics
   python radiomics/segment_brca.py               # BC   (BRCA) tumor radiomics
   ```
   SLURM submission wrappers for these steps are in `scripts/`.
3. Build the correlation table and Fig-2 annotation tables, then draw the figures:
   ```bash
   python make_figures.py --regenerate-tables
   ```

### Data & models required to run from scratch

Not redistributed here — all are **publicly available**; obtain and place them at
the paths below:

| What | Source | Expected path |
|------|--------|---------------|
| TCGA clinical + MAF (cRCC/HCC/BC) | GDC `api.gdc.cancer.gov` (via `data/download_genomic*.py`) | `data/genomic{,_lihc,_brca}/` |
| TCGA CT/MRI imaging | TCIA collections TCGA-KIRC / TCGA-LIHC / TCGA-BRCA | path in `config.py` |
| hg38 reference | UCSC/Ensembl (`hg38.fa`) | `data/reference/hg38.fa` |
| Evo2 (`evo2_7b`) | Arc Institute Evo2 (`pip install evo2`, weights auto-download) | — |
| KiTS23 nnU-Net v1 (cRCC) | KiTS23 challenge model | `models/nnunet/results/.../Task779_Kidneys_KIRC/` |
| MAMA-MIA nnU-Net v2 (BC) | MAMA-MIA challenge model | `models/mama_mia/nnUNet_results/Dataset101_MAMAMMIA/` |
| TotalSegmentator (HCC) | `pip install totalsegmentator` (weights auto-download) | — |
| OncoKB cancer-gene list | oncokb.org cancer gene list | `data/oncokb/cancer_gene_list.tsv` |
| ClinVar / KEGG | auto-fetched (NCBI ClinVar FTP, KEGG REST) and cached | `data/clinvar/`, `data/kegg_pathways.json` |

### Pipeline overview

```
data/download_genomic*.py           clinical + MAF from GDC
genomics/mutation_scoring.py        Evo2 |ΔLL| per variant                 (GPU, hg38, evo2_7b)
radiomics/{run_kits23,segment_lihc,segment_brca}.py   radiomic features    (GPU)
analysis/comprehensive_results.py   -> results/final_results/discovery_wide_table.csv
analysis/test_kegg_groups.py        -> KEGG functional-group tables (Fig 2b)
analysis/test_clinvar_novel.py      -> ClinVar/OMIM table (Fig 2e)
analysis/publication_figures.py     -> fig1_rediscovery, fig2_discovery
```

Both figures derive solely from `discovery_wide_table.csv`; only mutation scoring
and segmentation need a GPU — everything downstream runs on a CPU.

### Reproduction of manuscript results (optional)

To reproduce the two main-text figures exactly, run the demo above against the
released intermediate tables. `python make_figures.py --regenerate-tables`
rebuilds `discovery_wide_table.csv` from the shipped mutation-score and
radiomic-feature CSVs and reproduces it byte-for-byte before drawing.

---

## License

Released under the **MIT License** — an
[OSI-approved](https://opensource.org/licenses/MIT) open-source license.
See [`LICENSE`](LICENSE) for the full text.
