# OncoGenesRadiologyDiscovery

Associations between **somatic mutation severity** (scored zero-shot by the Evo2
genomic language model, |ΔLL|) and **tumor radiomic phenotype** across three TCGA
cohorts — KIRC (kidney), LIHC (liver), BRCA (breast).

Produces two figures from one TMB-residualized per-gene-metric × imaging
correlation table (`discovery_wide_table.csv`):

- **Fig 1 — Rediscovery:** top OncoKB/CGC genes whose Evo2 severity correlates with imaging.
- **Fig 2 — Discovery:** novel imaging-correlating genes + KEGG groups + ClinVar/OMIM lookup + cross-metric concordance.

## Pipeline

```
download_genomic*.py        clinical + MAF from GDC
genomics/mutation_scoring.py  Evo2 |ΔLL| per variant      (GPU, hg38, evo2_7b)
radiomics/{run_kits23,segment_lihc,segment_brca}.py  radiomic features (GPU)
analysis/comprehensive_results.py  -> discovery_wide_table.csv
analysis/test_kegg_groups.py / test_clinvar_novel.py  Fig 2 annotation tables
analysis/publication_figures.py    -> fig1_rediscovery, fig2_discovery
```

Run figures: `pip install -r requirements.txt && python make_figures.py`
(`--regenerate-tables` rebuilds the correlation table first).

## Data & models (to run the pipeline from scratch)

Not redistributed here — obtain and place at the paths below:

| What | Source | Expected path |
|------|--------|---------------|
| TCGA clinical + MAF (KIRC/LIHC/BRCA) | GDC `api.gdc.cancer.gov` (via `data/download_genomic*.py`) | `data/genomic{,_lihc,_brca}/` |
| TCGA CT/MRI imaging | TCIA collections TCGA-KIRC / TCGA-LIHC / TCGA-BRCA | path in `config.py` |
| hg38 reference | UCSC/Ensembl (`hg38.fa`) | `data/reference/hg38.fa` |
| Evo2 (`evo2_7b`) | Arc Institute Evo2 (`pip install evo2`, weights auto-download) | — |
| KiTS23 nnU-Net v1 (KIRC) | KiTS23 challenge model | `models/nnunet/results/.../Task779_Kidneys_KIRC/` |
| MAMA-MIA nnU-Netv2 (BRCA) | MAMA-MIA challenge model | `models/mama_mia/nnUNet_results/Dataset101_MAMAMMIA/` |
| TotalSegmentator (LIHC) | `pip install totalsegmentator` (weights auto-download) | — |
| OncoKB cancer-gene list | oncokb.org cancer gene list | `data/oncokb/cancer_gene_list.tsv` |
| ClinVar / KEGG | auto-fetched (NCBI ClinVar FTP, KEGG REST) and cached | `data/clinvar/`, `data/kegg_pathways.json` |

Only the figure step (`analysis/comprehensive_results.py` → `publication_figures.py`)
runs on a CPU; mutation scoring and segmentation need a GPU.
