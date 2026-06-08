"""Central configuration for the EvoRadioFeatures pipeline."""
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Input dataset (imaging)
IMAGING_DATASET = Path("/hpcwork/p0021834/dataset")
DICOM_ROOT = IMAGING_DATASET / "download" / "TCIA_TCGA-KIRC_09-16-2015" / "TCGA-KIRC"
IMAGING_METADATA = IMAGING_DATASET / "download" / "TCIA_TCGA-KIRC_09-16-2015" / "metadata.csv"

# Downloaded genomic/clinical data
GENOMIC_DIR = DATA_DIR / "genomic"
MAF_DIR = GENOMIC_DIR / "maf_files"
CLINICAL_CSV = GENOMIC_DIR / "clinical.csv"
PATIENT_MANIFEST = GENOMIC_DIR / "patient_manifest.json"

# Reference genome
REFERENCE_DIR = DATA_DIR / "reference"
REFERENCE_GENOME = REFERENCE_DIR / "hg38.fa"

# Outputs
RADIOMICS_DIR = RESULTS_DIR / "radiomics"
RADIOMICS_FEATURES_CSV = RADIOMICS_DIR / "radiomics_features.csv"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "evo2_embeddings.npz"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
MUTATION_SCORES_DIR = RESULTS_DIR / "mutation_scores"
MUTATION_SCORES_CSV = MUTATION_SCORES_DIR / "all_mutation_scores.csv"
COMPREHENSIVE_ANALYSIS_DIR = RESULTS_DIR / "comprehensive_analysis"
PATIENT_FEATURES_CSV = COMPREHENSIVE_ANALYSIS_DIR / "data" / "patient_feature_matrix.csv"

# === GDC API ===
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
TCGA_PROJECT = "TCGA-KIRC"

# === Evo2 ===
EVO2_MODEL = "evo2_7b"
# Layer to extract embeddings from (intermediate layer, per Goodfire SAE analysis
# showing most features of interest are represented at layer 26)
EVO2_EMBED_LAYER = "blocks.26.mlp.l3"
# Context window around each mutation (bp upstream + downstream)
MUTATION_CONTEXT_BP = 512  # 512bp on each side = 1024bp total per mutation

# === Genomic Constants ===
KIRC_DRIVER_GENES = {
    "VHL", "PBRM1", "BAP1", "SETD2", "KDM5C", "MTOR", "PIK3CA",
    "TP53", "ARID1A", "PTEN", "TSC1", "TSC2", "TCEB1",
}
FUNCTIONAL_MUTATIONS = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "Nonstop_Mutation",
}
SILENT_MUTATIONS = {"Silent", "3'UTR", "5'UTR", "Intron"}

# === Radiomics ===
# HU window for kidney/tumor tissue in CT
HU_KIDNEY_MIN = -50
HU_KIDNEY_MAX = 300
# Minimum connected component size (voxels) to consider as potential tumor
MIN_TUMOR_VOXELS = 100
