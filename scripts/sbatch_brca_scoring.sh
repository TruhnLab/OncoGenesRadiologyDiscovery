#!/usr/local_rwth/bin/zsh

#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --cpus-per-task=8
#SBATCH --job-name=brca_score
#SBATCH --output=Logs/brca_scoring_output.txt
#SBATCH --error=Logs/brca_scoring_error.txt
#SBATCH --gres=gpu:1
#SBATCH --time=0-16:00:00

date
module load Python/3.12.3
export VIRTUAL_ENV=/hpcwork/ce555345/envs/evo2_env
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages:$VIRTUAL_ENV/lib64/python3.12/site-packages"
export CC=gcc
export PYTHONUNBUFFERED=1

cd /hpcwork/ce555345/projects/EvoRadioFeatures

python3 -c "
import sys, csv, logging
from pathlib import Path
sys.path.insert(0, '.')
import config
from genomics.mutation_scoring import score_patient, FIELDNAMES, load_scored_patients
from data.dataset_brca import TCGABRCADataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

output_csv = Path('results/brca/mutation_scores.csv')
output_csv.parent.mkdir(parents=True, exist_ok=True)

scored = load_scored_patients(output_csv)
log.info('Already scored: %d', len(scored))

dataset = TCGABRCADataset(require_both=False)
patients = [p for p in dataset.patients if p.n_mutations > 0 and p.patient_id not in scored]
log.info('To score: %d', len(patients))
if not patients: sys.exit(0)

from evo2 import Evo2
model = Evo2(config.EVO2_MODEL)

write_header = not output_csv.exists() or output_csv.stat().st_size == 0
with open(output_csv, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header: writer.writeheader()
    for i, p in enumerate(patients):
        log.info('[%d/%d] %s (%d mutations)', i+1, len(patients), p.patient_id, p.n_mutations)
        try:
            results = score_patient(model, p, context_bp=4096, batch_size=4)
            writer.writerows(results)
            f.flush()
            n_del = sum(1 for r in results if r['delta_ll'] < 0)
            log.info('  %d scored, %d deleterious', len(results), n_del)
        except Exception as e:
            log.error('  Failed: %s', e)
log.info('Done.')
"

date
