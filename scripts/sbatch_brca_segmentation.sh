#!/usr/local_rwth/bin/zsh

#SBATCH --account=p0021834
#SBATCH --cpus-per-task=8
#SBATCH --job-name=brca_seg
#SBATCH --output=Logs/brca_segmentation_output.txt
#SBATCH --error=Logs/brca_segmentation_error.txt
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00

date
module load Python/3.12.3
export VIRTUAL_ENV=/hpcwork/ce555345/envs/evo2_env
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages:$VIRTUAL_ENV/lib64/python3.12/site-packages"
export CC=gcc
export PYTHONUNBUFFERED=1

cd /hpcwork/ce555345/projects/EvoRadioFeatures
python3 radiomics/segment_brca.py

date
