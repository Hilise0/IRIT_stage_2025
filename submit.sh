#!/bin/sh

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=FlowMatching
#SBATCH -o ./slurm_logs/slurm.%j.out # STDOUT
#SBATCH -e ./slurm_logs/slurm.%j.err # STDERR
#SBATCH --partition=GPUNodes
#SBATCH --nodelist=r9ng-1080-[2-6]
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

source .venv/bin/activate

python infer_vevostyle.py