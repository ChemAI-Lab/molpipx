#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --account=rrg-ravh011
#SBATCH --job-name=results_gp_200_RBF
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/fs0/scratch/r/ravh011/asmaj/PIPMSA_jax/scripts_and_logs/PIPGP/out_results_gp_200_RBF.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
module load python/3.11
source /home/r/ravh011/asmaj/virtual_envs/PIP/bin/activate
python /scratch/r/ravh011/asmaj/PIPMSA_jax/examples/pipgp/main.py --workdir /gpfs/fs0/scratch/r/ravh011/asmaj/PIPMSA_jax --kernel_type RBF --decay_steps 700 --peak_value 0.01 --warmup_steps 75 --end_value 0.0 --molecule A4B --poly_degree 3 --ntr 200 --ntst 5000 --num_iter 800


