#!/bin/bash
#SBATCH --job-name=df_bs
#SBATCH --ntasks=8
#SBATCH --mem=15GB
#SBATCH --time=24:00:00
#SBATCH --partition=geocean
 
/software/geocean/path/to/lanzador.sh /path/to/project/cases/case_$SLURM_ARRAY_TASK_ID/input.sws