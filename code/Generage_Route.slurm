#!/bin/bash
#SBATCH --job=wml
#SBATCH --partition=kshdnormal
#SBATCH --exclude=./black_list
#SBATCH --mem=90G
#SBATCH --exclusive

#SBATCH --ntasks-per-node=8
## SBATCH --cpus-per-task=8

#SBATCH -N 251
#SBATCH --gres=dcu:4

export UCX_TLS=sm,rc_x 

date

module rm compiler/rocm/3.9.1
module add compiler/cmake/3.15.6
module add compiler/rocm/4.0.1

mpirun --bind-to none --mca pml ucx --mca osc ucx -np 2001 python map_analysis.py

date
