#!/bin/sh
#SBATCH -n 36
#SBATCH -N 2
#SBATCH -t 10:00:00
#SBATCH -p kozinsky
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=ALL


module load cmake/3.17.3-fasrc01
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01                                                                                        
module load intel-mkl/2019.5.281-fasrc01
module load eigen/3.3.7-fasrc01

export OMP_NUM_THREADS=1
mpirun -n 36 ./time_mpi 100
