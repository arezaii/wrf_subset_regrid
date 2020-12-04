#!/bin/bash
#SBATCH -J pf_regrid         # job name
#SBATCH -o regrid.o%j.txt   # output and error file name (%j expands to jobID)
#SBATCH -n 1                   # One core requested
##SBATCH --exclusive
#SBATCH -p  shortq           # queue (partition) -- defq, ipowerq, eduq, gpuq.
#SBATCH -t 3:00:00          # run time (d-hh:mm:ss)

# Mail alert at start, end and abortion of execution
#SBATCH --mail-type=FAIL,END

# send mail to this address
#SBATCH --mail-user=ahmadrezaii@u.boisestate.edu

#ulimit -v unlimited
#ulimit -u 1000
#ulimit -s unlimited

# Execute the program:
# Replace *.pls with the name of your pulse file.
# All locations are relative.
module purge
module load slurm
module load anaconda
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
cd ~/git/wrf_subset_regrid
conda activate wrf_subset_regrid
python /scratch/arezaii/wrf_subset_regrid/subset.py -i /scratch/arezaii/wrf_out/wy_2015/d01/ -o /scratch/arezaii/WRF/wy2015/ -x 704 -y 736 -s 10-01-2014 -e 10-01-2015 -l /scratch/arezaii/snake_river_shape_domain/input_files/snake_river.latlon.txt -d 60 -n 60
#python subset.py 365



