#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=cifar             # sets the job name
#SBATCH --output=cifar.out.%j        # indicates a file to redirect STDOUT to; %j is the jobid
#SBATCH --error=cifar.out.%j         # indicates a file to redirect STDERR to; %j is the jobid
#SBATCH --time=03:00:00                   # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default                       # set QOS, this will determine what resources can be
#SBATCH --gres=gpu:1

srun --partition=class --account=class --qos=default bash -c "
julia --project=. -e '
    include(\"cifar_classification.jl\");
    train()
'; 
echo \"Done.\"" &
# use srun to invoke commands within your job; using an '&'

wait                                                            # wait for any background processes to complete

# once the end of the batch script is reached your job allocation will be revoked
