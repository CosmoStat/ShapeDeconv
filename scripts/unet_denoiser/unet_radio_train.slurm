#!/bin/bash
#SBATCH --job-name=unet_radio_train       # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
# /!\ Caution, in the following line, "multithread" refers to hyperthreading.
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=100:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=unet_radio_train%j.out   # output file name
#SBATCH --error=unet_radio_train%j.out    # error file name
#SBATCH --qos=qos_gpu-t4


module purge
module load tensorflow-gpu/py3/1.15.2

set -x

cd $WORK/GitHub/ShapeDeconv/scripts/unet_denoiser

python ./unet_train_radio.py --data_dir=/gpfswork/rech/xdy/uze68md/data/meerkat_3600/ --model_dir=/gpfswork/rech/xdy/uze68md/trained_models/model_meerkat/ --n_col=128 --n_row=128 --batch_size=32 --steps=6500 --epochs=20