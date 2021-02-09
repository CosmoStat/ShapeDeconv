#!/bin/bash
#SBATCH --job-name=gpu_mono         # nom du job
##SBATCH --partition=gpu_p1          # partition GPU choisie
#SBATCH --ntasks=1                  # nombre de taches a reserver (=nombre de GPU ici)
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de coeurs CPU par tache (un quart du noeud ici)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread        # hyperthreading desactive
#SBATCH --time=08:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=cfht2hst_parametric%j.out     # nom du fichier de sortie
#SBATCH --error=cfht2hst_parametric%j.out      # nom du fichier d'erreur (ici commun avec la sortie)
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load tensorflow-gpu/py3/1.15.2
 
# echo des commandes lancees
set -x
 
# execution du code
export OMP_NUM_THREADS=10
/gpfswork/rech/xdy/uze68md/GitHub/galaxy2galaxy/galaxy2galaxy/bin/g2g-datagen --problem=attrs2img_cosmos_parametric_cfht2hst --data_dir=$WORK/data/attrs2img_cosmos_parametric_cfht2hst --tmp_dir=$WORK/data/