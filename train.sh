#!/bin/bash
#SBATCH --time=4:0:0
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --account=def-cbravo 
#SBATCH --output ./%J.out # STDOUT

cd /home/emiliano/projects/def-cbravo/emiliano/DyGLib/
source ~/TGN/bin/activate

python  train_link_prediction.py --dataset_name Synthetic --model_name DyGFormer --load_best_configs --num_runs 1 --gpu 1 --num_epochs 10 


python  train_snapshot.py --dataset_name Synthetic --model_name DyGFormer --load_best_configs --num_runs 1 --gpu 1 --num_epochs 10 

python  produce_embeddings.py --dataset_name Synthetic --model_name DyGFormer --load_best_configs --num_runs 1 --gpu 1 --num_epochs 10 


