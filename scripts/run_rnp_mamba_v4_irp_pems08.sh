#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=1

python train.py \
  --model rnp_mamba_v4_irp \
  --data_dir /root/autodl-tmp/BasicTS/datasets/PEMS08 \
  --save_dir checkpoints/rnp_mamba_v4_irp_pems08 \
  --input_len 12 \
  --output_len 12 \
  --points_per_day 288 \
  --batch_size 64 \
  --epochs 100 \
  --patience 15 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --milestones 1 50 80 \
  --gamma 0.5 \
  --dropout 0.15 \
  --rnp_d_model 64 \
  --rnp_decomp_kernel 3 \
  --rnp_d_state 16 \
  --rnp_d_conv 2 \
  --rnp_expand 1 \
  --rnp_spatial_topk 8 \
  --rnp_spatial_node_dim 16 \
  --rnp_stid_embed_dim 32 \
  --rnp_stid_layers 3 \
  --rnp_residual_head_layers 1 \
  --rnp_residual_scale_init 0.1 \
  --rnp_use_energy_gate 1 \
  --embed_dim 32 \
  --num_layers 3 \
  --node_dim 32 \
  --tid_dim 32 \
  --diw_dim 32 \
  --device cuda:0
