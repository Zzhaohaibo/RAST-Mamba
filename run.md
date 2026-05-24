### PEMS08运行命令
```
export OMP_NUM_THREADS=1

python train.py \
  --model rast_mamba \
  --data_dir /root/autodl-tmp/datasets/PEMS08 \
  --save_dir checkpoints/rast_mamba_pems08 \
  --input_len 12 \
  --output_len 12 \
  --points_per_day 288 \
  --batch_size 64 \
  --epochs 100 \
  --patience 15 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --milestones 50 80 \
  --gamma 0.1 \
  --grad_clip 5.0 \
  --loss_scale raw \
  --rnp_d_model 64 \
  --rnp_d_state 16 \
  --rnp_d_conv 2 \
  --rnp_expand 1 \
  --rnp_spatial_topk 8 \
  --rnp_spatial_node_dim 16 \
  --dropout 0.15 \
  --device cuda:0
```

```
export OMP_NUM_THREADS=1

python -u train.py \
  --model rnp_mamba_v6_dual \
  --data_dir /root/autodl-tmp/BasicTS/datasets/PEMS08 \
  --save_dir checkpoints/rnp_mamba_v6_stembed_pems08 \
  --input_len 12 \
  --output_len 12 \
  --points_per_day 288 \
  --batch_size 64 \
  --epochs 100 \
  --patience 15 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --milestones 50 80 \
  --gamma 0.1 \
  --grad_clip 5.0 \
  --loss_scale raw \
  --rnp_d_model 64 \
  --rnp_d_state 16 \
  --rnp_d_conv 2 \
  --rnp_expand 1 \
  --dropout 0.15 \
  --rnp_decomp_mode st_embed \
  --rnp_spatial_topk 8 \
  --rnp_spatial_node_dim 16 \
  --rnp_spatial_mode local_graph_mamba \
  --rnp_spatial_time_mode summary \
  --rnp_residual_scale_init 1.0          # 从 0.2 拉到 1.0
  --rnp_residual_gate_bias_init 0.0      # 从 -0.6 拉到 0，开局门 = 0.5
  --rnp_aux_nonrec_weight 0.3            # 从 0.05 拉到 0.3
  --rnp_aux_detach_rec 1                 # 保持 detach，让 aux loss 只推 non-rec
  --device cuda:0

```
## Recommended Commands (v6 baseline / v6.2-lite)

### v6 baseline

```bash
export OMP_NUM_THREADS=1

python -u train.py \
  --model rnp_mamba_v6_dual \
  --data_dir /root/autodl-tmp/BasicTS/datasets/PEMS08 \
  --save_dir checkpoints/rnp_mamba_v6_dual_pems08 \
  --input_len 12 \
  --output_len 12 \
  --points_per_day 288 \
  --batch_size 64 \
  --epochs 100 \
  --patience 15 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --milestones 50 80 \
  --gamma 0.1 \
  --grad_clip 5.0 \
  --loss_scale raw \
  --rnp_d_model 64 \
  --rnp_d_state 16 \
  --rnp_d_conv 2 \
  --rnp_expand 1 \
  --dropout 0.15 \
  --rnp_decomp_mode st_embed \
  --rnp_spatial_topk 8 \
  --rnp_spatial_node_dim 16 \
  --rnp_spatial_mode local_graph_mamba \
  --rnp_spatial_time_mode summary \
  --rnp_residual_scale_init 0.2 \
  --rnp_residual_gate_bias_init -0.6 \
  --rnp_aux_nonrec_weight 0.05 \
  --rnp_aux_detach_rec 1 \
  --device cuda:0
```

### v6.2-lite

```bash
export OMP_NUM_THREADS=1

python -u train.py \
  --model rnp_mamba_v62_lite \
  --data_dir /root/autodl-tmp/BasicTS/datasets/PEMS08 \
  --save_dir checkpoints/rnp_mamba_v62_lite_pems08 \
  --input_len 12 \
  --output_len 12 \
  --points_per_day 288 \
  --batch_size 64 \
  --epochs 100 \
  --patience 15 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --milestones 50 80 \
  --gamma 0.1 \
  --grad_clip 5.0 \
  --loss_scale raw \
  --rnp_d_model 64 \
  --rnp_d_state 16 \
  --rnp_d_conv 2 \
  --rnp_expand 1 \
  --dropout 0.15 \
  --rnp_decomp_mode st_embed \
  --rnp_spatial_topk 8 \
  --rnp_spatial_node_dim 16 \
  --rnp_spatial_mode local_graph_mamba \
  --rnp_spatial_time_mode summary \
  --rnp_residual_scale_init 0.2 \
  --rnp_residual_gate_bias_init -0.6 \
  --rnp_aux_nonrec_weight 0.05 \
  --rnp_aux_detach_rec 1 \
  --device cuda:0
```
