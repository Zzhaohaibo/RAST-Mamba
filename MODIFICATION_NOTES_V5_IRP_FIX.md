# RNP-Mamba v5 IRP Fix

新增模型：`rnp_mamba_v5_irp_fix`。

本版本基于 v4_irp 的失败排查结果，按 Claude 建议做高 ROI 修正，但不覆盖原有 v4/v3/v2 代码。

## 主要修改

1. **新增 raw-scale loss 选项**
   - `train.py` 新增 `--loss_scale {norm,raw}`。
   - 脚本默认使用 `--loss_scale raw`，训练 loss 与验证/测试的 raw MAE 对齐。

2. **残差分支不再只吃 MA residual**
   - v4: `delta_y = residual_branch(x - MA(x))`
   - v5: `delta_y = residual_branch(x, x_ts)`
   - 分解模块仍保留在残差分支内部，用作辅助上下文，而不是把输入信息提前截断。

3. **删除 residual-energy sigmoid gate**
   - v5 不再使用 `ResidualEnergyScale`。
   - 输出融合为：`y_hat = y_stid + residual_scale * horizon_scale * delta_y`。
   - `residual_scale_init` 默认脚本设为 1.0。

4. **残差分支加入 temporal identity**
   - 在 residual token 中加入 node embedding、time-of-day embedding、day-of-week embedding。
   - 解决 v4 residual branch 不知道当前时段的问题。

5. **Graph guidance 从 static top-k 升级为 batch-wise dynamic top-k**
   - 保留 AGCRN-style static node-embedding adjacency。
   - 增加当前 token 的 QK 动态图分数。
   - Top-K 在 batch 维度上构造 `[B,N,N]` adjacency。

## 运行

```bash
bash scripts/run_rnp_mamba_v5_irp_fix_pems08.sh
```

## 建议消融顺序

1. `rnp_mamba_v5_irp_fix` 默认脚本。
2. `--loss_scale norm`：验证 raw loss 是否有效。
3. `--rnp_residual_scale_init 0.5`：验证残差修正强度。
4. `--rnp_use_decomp_context 0`：验证内部 MA context 是否有益。
5. `--rnp_spatial_topk 4/8/16`：验证动态图稀疏度。
