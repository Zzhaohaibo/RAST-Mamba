# RNP-Mamba v4 / IRP-Mamba implementation notes

新增模型名：`rnp_mamba_v4_irp`。

## Framework

```text
raw x ---------------------------> STIDCore --------------------> y_stid
   |                                                              |
   |-> Moving Average Decomposition -> x_res                      |
                                  |                               |
                                  -> Sparse Top-K Graph           |
                                  -> Spatial-view Bi-Mamba        |
                                  -> Residual head -> delta_y ----|
                                                                  v
y_hat = y_stid + beta * energy_gate(x_res) * delta_y
```

## Code-source lineage

- `STIDCore`: follows the official STID code pattern: history Conv2d embedding + node/time identity + residual 1x1 Conv MLP + Conv2d prediction head.
- `RawMovingAverageDecomposition`: keeps the decomposed traffic forecasting direction used by decomposition-based Mamba/STGNN methods.
- `SparseAdaptiveGraphGuidance`: follows AGCRN's adaptive adjacency idea: learn node embeddings and construct graph supports from node embedding similarity. This implementation adds Top-K sparsification for stability.
- `SpatialViewBiMamba`: follows DST-Mamba's node-token/spatial-view Mamba scan: first embed each node's historical residual sequence as one token, then scan over the node dimension. A reversed scan is included for bidirectional context.
- `ResidualEnergyScale`: a very small scalar controller. It is not a feature-wise gate; it only scales the residual correction based on residual energy.

## Run

```bash
bash scripts/run_rnp_mamba_v4_irp_pems08.sh
```

## Recommended ablations

1. STID baseline: `--model stid_core`
2. Fixed residual correction: `--model rnp_mamba_v4_irp --rnp_use_energy_gate 0`
3. Smaller residual correction: `--rnp_residual_scale_init 0.05`
4. Graph Top-K sweep: `--rnp_spatial_topk 4/8/16`
5. Decomposition kernel sweep: `--rnp_decomp_kernel 3/5/7`
