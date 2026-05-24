import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from models.stid_core import STIDCore


def _init_bounded_scale(value: float, max_value: float, eps: float = 1e-4) -> torch.Tensor:
    ratio = float(value) / max(float(max_value), eps)
    ratio = min(max(ratio, eps), 1.0 - eps)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


def _bounded_positive_scale(raw: torch.Tensor, max_value: float) -> torch.Tensor:
    return float(max_value) * torch.sigmoid(raw)


class STIDStyleResidualMLP(nn.Module):
    """
    STID-style residual MLP block.

    Reference:
        STID/stid/arch/mlp.py
        Conv2d -> ReLU -> Dropout -> Conv2d -> Dropout -> Residual

    Input / output:
        x: [B, C, N, 1]
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()

        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.fc1(x)
        hidden = self.act(hidden)
        hidden = self.drop(hidden)
        hidden = self.fc2(hidden)
        hidden = self.drop(hidden)
        return hidden + residual


class STIDStylePostFusionHead(nn.Module):
    """
    STID-style post-fusion encoder + regression head.

    Difference from v1.1 Linear prediction head:
        v1.1:
            [B,N,D] -> Linear -> GELU -> Linear -> [B,N,H]

        v1.2:
            [B,N,D] -> [B,D,N,1]
                     -> STID-style Residual MLP x L
                     -> Conv2d(D -> H)
                     -> [B,H,N]

    Input:
        z: [B, N, D]

    Output:
        pred: [B, H, N]
    """

    def __init__(
        self,
        d_model: int,
        output_len: int,
        num_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            *[
                STIDStyleResidualMLP(
                    input_dim=d_model,
                    hidden_dim=d_model,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.regression_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, N, D]

        Returns:
            pred: [B, H, N]
        """
        if z.dim() != 3:
            raise ValueError(f"STIDStylePostFusionHead expects [B,N,D], got {tuple(z.shape)}")

        # [B,N,D] -> [B,D,N,1]
        hidden = z.permute(0, 2, 1).unsqueeze(-1).contiguous()

        hidden = self.encoder(hidden)

        # [B,D,N,1] -> [B,H,N,1] -> [B,H,N]
        pred = self.regression_layer(hidden).squeeze(-1)

        return pred


class RawMovingAverageDecomposition(nn.Module):
    """
    Raw-space decomposition:
        x_rec = Smooth(x)
        x_non = x - x_rec

    Input:
        x: [B, T, N]
    Output:
        x_rec: [B, T, N]
        x_non: [B, T, N]
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor):
        B, T, N = x.shape

        # [B,T,N] -> [B,N,T]
        x_ = x.permute(0, 2, 1).contiguous()

        # replicate padding on temporal dimension
        x_ = F.pad(x_, (self.padding, self.padding), mode="replicate")
        x_rec = F.avg_pool1d(x_, kernel_size=self.kernel_size, stride=1)

        # [B,N,T] -> [B,T,N]
        x_rec = x_rec.permute(0, 2, 1).contiguous()
        x_non = x - x_rec
        return x_rec, x_non


class TemporalMLPBranch(nn.Module):
    """
    Lightweight temporal MLP branch for recurring patterns.

    Input / output:
        [B, T, N, D]

    It learns temporal mixing over T and feature mixing over D.
    """

    def __init__(self, input_len: int, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.temporal_mixer = nn.Linear(input_len, input_len)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        # x: [B,T,N,D]
        residual = x

        # temporal mixing: [B,T,N,D] -> [B,N,D,T]
        y = x.permute(0, 2, 3, 1).contiguous()
        y = self.temporal_mixer(y)
        y = y.permute(0, 3, 1, 2).contiguous()

        x = residual + y
        x = x + self.ffn(self.norm(x))
        return x


class BiTemporalMambaBranch(nn.Module):
    """
    Bi-directional temporal Mamba branch.

    Mamba scans along temporal dimension T for each node.
    This keeps your original temporal-Mamba design, but improves it
    from uni-directional to bi-directional.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, N, D = x.shape

        # [B,T,N,D] -> [B*N,T,D]
        x_seq = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)

        y_fwd = self.mamba_fwd(x_seq)

        x_rev = torch.flip(x_seq, dims=[1])
        y_bwd = self.mamba_bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.out_proj(y)
        y = self.dropout(y)
        y = self.norm(y + x_seq)

        # [B*N,T,D] -> [B,T,N,D]
        y = y.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return y


class TimeAwareAggregation(nn.Module):
    """
    Time-aware weighted aggregation.

    TimePro-inspired idea:
        do not simply take the last step;
        learn which historical time points are important.

    Input:
        x: [B, T, N, D]
    Output:
        z: [B, N, D]
    """

    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        hidden = max(d_model // 2, 16)
        self.selector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        # score: [B,T,N,1]
        score = self.selector(x)
        weight = torch.softmax(score, dim=1)
        z = (weight * x).sum(dim=1)
        return z


class SpatialViewBiMamba(nn.Module):
    """
    Spatial-view bidirectional Mamba.

    Open-code lineage:
        - DST-Mamba: shift from temporal view to spatial/node view, and use
          forward + reversed Mamba scans over node tokens.

    Input / output:
        x: [B, N, D]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SpatialViewBiMamba expects [B,N,D], got {tuple(x.shape)}")

        y_fwd = self.mamba_fwd(x)
        x_rev = torch.flip(x, dims=[1])
        y_bwd = self.mamba_bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.dropout(self.out_proj(y))
        return self.norm(x + y)


class DynamicSparseAdaptiveGraphGuidance(nn.Module):
    """
    Dynamic sparse adaptive graph guidance.

    Open-code lineage:
        - AGCRN: node-embedding adaptive adjacency as a static spatial prior.
        - Dynamic traffic graph models such as D2STGNN/STAEformer-style designs:
          current sample tokens modulate the adjacency.

    This class keeps the AGCRN-style static graph and adds a lightweight
    batch-wise dynamic score from current node tokens.

    Input / output:
        x: [B, N, D]
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        node_emb_dim: int = 16,
        topk: int = 8,
        dropout: float = 0.15,
        init_gamma: float = 0.1,
        dynamic_alpha: float = 0.5,
    ):
        super().__init__()
        if topk <= 0:
            raise ValueError("topk must be positive.")

        self.num_nodes = num_nodes
        self.topk = topk
        self.dynamic_alpha = float(dynamic_alpha)

        self.node_emb1 = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        self.node_emb2 = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb1)
        nn.init.xavier_uniform_(self.node_emb2)

        self.q_proj = nn.Linear(d_model, node_emb_dim)
        self.k_proj = nn.Linear(d_model, node_emb_dim)

        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"DynamicSparseAdaptiveGraphGuidance expects [B,N,D], got {tuple(x.shape)}")

        B, N, D = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        k = min(self.topk, N)

        # Static AGCRN-style adaptive prior: [N,N].
        static_score = torch.relu(torch.matmul(self.node_emb1, self.node_emb2.transpose(0, 1)))

        # Batch-wise dynamic score from current tokens: [B,N,N].
        q = self.q_proj(x)
        k_token = self.k_proj(x)
        dyn_score = torch.einsum("bnd,bmd->bnm", q, k_token) / (q.shape[-1] ** 0.5)

        score = static_score.unsqueeze(0) + self.dynamic_alpha * dyn_score

        topk_val, topk_idx = torch.topk(score, k=k, dim=-1)
        topk_attn = torch.softmax(topk_val, dim=-1)

        adj = torch.zeros_like(score)
        adj.scatter_(dim=-1, index=topk_idx, src=topk_attn)

        x_agg = torch.einsum("bij,bjd->bid", adj, x)
        x_agg = self.dropout(self.proj(x_agg))
        return self.norm(x + self.gamma * x_agg)


class ContextAwareResidualSpatialMambaBranch(nn.Module):
    """
    Context-aware residual correction branch for v5.

    Changes relative to v4 branch:
        1. It receives the full normalized history x instead of only MA residual.
        2. MA decomposition is kept inside the branch as auxiliary context.
        3. Temporal identity embeddings are injected into residual tokens.
        4. The graph guidance is batch-wise dynamic, not purely static.

    Input:
        x:    [B,T,N]
        x_ts: [B,T,2]
    Output:
        delta_y: [B,H,N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        d_model: int = 64,
        decomp_kernel: int = 3,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        spatial_topk: int = 8,
        spatial_node_dim: int = 16,
        head_layers: int = 1,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        use_decomp_context: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.use_decomp_context = bool(use_decomp_context)

        self.raw_decomp = RawMovingAverageDecomposition(kernel_size=decomp_kernel)

        # Full-history token embedding: [B,T,N] -> [B,N,D].
        self.full_token_emb = nn.Conv2d(
            in_channels=input_len,
            out_channels=d_model,
            kernel_size=(1, 1),
            bias=True,
        )

        if self.use_decomp_context:
            # Decomposition context. This preserves the decomposition story, but
            # the branch is no longer starved by residual-only input.
            self.res_token_emb = nn.Conv2d(input_len, d_model, kernel_size=(1, 1), bias=True)
            self.trend_token_emb = nn.Conv2d(input_len, d_model, kernel_size=(1, 1), bias=True)
            self.decomp_context_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.res_token_emb = None
            self.trend_token_emb = None
            self.decomp_context_scale = None

        self.residual_node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        nn.init.xavier_uniform_(self.residual_node_emb)

        self.tid_emb = nn.Parameter(torch.empty(time_of_day_size, d_model))
        self.diw_emb = nn.Parameter(torch.empty(day_of_week_size, d_model))
        nn.init.xavier_uniform_(self.tid_emb)
        nn.init.xavier_uniform_(self.diw_emb)

        self.token_norm = nn.LayerNorm(d_model)

        self.graph_guidance = DynamicSparseAdaptiveGraphGuidance(
            num_nodes=num_nodes,
            d_model=d_model,
            node_emb_dim=spatial_node_dim,
            topk=spatial_topk,
            dropout=dropout,
            init_gamma=0.1,
            dynamic_alpha=0.5,
        )

        self.spatial_mamba = SpatialViewBiMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        self.pred_head = STIDStylePostFusionHead(
            d_model=d_model,
            output_len=output_len,
            num_layers=head_layers,
            dropout=dropout,
        )

    def _history_to_token(self, x: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
        # [B,T,N] -> [B,T,N,1] -> [B,D,N,1] -> [B,N,D]
        return layer(x.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ContextAwareResidualSpatialMambaBranch expects x [B,T,N], got {tuple(x.shape)}")
        if x_ts.dim() != 3 or x_ts.shape[-1] < 2:
            raise ValueError(f"x_ts must be [B,T,2], got {tuple(x_ts.shape)}")

        B, T, N = x.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        token = self._history_to_token(x, self.full_token_emb)

        if self.use_decomp_context:
            x_trend, x_res = self.raw_decomp(x)
            token_res = self._history_to_token(x_res, self.res_token_emb)
            token_trend = self._history_to_token(x_trend, self.trend_token_emb)
            token = token + self.decomp_context_scale * (token_res + token_trend)

        token = token + self.residual_node_emb.unsqueeze(0)

        tid = x_ts[:, -1, 0].long().clamp(0, self.time_of_day_size - 1)
        diw = x_ts[:, -1, 1].long().clamp(0, self.day_of_week_size - 1)
        token = token + self.tid_emb[tid].unsqueeze(1)
        token = token + self.diw_emb[diw].unsqueeze(1)

        token = self.token_norm(token)
        token = self.graph_guidance(token)
        token = self.spatial_mamba(token)
        delta_y = self.pred_head(token)
        return delta_y


class RNPMambaV5IRPFix(nn.Module):
    """
    RNP-Mamba-v5 / Claude-fix implementation.

    It keeps the v4 identity-preserved residual-correction idea, but applies
    the high-ROI corrections suggested from the v4 failure analysis:
        - no residual-energy sigmoid gate;
        - residual branch receives full x, not only MA residual;
        - temporal identities are injected into the residual branch;
        - static AGCRN-style graph is upgraded with batch-wise dynamic scores;
        - correction is output-level with learnable residual and horizon scales.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        d_model: int = 64,
        decomp_kernel: int = 3,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        node_dim: int = 32,
        temp_dim_tid: int = 32,
        temp_dim_diw: int = 32,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        spatial_topk: int = 8,
        spatial_node_dim: int = 16,
        stid_embed_dim: int = 32,
        stid_layers: int = 3,
        residual_head_layers: int = 1,
        residual_scale_init: float = 1.0,
        use_decomp_context: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len

        self.stid_backbone = STIDCore(
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            embed_dim=stid_embed_dim,
            num_layers=stid_layers,
            dropout=dropout,
            if_node=True,
            node_dim=node_dim,
            if_time_in_day=True,
            if_day_in_week=True,
            temp_dim_tid=temp_dim_tid,
            temp_dim_diw=temp_dim_diw,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
        )

        self.residual_branch = ContextAwareResidualSpatialMambaBranch(
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            d_model=d_model,
            decomp_kernel=decomp_kernel,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            spatial_topk=spatial_topk,
            spatial_node_dim=spatial_node_dim,
            head_layers=residual_head_layers,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
            use_decomp_context=use_decomp_context,
        )

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.residual_horizon_scale = nn.Parameter(torch.ones(output_len))

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"RNPMambaV5IRPFix expects x shape [B,T,N], got {tuple(x.shape)}")

        B, T, N = x.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        y_stid = self.stid_backbone(x, x_ts)
        delta_y = self.residual_branch(x, x_ts)
        delta_y = delta_y * self.residual_horizon_scale.view(1, -1, 1)
        return y_stid + self.residual_scale * delta_y

class MultiScalePeriodicRNPDecomposition(nn.Module):
    """
    Traffic-specific recurring / non-recurring decomposition.

    Input:
        x:    [B, T, N]
        x_ts: [B, T, 2], integer time-of-day and day-of-week indices

    Output:
        x_rec: [B, T, N]
        x_non: [B, T, N]

    Design:
        1. Multi-scale moving averages produce local recurring candidates.
        2. A learnable periodic prototype captures time-of-day / day-of-week regularity.
        3. A small gate chooses between local smoothness and periodic prototype.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        kernels=(3, 5, 7),
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        hidden_dim: int = 16,
        proto_scale_init: float = 0.1,
        proto_scale_max: float = 0.5,
        mode: str = "ma_plus_periodic",
        gate_temperature: float = 1.5,
        gate_min: float = 0.05,
        gate_max: float = 0.95,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.kernels = tuple(int(k) for k in kernels)
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.mode = mode
        self.gate_temperature = float(gate_temperature)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self._last_stats = {}

        valid_modes = {"multi_ma", "multi_ema", "ma_plus_periodic", "none"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown rnp_decomp_mode={mode}. Choices: {sorted(valid_modes)}")

        for k in self.kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError(f"All decomposition kernels must be positive odd integers, got {self.kernels}.")

        # Time-conditioned weights over multiple smoothing scales.
        self.tod_emb = nn.Embedding(time_of_day_size, hidden_dim)
        self.dow_emb = nn.Embedding(day_of_week_size, hidden_dim)
        self.scale_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.kernels)),
        )

        # Learnable periodic prototypes in normalized value space.
        self.tod_proto = nn.Parameter(torch.zeros(time_of_day_size, num_nodes))
        self.dow_proto = nn.Parameter(torch.zeros(day_of_week_size, num_nodes))
        self.proto_scale_max = float(proto_scale_max)
        self.proto_scale = nn.Parameter(_init_bounded_scale(proto_scale_init, self.proto_scale_max))

        # Element-wise gate between smooth local pattern and periodic prototype.
        self.rec_gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _moving_average(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        # x: [B,T,N]
        padding = (kernel_size - 1) // 2
        x_ = x.permute(0, 2, 1).contiguous()  # [B,N,T]
        x_ = F.pad(x_, (padding, padding), mode="replicate")
        x_ = F.avg_pool1d(x_, kernel_size=kernel_size, stride=1)
        return x_.permute(0, 2, 1).contiguous()  # [B,T,N]

    def _exponential_smoothing(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        # x: [B,T,N]. Use a causal EMA candidate, following the stable dual-stream
        # decomposition style used by recent time-series models such as xPatch.
        alpha = 2.0 / float(kernel_size + 1)
        prev = x[:, 0, :]
        out = [prev]
        for t in range(1, x.shape[1]):
            prev = alpha * x[:, t, :] + (1.0 - alpha) * prev
            out.append(prev)
        return torch.stack(out, dim=1)

    def _safe_record_stats(
        self,
        x_rec: torch.Tensor,
        x_non: torch.Tensor,
        gate: torch.Tensor,
        scale_weight: torch.Tensor,
    ) -> None:
        if self.training:
            return
        with torch.no_grad():
            self._last_stats = {
                "x_rec_std": float(x_rec.detach().std(unbiased=False).cpu()),
                "x_non_std": float(x_non.detach().std(unbiased=False).cpu()),
                "rec_gate_mean": float(gate.detach().mean().cpu()),
                "rec_gate_min": float(gate.detach().min().cpu()),
                "rec_gate_max": float(gate.detach().max().cpu()),
                "scale_weight_mean": float(scale_weight.detach().mean().cpu()),
                "proto_scale": float(_bounded_positive_scale(self.proto_scale.detach(), self.proto_scale_max).cpu()),
            }

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"MultiScalePeriodicRNPDecomposition expects x [B,T,N], got {tuple(x.shape)}")
        if x_ts.dim() != 3 or x_ts.shape[-1] < 2:
            raise ValueError(f"x_ts must be [B,T,2], got {tuple(x_ts.shape)}")

        B, T, N = x.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        tid = x_ts[:, :, 0].long().clamp(0, self.time_of_day_size - 1)
        dow = x_ts[:, :, 1].long().clamp(0, self.day_of_week_size - 1)

        if self.mode == "none":
            gate = torch.ones_like(x)
            scale_weight = torch.ones(B, T, len(self.kernels), device=x.device, dtype=x.dtype)
            scale_weight = scale_weight / max(len(self.kernels), 1)
            x_rec = x
            x_non = torch.zeros_like(x)
            self._safe_record_stats(x_rec, x_non, gate, scale_weight)
            return x_rec, x_non

        if self.mode == "multi_ema":
            smooth_list = [self._exponential_smoothing(x, k) for k in self.kernels]
        else:
            smooth_list = [self._moving_average(x, k) for k in self.kernels]
        smooth_stack = torch.stack(smooth_list, dim=-1)  # [B,T,N,K]

        time_feat = torch.cat([self.tod_emb(tid), self.dow_emb(dow)], dim=-1)  # [B,T,2H]
        scale_weight = torch.softmax(self.scale_selector(time_feat), dim=-1)  # [B,T,K]
        x_smooth = (smooth_stack * scale_weight.unsqueeze(2)).sum(dim=-1)  # [B,T,N]

        if self.mode in {"multi_ma", "multi_ema"}:
            gate = torch.ones_like(x)
            x_rec = x_smooth
            x_non = x - x_rec
            self._safe_record_stats(x_rec, x_non, gate, scale_weight)
            return x_rec, x_non

        x_proto = self.tod_proto[tid] + self.dow_proto[dow]  # [B,T,N]
        proto_scale = _bounded_positive_scale(self.proto_scale, self.proto_scale_max).to(dtype=x.dtype)
        x_periodic = x_smooth + proto_scale * torch.tanh(x_proto)

        gate_input = torch.stack([x, x_smooth, x_periodic], dim=-1)  # [B,T,N,3]
        gate_logits = self.rec_gate(gate_input).squeeze(-1)  # [B,T,N]
        gate = torch.sigmoid(gate_logits / max(self.gate_temperature, 1e-6))
        gate = gate.clamp(self.gate_min, self.gate_max)

        x_rec = gate * x_smooth + (1.0 - gate) * x_periodic
        x_non = x - x_rec
        self._safe_record_stats(x_rec, x_non, gate, scale_weight)
        return x_rec, x_non


class STEmbeddingGuidedRNPDecomposition(nn.Module):
    """
    STDN-style spatio-temporal embedding guided value-space decomposition.

    It keeps the public RNP decomposition interface unchanged:
        x [B,T,N], x_ts [B,T,2] -> x_rec [B,T,N], x_non [B,T,N]

    The reusable STDN pattern is:
        spatio-temporal embedding -> sin nonlinearity -> aligned projection,
        then an element-wise recurring gate and residual irregular component.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        hidden_dim: int = 64,
        gate_min: float = 0.05,
        gate_max: float = 0.95,
        init_rec_ratio: float = 0.75,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.hidden_dim = hidden_dim
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self._last_stats = {}

        if not (0.0 <= self.gate_min < self.gate_max <= 1.0):
            raise ValueError(f"Expected 0 <= gate_min < gate_max <= 1, got {gate_min}, {gate_max}.")

        self.value_proj = nn.Linear(1, hidden_dim)

        self.node_emb = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.tod_emb = nn.Parameter(torch.empty(time_of_day_size, hidden_dim))
        self.dow_emb = nn.Parameter(torch.empty(day_of_week_size, hidden_dim))
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.tod_emb)
        nn.init.xavier_uniform_(self.dow_emb)

        self.st_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        p = (float(init_rec_ratio) - self.gate_min) / max(self.gate_max - self.gate_min, 1e-6)
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, float(torch.logit(torch.tensor(p))))

    def _safe_record_stats(self, x_rec: torch.Tensor, x_non: torch.Tensor, rec_gate: torch.Tensor) -> None:
        if self.training:
            return
        with torch.no_grad():
            self._last_stats = {
                "x_rec_std": float(x_rec.detach().std(unbiased=False).cpu()),
                "x_non_std": float(x_non.detach().std(unbiased=False).cpu()),
                "rec_gate_mean": float(rec_gate.detach().mean().cpu()),
                "rec_gate_min": float(rec_gate.detach().min().cpu()),
                "rec_gate_max": float(rec_gate.detach().max().cpu()),
                "scale_weight_mean": None,
                "proto_scale": None,
                "decomp_mode": "st_embed",
            }

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"STEmbeddingGuidedRNPDecomposition expects x [B,T,N], got {tuple(x.shape)}")
        if x_ts.dim() != 3 or x_ts.shape[-1] < 2:
            raise ValueError(f"x_ts must be [B,T,2], got {tuple(x_ts.shape)}")

        B, T, N = x.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        tid = x_ts[:, :, 0].long().clamp(0, self.time_of_day_size - 1)
        dow = x_ts[:, :, 1].long().clamp(0, self.day_of_week_size - 1)

        h_value = self.value_proj(x.unsqueeze(-1))  # [B,T,N,D]
        D = h_value.shape[-1]

        node_h = self.node_emb.view(1, 1, N, D).expand(B, T, N, D)
        tod_h = self.tod_emb[tid].unsqueeze(2).expand(B, T, N, D)
        dow_h = self.dow_emb[dow].unsqueeze(2).expand(B, T, N, D)

        st_h = node_h + tod_h + dow_h
        st_h = torch.sin(st_h)
        st_h = self.st_proj(st_h)

        gate_input = torch.cat([h_value, st_h], dim=-1)  # [B,T,N,2D]
        raw_gate = torch.sigmoid(self.gate_mlp(gate_input)).squeeze(-1)  # [B,T,N]
        rec_gate = self.gate_min + (self.gate_max - self.gate_min) * raw_gate

        x_rec = rec_gate * x
        x_non = x - x_rec
        self._safe_record_stats(x_rec, x_non, rec_gate)
        return x_rec, x_non


class RecurringSTIDLightBranch(nn.Module):
    """
    Lightweight recurring-pattern forecaster.

    This absorbs the useful STID idea into the recurring branch:
        value embedding + node identity + time-of-day identity + day-of-week identity
        + light temporal MLP + STID-style prediction head.

    Input:
        x_rec: [B,T,N]
        x_ts:  [B,T,2]
    Output:
        y_rec: [B,H,N]
        z_rec: [B,N,D]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        d_model: int = 64,
        dropout: float = 0.15,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        head_layers: int = 2,
        identity_scale_init: float = 0.3,
        identity_scale_max: float = 0.8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        self.value_proj = nn.Linear(1, d_model)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        self.tod_emb = nn.Parameter(torch.empty(time_of_day_size, d_model))
        self.dow_emb = nn.Parameter(torch.empty(day_of_week_size, d_model))
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.tod_emb)
        nn.init.xavier_uniform_(self.dow_emb)

        self.identity_scale_max = float(identity_scale_max)
        self.identity_scale = nn.Parameter(_init_bounded_scale(identity_scale_init, self.identity_scale_max))
        self.input_norm = nn.LayerNorm(d_model)
        self.dw_temporal = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(3, 1),
                padding=(1, 0),
                groups=d_model,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1), bias=True),
            nn.Dropout(dropout),
        )
        self.dw_norm = nn.LayerNorm(d_model)

        self.temporal_mlp = TemporalMLPBranch(
            input_len=input_len,
            d_model=d_model,
            dropout=dropout,
        )
        self.agg = TimeAwareAggregation(d_model=d_model, dropout=dropout)
        self.pred_head = STIDStylePostFusionHead(
            d_model=d_model,
            output_len=output_len,
            num_layers=head_layers,
            dropout=dropout,
        )

    def build_identity(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        tid = x_ts[:, :, 0].long().clamp(0, self.time_of_day_size - 1)
        dow = x_ts[:, :, 1].long().clamp(0, self.day_of_week_size - 1)

        node_emb = self.node_emb.view(1, 1, N, -1).expand(B, T, -1, -1)
        tid_emb = self.tod_emb[tid].unsqueeze(2).expand(-1, -1, N, -1)
        dow_emb = self.dow_emb[dow].unsqueeze(2).expand(-1, -1, N, -1)
        return node_emb + tid_emb + dow_emb

    def forward(self, x_rec: torch.Tensor, x_ts: torch.Tensor):
        if x_rec.dim() != 3:
            raise ValueError(f"RecurringSTIDLightBranch expects x_rec [B,T,N], got {tuple(x_rec.shape)}")
        B, T, N = x_rec.shape
        if T != self.input_len or N != self.num_nodes:
            raise ValueError(f"Expected [B,{self.input_len},{self.num_nodes}], got {tuple(x_rec.shape)}")

        h = self.value_proj(x_rec.unsqueeze(-1))
        identity_scale = _bounded_positive_scale(self.identity_scale, self.identity_scale_max).to(dtype=h.dtype)
        h = h + identity_scale * self.build_identity(x_rec, x_ts)
        h = self.input_norm(h)
        h_dw = self.dw_temporal(h.permute(0, 3, 1, 2).contiguous())
        h = self.dw_norm(h + h_dw.permute(0, 2, 3, 1).contiguous())
        h = self.temporal_mlp(h)
        z_rec = self.agg(h)
        y_rec = self.pred_head(z_rec)
        return y_rec, z_rec


class DynamicTopKGraphConstructor(nn.Module):
    """
    Build an adaptive dynamic graph from temporally encoded node states.

    Input:
        z: [B,N,D]
    Output:
        A_dyn:     [B,N,N]
        A_adp:     [N,N]
        topk_idx:  [B,N,K]
        topk_attn: [B,N,K]
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        node_emb_dim: int = 16,
        topk: int = 8,
        dynamic_alpha: float = 1.0,
    ):
        super().__init__()
        if topk <= 0:
            raise ValueError("topk must be positive.")
        self.num_nodes = num_nodes
        self.topk = int(topk)
        self.dynamic_alpha = float(dynamic_alpha)

        self.node_emb1 = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        self.node_emb2 = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb1)
        nn.init.xavier_uniform_(self.node_emb2)

        self.q_proj = nn.Linear(d_model, node_emb_dim)
        self.k_proj = nn.Linear(d_model, node_emb_dim)
        # [adp, dyn, self, phy]. Softmax keeps graph fusion bounded.
        self.graph_logits = nn.Parameter(torch.tensor([1.0, 1.0, 0.2, 0.0]))
        self._last_stats = {
            "lambda_adp": 0.0,
            "lambda_dyn": 0.0,
            "lambda_self": 0.0,
            "lambda_phy": None,
        }

    def _row_softmax_graph(self, score: torch.Tensor, mask_self: bool = True) -> torch.Tensor:
        N = score.shape[-1]
        if N <= 1:
            return torch.zeros_like(score)
        if mask_self:
            eye = torch.eye(N, device=score.device, dtype=torch.bool)
            if score.dim() == 3:
                eye = eye.view(1, N, N)
            score = score.masked_fill(eye, float("-inf"))
        return torch.softmax(score, dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        A_phy: torch.Tensor = None,
        disable_dynamic: bool = False,
    ):
        if z.dim() != 3:
            raise ValueError(f"DynamicTopKGraphConstructor expects z [B,N,D], got {tuple(z.shape)}")
        B, N, D = z.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        k = min(self.topk, max(N - 1, 0))

        static_score = torch.relu(torch.matmul(self.node_emb1, self.node_emb2.transpose(0, 1)))  # [N,N]
        q = self.q_proj(z)
        key = self.k_proj(z)
        dyn_score = torch.einsum("bnd,bmd->bnm", q, key) / (q.shape[-1] ** 0.5)

        A_adp = self._row_softmax_graph(static_score, mask_self=True)  # [N,N]
        A_dyn = self._row_softmax_graph(self.dynamic_alpha * dyn_score, mask_self=True)  # [B,N,N]
        A_self = torch.eye(N, device=z.device, dtype=z.dtype).view(1, N, N).expand(B, -1, -1)

        has_phy = A_phy is not None
        if has_phy:
            if A_phy.dim() == 2:
                A_phy = A_phy.to(device=z.device, dtype=z.dtype).unsqueeze(0).expand(B, -1, -1)
            elif A_phy.dim() == 3:
                A_phy = A_phy.to(device=z.device, dtype=z.dtype)
                if A_phy.shape[0] == 1:
                    A_phy = A_phy.expand(B, -1, -1)
            else:
                raise ValueError(f"A_phy must be [N,N] or [B,N,N], got {tuple(A_phy.shape)}")
            if A_phy.shape[-2:] != (N, N):
                raise ValueError(f"A_phy shape must end with [{N},{N}], got {tuple(A_phy.shape)}")
            A_phy = self._row_softmax_graph(A_phy, mask_self=True)
        else:
            A_phy = torch.zeros(B, N, N, device=z.device, dtype=z.dtype)

        # The center node is inserted explicitly as the first token of every
        # local spatial sequence, so self-loop strength should not consume the
        # neighbor-graph softmax mass used for top-k selection.
        neighbor_logits = torch.stack([self.graph_logits[0], self.graph_logits[1], self.graph_logits[3]])
        neighbor_mask = torch.zeros_like(neighbor_logits)
        if disable_dynamic:
            neighbor_mask[1] = -1e9
        if not has_phy:
            neighbor_mask[2] = -1e9
        neighbor_lambdas = torch.softmax(neighbor_logits + neighbor_mask, dim=0)
        lambda_adp, lambda_dyn, lambda_phy = neighbor_lambdas
        lambda_self = torch.sigmoid(self.graph_logits[2])

        score = (
            lambda_adp * A_adp.unsqueeze(0)
            + lambda_dyn * A_dyn
            + lambda_phy * A_phy
        )

        # Exclude self from neighbor list because center node is inserted explicitly.
        if k == 0:
            topk_idx = torch.empty(B, N, 0, device=z.device, dtype=torch.long)
            topk_attn = torch.empty(B, N, 0, device=z.device, dtype=z.dtype)
        else:
            eye = torch.eye(N, device=z.device, dtype=torch.bool).view(1, N, N)
            score = score.masked_fill(eye, float("-inf"))
            topk_val, topk_idx = torch.topk(score, k=k, dim=-1)
            topk_attn = torch.softmax(topk_val, dim=-1)

        if not self.training:
            self._last_stats = {
                "lambda_adp": float(lambda_adp.detach().cpu()),
                "lambda_dyn": float(lambda_dyn.detach().cpu()),
                "lambda_self": float(lambda_self.detach().cpu()),
                "lambda_phy": float(lambda_phy.detach().cpu()) if has_phy else None,
                "topk": float(k),
            }

        graph_info = {
            "A_dyn": A_dyn,
            "A_adp": A_adp,
            "topk_idx": topk_idx,
            "topk_attn": topk_attn,
            "A_self": A_self,
            "lambda_adp": lambda_adp,
            "lambda_dyn": lambda_dyn,
            "lambda_self": lambda_self,
            "lambda_phy": lambda_phy,
        }
        return topk_idx, topk_attn, graph_info


class GraphGuidedLocalSpatialMamba(nn.Module):
    """
    Graph-guided local spatial Mamba.

    For each center node i, build a short sequence:
        [center_i, neighbor_j1, ..., neighbor_jK]
    and scan this graph-neighborhood sequence with bidirectional Mamba.

    Input:
        z:          [B,T,N,D] or [B,N,D]
        topk_idx:   [B,N,K]
        topk_attn:  [B,N,K]
    Output:
        z_spatial:  same leading shape as z
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        init_gamma: float = 0.1,
        gamma_max: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if self.bidirectional:
            self.mamba_bwd = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.out_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.mamba_bwd = None
            self.out_proj = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.gamma_max = float(gamma_max)
        self.gamma = nn.Parameter(_init_bounded_scale(init_gamma, self.gamma_max))

    def forward(self, z: torch.Tensor, topk_idx: torch.Tensor, topk_attn: torch.Tensor) -> torch.Tensor:
        squeeze_time = False
        if z.dim() == 3:
            z = z.unsqueeze(1)
            squeeze_time = True
        elif z.dim() != 4:
            raise ValueError(f"GraphGuidedLocalSpatialMamba expects z [B,T,N,D] or [B,N,D], got {tuple(z.shape)}")

        B, T, N, D = z.shape
        K = topk_idx.shape[-1]
        if K == 0:
            return z.squeeze(1) if squeeze_time else z

        batch_idx = torch.arange(B, device=z.device).view(B, 1, 1, 1).expand(B, T, N, K)
        time_idx = torch.arange(T, device=z.device).view(1, T, 1, 1).expand(B, T, N, K)
        node_idx = topk_idx.unsqueeze(1).expand(B, T, N, K)
        neighbor = z[batch_idx, time_idx, node_idx]  # [B,T,N,K,D]
        center = z.unsqueeze(3)                      # [B,T,N,1,D]
        seq = torch.cat([center, neighbor], dim=3)   # [B,T,N,K+1,D]
        seq_flat = seq.reshape(B * T * N, K + 1, D)

        y_fwd = self.mamba_fwd(seq_flat)
        if self.bidirectional:
            y_bwd = self.mamba_bwd(torch.flip(seq_flat, dims=[1]))
            y_bwd = torch.flip(y_bwd, dims=[1])
            y = self.out_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        else:
            y = self.out_proj(y_fwd)
        y = self.dropout(y)
        y = self.norm(y + seq_flat)

        y = y.view(B, T, N, K + 1, D)
        center_out = y[:, :, :, 0, :]
        neigh_out = y[:, :, :, 1:, :]
        neigh_agg = (topk_attn.unsqueeze(1).unsqueeze(-1) * neigh_out).sum(dim=3)

        gamma = _bounded_positive_scale(self.gamma, self.gamma_max).to(dtype=z.dtype)
        out = self.norm(z + gamma * (center_out + neigh_agg))
        return out.squeeze(1) if squeeze_time else out


class NonRecurringTemporalGraphSpatialMambaBranch(nn.Module):
    """
    Heavy branch for non-recurring disturbances:
        X_non -> Temporal Mamba -> Dynamic Top-K Graph -> Graph-guided Spatial Mamba -> component output.

    Input:
        x_non: [B,T,N]
        x_ts:  [B,T,2]
    Output:
        delta_y: [B,H,N]
        z_non:   [B,N,D]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        spatial_topk: int = 8,
        spatial_node_dim: int = 16,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        head_layers: int = 1,
        identity_scale_init: float = 0.1,
        identity_scale_max: float = 0.2,
        spatial_gate_max: float = 0.5,
        disable_temporal_mamba: bool = False,
        disable_spatial_mamba: bool = False,
        disable_dynamic_graph: bool = False,
        spatial_mode: str = "local_graph_mamba",
        spatial_time_mode: str = "summary",
        spatial_bidirectional: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.disable_temporal_mamba = bool(disable_temporal_mamba)
        self.disable_spatial_mamba = bool(disable_spatial_mamba)
        self.disable_dynamic_graph = bool(disable_dynamic_graph)
        self.spatial_mode = spatial_mode
        self.spatial_time_mode = spatial_time_mode
        self.spatial_bidirectional = bool(spatial_bidirectional)
        self.spatial_gate_max = float(spatial_gate_max)
        if spatial_mode not in {"local_graph_mamba", "node_order_mamba", "none"}:
            raise ValueError("rnp_spatial_mode must be local_graph_mamba, node_order_mamba, or none.")
        if spatial_time_mode not in {"summary", "all"}:
            raise ValueError("rnp_spatial_time_mode must be summary or all.")

        self.value_proj = nn.Linear(1, d_model)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        self.tod_emb = nn.Parameter(torch.empty(time_of_day_size, d_model))
        self.dow_emb = nn.Parameter(torch.empty(day_of_week_size, d_model))
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.tod_emb)
        nn.init.xavier_uniform_(self.dow_emb)
        self.identity_scale_max = float(identity_scale_max)
        self.identity_scale = nn.Parameter(_init_bounded_scale(identity_scale_init, self.identity_scale_max))
        self.input_norm = nn.LayerNorm(d_model)

        self.temporal_mamba = BiTemporalMambaBranch(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.temporal_agg = TimeAwareAggregation(d_model=d_model, dropout=dropout)

        self.graph_constructor = DynamicTopKGraphConstructor(
            num_nodes=num_nodes,
            d_model=d_model,
            node_emb_dim=spatial_node_dim,
            topk=spatial_topk,
            dynamic_alpha=1.0,
        )
        self.spatial_mamba = GraphGuidedLocalSpatialMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            init_gamma=0.05,
            gamma_max=0.2,
            bidirectional=spatial_bidirectional,
        )
        if spatial_mode == "node_order_mamba":
            self.node_order_mamba_fwd = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.node_order_mamba_bwd = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.node_order_proj = nn.Linear(d_model * 2, d_model)
            self.node_order_norm = nn.LayerNorm(d_model)
            self.node_order_dropout = nn.Dropout(dropout)
        else:
            self.node_order_mamba_fwd = None
            self.node_order_mamba_bwd = None
            self.node_order_proj = None
            self.node_order_norm = None
            self.node_order_dropout = None

        self.ts_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.ts_gate[-2].bias, -2.0)
        self.output_norm = nn.LayerNorm(d_model)
        self.pred_head = STIDStylePostFusionHead(
            d_model=d_model,
            output_len=output_len,
            num_layers=head_layers,
            dropout=dropout,
        )
        self._last_stats = {}

    def build_identity(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        tid = x_ts[:, :, 0].long().clamp(0, self.time_of_day_size - 1)
        dow = x_ts[:, :, 1].long().clamp(0, self.day_of_week_size - 1)

        node_emb = self.node_emb.view(1, 1, N, -1).expand(B, T, -1, -1)
        tid_emb = self.tod_emb[tid].unsqueeze(2).expand(-1, -1, N, -1)
        dow_emb = self.dow_emb[dow].unsqueeze(2).expand(-1, -1, N, -1)
        return node_emb + tid_emb + dow_emb

    def _node_order_spatial_mamba(self, h_t: torch.Tensor) -> torch.Tensor:
        # Ablation only: scan raw node order. The default local graph mode is more
        # traffic-aware and should be used for main results.
        if self.node_order_mamba_fwd is None:
            raise RuntimeError("node_order_mamba modules are only initialized when rnp_spatial_mode=node_order_mamba.")
        B, T, N, D = h_t.shape
        x_seq = h_t.reshape(B * T, N, D)
        y_fwd = self.node_order_mamba_fwd(x_seq)
        y_bwd = self.node_order_mamba_bwd(torch.flip(x_seq, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])
        y = self.node_order_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        y = self.node_order_dropout(y)
        y = self.node_order_norm(y + x_seq)
        return y.view(B, T, N, D)

    def forward(self, x_non: torch.Tensor, x_ts: torch.Tensor, A_phy: torch.Tensor = None):
        if x_non.dim() != 3:
            raise ValueError(f"NonRecurringTemporalGraphSpatialMambaBranch expects x_non [B,T,N], got {tuple(x_non.shape)}")
        B, T, N = x_non.shape
        if T != self.input_len or N != self.num_nodes:
            raise ValueError(f"Expected [B,{self.input_len},{self.num_nodes}], got {tuple(x_non.shape)}")

        h = self.value_proj(x_non.unsqueeze(-1))
        identity_scale = _bounded_positive_scale(self.identity_scale, self.identity_scale_max).to(dtype=h.dtype)
        h = h + identity_scale * self.build_identity(x_non, x_ts)
        h = self.input_norm(h)

        h_t = h if self.disable_temporal_mamba else self.temporal_mamba(h)  # [B,T,N,D]
        z_t = self.temporal_agg(h_t)        # [B,N,D]

        if self.disable_spatial_mamba or self.spatial_mode == "none":
            h_s = h_t
            z_s = z_t
            graph_info = None
        elif self.spatial_mode == "node_order_mamba":
            h_s = self._node_order_spatial_mamba(h_t)
            z_s = self.temporal_agg(h_s)
            graph_info = None
        else:
            topk_idx, topk_attn, graph_info = self.graph_constructor(
                z_t,
                A_phy=A_phy,
                disable_dynamic=self.disable_dynamic_graph,
            )
            if self.spatial_time_mode == "all":
                h_s = self.spatial_mamba(h_t, topk_idx, topk_attn)  # [B,T,N,D]
                z_s = self.temporal_agg(h_s)                        # [B,N,D]
            else:
                z_s = self.spatial_mamba(z_t, topk_idx, topk_attn)  # [B,N,D]
                h_s = h_t

        gate = self.spatial_gate_max * self.ts_gate(torch.cat([z_t, z_s], dim=-1))
        z_non = self.output_norm(z_t + gate * z_s)
        y_non = self.pred_head(z_non)

        if not self.training:
            self._last_stats = {
                "ts_gate_mean": float(gate.detach().mean().cpu()),
                "ts_gate_min": float(gate.detach().min().cpu()),
                "ts_gate_max": float(gate.detach().max().cpu()),
                "spatial_gamma": float(_bounded_positive_scale(self.spatial_mamba.gamma.detach(), self.spatial_mamba.gamma_max).cpu()),
                "non_identity_scale": float(_bounded_positive_scale(self.identity_scale.detach(), self.identity_scale_max).cpu()),
                "spatial_mode": self.spatial_mode,
                "spatial_time_mode": self.spatial_time_mode,
                "spatial_bidirectional": float(self.spatial_bidirectional),
                "disable_temporal_mamba": float(self.disable_temporal_mamba),
                "disable_spatial_mamba": float(self.disable_spatial_mamba),
                "disable_dynamic_graph": float(self.disable_dynamic_graph),
            }
            if graph_info is None:
                self.graph_constructor._last_stats = {
                    "lambda_adp": 0.0,
                    "lambda_dyn": 0.0,
                    "lambda_self": 0.0,
                    "lambda_phy": None,
                    "topk": 0.0,
                }

        return y_non, z_non


class MultiScaleDepthwiseTemporalConv(nn.Module):
    """
    Lightweight multi-scale temporal convolution.

    Input:
        h: [B, T, N, D]
    Output:
        h_ms: [B, T, N, D, Q]
    """

    def __init__(self, d_model: int, kernels=(1, 3, 5)):
        super().__init__()
        if not kernels:
            raise ValueError("kernels must not be empty.")
        self.kernels = tuple(int(k) for k in kernels)
        for k in self.kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError("All temporal kernels must be positive odd integers.")

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=k,
                groups=d_model,
                bias=True,
            )
            for k in self.kernels
        ])
        self.scale_emb = nn.Parameter(torch.zeros(len(self.kernels), d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 4:
            raise ValueError(f"MultiScaleDepthwiseTemporalConv expects h [B,T,N,D], got {tuple(h.shape)}")

        B, T, N, D = h.shape
        x = h.permute(0, 2, 3, 1).contiguous().view(B * N, D, T)

        outputs = []
        for conv, kernel in zip(self.convs, self.kernels):
            pad = (kernel - 1) // 2
            x_in = F.pad(x, (pad, pad), mode="replicate") if pad > 0 else x
            y = conv(x_in)
            y = y.view(B, N, D, T).permute(0, 3, 1, 2).contiguous()
            outputs.append(y)

        h_ms = torch.stack(outputs, dim=-1)
        scale_bias = self.scale_emb.transpose(0, 1).view(1, 1, 1, D, len(self.kernels))
        return h_ms + scale_bias


class CausalScaleFusion(nn.Module):
    """
    Lightweight scale-only attention.

    Query scale i can attend to itself and coarser scales j >= i.
    It never attends over T or N. The implementation is intentionally
    manual instead of nn.MultiheadAttention because Q is tiny and the
    fused CUDA SDPA path can fail for very large B*T*N with Q=3.
    """

    def __init__(self, d_model: int, num_scales: int, num_heads: int = 1, dropout: float = 0.15):
        super().__init__()
        if num_scales <= 0:
            raise ValueError("num_scales must be positive.")
        if num_heads != 1:
            raise ValueError("CausalScaleFusion uses a single lightweight scale head.")
        self.num_scales = int(num_scales)
        self.scale = float(d_model) ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        mask = torch.zeros(self.num_scales, self.num_scales, dtype=torch.bool)
        for i in range(self.num_scales):
            for j in range(self.num_scales):
                if j < i:
                    mask[i, j] = True
        self.register_buffer("scale_mask", mask, persistent=False)

    def forward(self, h_ms: torch.Tensor) -> torch.Tensor:
        if h_ms.dim() != 5:
            raise ValueError(f"CausalScaleFusion expects h_ms [B,T,N,D,Q], got {tuple(h_ms.shape)}")

        B, T, N, D, Q = h_ms.shape
        if Q != self.num_scales:
            raise ValueError(f"Expected num_scales={self.num_scales}, got Q={Q}")

        h_qd = h_ms.permute(0, 1, 2, 4, 3).contiguous()  # [B,T,N,Q,D]
        q = self.q_proj(h_qd)
        k = self.k_proj(h_qd)
        v = self.v_proj(h_qd)

        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,T,N,Q,Q]
        mask = self.scale_mask.view(1, 1, 1, Q, Q)
        score = score.masked_fill(mask, torch.finfo(score.dtype).min)
        attn = torch.softmax(score, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [B,T,N,Q,D]
        out = self.out_dropout(self.out_proj(out))
        out = self.norm(h_qd + out)
        return out.permute(0, 1, 2, 4, 3).contiguous()


class LearnableScalePool(nn.Module):
    """
    Learn a global scale mixture over Q lightweight temporal scales.

    Input:
        h_ms: [B, T, N, D, Q]
    Output:
        h: [B, T, N, D]
    """

    def __init__(self, num_scales: int):
        super().__init__()
        if num_scales <= 0:
            raise ValueError("num_scales must be positive.")
        self.num_scales = int(num_scales)
        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))
        self._last_stats = {}

    def forward(self, h_ms: torch.Tensor) -> torch.Tensor:
        if h_ms.dim() != 5:
            raise ValueError(f"LearnableScalePool expects h_ms [B,T,N,D,Q], got {tuple(h_ms.shape)}")
        Q = h_ms.shape[-1]
        if Q != self.num_scales:
            raise ValueError(f"Expected num_scales={self.num_scales}, got Q={Q}")

        weight = torch.softmax(self.scale_logits, dim=-1)
        h = (h_ms * weight.view(1, 1, 1, 1, Q)).sum(dim=-1)
        if not self.training:
            stats = {
                "scale_weight_mean": float(weight.detach().mean().cpu()),
            }
            for i in range(min(Q, 8)):
                stats[f"lite_scale_weight_{i}"] = float(weight[i].detach().cpu())
            self._last_stats = stats
        return h


class LiteNonRecurringScaleGraphBranch(nn.Module):
    """
    Lightweight non-recurring branch for v6.2-lite.

    It replaces temporal Mamba with cheap multi-scale depthwise temporal
    convolutions, then keeps one dynamic graph construction and one
    graph-guided local spatial Mamba for disturbance propagation.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        d_model: int = 64,
        dropout: float = 0.15,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        spatial_topk: int = 8,
        spatial_node_dim: int = 16,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        head_layers: int = 1,
        identity_scale_init: float = 0.1,
        identity_scale_max: float = 0.2,
        spatial_alpha_init: float = 0.1,
        spatial_alpha_max: float = 0.3,
        kernels=(1, 3, 5),
        disable_spatial_mamba: bool = False,
        disable_dynamic_graph: bool = False,
        spatial_mode: str = "local_graph_mamba",
        spatial_bidirectional: bool = False,
    ):
        super().__init__()
        if spatial_mode not in {"local_graph_mamba", "none"}:
            raise ValueError("v6.2-lite supports rnp_spatial_mode=local_graph_mamba or none.")

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.time_of_day_size = int(time_of_day_size)
        self.day_of_week_size = int(day_of_week_size)
        self.identity_scale_max = float(identity_scale_max)
        self.spatial_alpha_max = float(spatial_alpha_max)
        self.disable_spatial_mamba = bool(disable_spatial_mamba)
        self.disable_dynamic_graph = bool(disable_dynamic_graph)
        self.spatial_mode = spatial_mode
        self.spatial_bidirectional = bool(spatial_bidirectional)

        self.value_proj = nn.Linear(1, d_model)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        self.tod_emb = nn.Parameter(torch.empty(time_of_day_size, d_model))
        self.dow_emb = nn.Parameter(torch.empty(day_of_week_size, d_model))
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.tod_emb)
        nn.init.xavier_uniform_(self.dow_emb)
        self.identity_scale = nn.Parameter(_init_bounded_scale(identity_scale_init, self.identity_scale_max))
        self.input_norm = nn.LayerNorm(d_model)

        self.ms_conv = MultiScaleDepthwiseTemporalConv(d_model=d_model, kernels=kernels)
        self.scale_fusion = CausalScaleFusion(
            d_model=d_model,
            num_scales=len(tuple(kernels)),
            num_heads=1,
            dropout=dropout,
        )
        self.scale_pool = LearnableScalePool(num_scales=len(tuple(kernels)))
        self.temporal_agg = TimeAwareAggregation(d_model=d_model, dropout=dropout)

        self.graph_constructor = DynamicTopKGraphConstructor(
            num_nodes=num_nodes,
            d_model=d_model,
            node_emb_dim=spatial_node_dim,
            topk=spatial_topk,
            dynamic_alpha=1.0,
        )
        self.spatial_mamba = GraphGuidedLocalSpatialMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            init_gamma=0.05,
            gamma_max=0.2,
            bidirectional=spatial_bidirectional,
        )

        self.spatial_alpha = nn.Parameter(_init_bounded_scale(spatial_alpha_init, self.spatial_alpha_max))
        self.output_norm = nn.LayerNorm(d_model)
        self.pred_head = STIDStylePostFusionHead(
            d_model=d_model,
            output_len=output_len,
            num_layers=head_layers,
            dropout=dropout,
        )
        self._last_stats = {}

    def build_identity(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        tid = x_ts[:, :, 0].long().clamp(0, self.time_of_day_size - 1)
        dow = x_ts[:, :, 1].long().clamp(0, self.day_of_week_size - 1)
        node_emb = self.node_emb.view(1, 1, N, -1).expand(B, T, -1, -1)
        tid_emb = self.tod_emb[tid].unsqueeze(2).expand(-1, -1, N, -1)
        dow_emb = self.dow_emb[dow].unsqueeze(2).expand(-1, -1, N, -1)
        return node_emb + tid_emb + dow_emb

    def forward(self, x_non: torch.Tensor, x_ts: torch.Tensor, A_phy: torch.Tensor = None):
        if x_non.dim() != 3:
            raise ValueError(f"LiteNonRecurringScaleGraphBranch expects x_non [B,T,N], got {tuple(x_non.shape)}")
        if x_ts.dim() != 3 or x_ts.shape[-1] < 2:
            raise ValueError(f"x_ts must be [B,T,2], got {tuple(x_ts.shape)}")

        B, T, N = x_non.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        h = self.value_proj(x_non.unsqueeze(-1))
        identity_scale = _bounded_positive_scale(self.identity_scale, self.identity_scale_max).to(dtype=h.dtype)
        h = self.input_norm(h + identity_scale * self.build_identity(x_non, x_ts))

        h_ms = self.ms_conv(h)
        h_ms = self.scale_fusion(h_ms)
        h_t = self.scale_pool(h_ms)
        z_t = self.temporal_agg(h_t)

        graph_info = None
        if self.disable_spatial_mamba or self.spatial_mode == "none":
            z_s = torch.zeros_like(z_t)
            self.graph_constructor._last_stats = {
                "lambda_adp": 0.0,
                "lambda_dyn": 0.0,
                "lambda_self": 0.0,
                "lambda_phy": None,
                "topk": 0.0,
            }
        else:
            topk_idx, topk_attn, graph_info = self.graph_constructor(
                z_t,
                A_phy=A_phy,
                disable_dynamic=self.disable_dynamic_graph,
            )
            z_s = self.spatial_mamba(z_t, topk_idx, topk_attn)

        alpha = _bounded_positive_scale(self.spatial_alpha, self.spatial_alpha_max).to(dtype=z_t.dtype)
        z_non = self.output_norm(z_t + alpha * z_s)
        y_non = self.pred_head(z_non)

        if not self.training:
            self._last_stats = {
                "lite_scale_weight_mean": getattr(self.scale_pool, "_last_stats", {}).get("scale_weight_mean"),
                "lite_scale_weight_0": getattr(self.scale_pool, "_last_stats", {}).get("lite_scale_weight_0"),
                "lite_scale_weight_1": getattr(self.scale_pool, "_last_stats", {}).get("lite_scale_weight_1"),
                "lite_scale_weight_2": getattr(self.scale_pool, "_last_stats", {}).get("lite_scale_weight_2"),
                "lite_spatial_alpha": float(alpha.detach().cpu()),
                "lite_y_non_std": float(y_non.detach().std(unbiased=False).cpu()),
                "lite_z_t_std": float(z_t.detach().std(unbiased=False).cpu()),
                "lite_z_s_std": float(z_s.detach().std(unbiased=False).cpu()),
                "lite_z_non_std": float(z_non.detach().std(unbiased=False).cpu()),
                "lite_non_identity_scale": float(identity_scale.detach().cpu()),
                "non_identity_scale": float(identity_scale.detach().cpu()),
                "spatial_gamma": float(
                    _bounded_positive_scale(
                        self.spatial_mamba.gamma.detach(),
                        self.spatial_mamba.gamma_max,
                    ).cpu()
                ),
                "spatial_mode": self.spatial_mode,
                "spatial_time_mode": "summary",
                "spatial_bidirectional": float(self.spatial_bidirectional),
                "disable_temporal_mamba": 1.0,
                "disable_spatial_mamba": float(self.disable_spatial_mamba),
                "disable_dynamic_graph": float(self.disable_dynamic_graph),
            }
            if graph_info is None:
                self._last_stats.update(self.graph_constructor._last_stats)

        return y_non, z_non


class RNPMambaV6DualBranch(nn.Module):
    """
    RNP-Mamba-v6 dual-branch implementation.

    Clean model structure:
        X -> recurring/non-recurring decomposition
          -> recurring branch: STID-style lightweight forecaster -> Y_rec
          -> non-recurring branch: Temporal -> Graph -> Spatial Mamba -> Y_non
          -> gated component fusion: Y_hat = Y_rec + gate * scale * Y_non

    This removes the independent STIDCore backbone used in v5.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        d_model: int = 64,
        decomp_kernel: int = 5,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        node_dim: int = 32,  # kept for train.py compatibility; not used directly
        temp_dim_tid: int = 32,  # kept for train.py compatibility; not used directly
        temp_dim_diw: int = 32,  # kept for train.py compatibility; not used directly
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        spatial_topk: int = 8,
        spatial_node_dim: int = 16,
        rec_head_layers: int = 2,
        non_head_layers: int = 1,
        residual_scale_init: float = 0.2,
        residual_gate_bias_init: float = -0.6,
        decomp_mode: str = "st_embed",
        disable_nonrec: bool = False,
        disable_rec: bool = False,
        disable_temporal_mamba: bool = False,
        disable_spatial_mamba: bool = False,
        disable_dynamic_graph: bool = False,
        spatial_mode: str = "local_graph_mamba",
        spatial_time_mode: str = "summary",
        spatial_bidirectional: bool = False,
        use_component_energy_gate: bool = False,
        component_energy_weight: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.disable_nonrec = bool(disable_nonrec)
        self.disable_rec = bool(disable_rec)
        self.use_component_energy_gate = bool(use_component_energy_gate)
        self.component_energy_weight = float(component_energy_weight)

        # Make a stable kernel set around the configured kernel.
        k0 = int(decomp_kernel)
        if k0 % 2 == 0:
            k0 += 1
        kernels = sorted(set([3, k0, min(input_len if input_len % 2 == 1 else input_len - 1, 7)]))
        kernels = tuple(k for k in kernels if k > 0)

        if decomp_mode == "st_embed":
            self.decomposition = STEmbeddingGuidedRNPDecomposition(
                num_nodes=num_nodes,
                input_len=input_len,
                time_of_day_size=time_of_day_size,
                day_of_week_size=day_of_week_size,
                hidden_dim=max(32, d_model),
                gate_min=0.05,
                gate_max=0.95,
                init_rec_ratio=0.75,
                dropout=dropout,
            )
        else:
            self.decomposition = MultiScalePeriodicRNPDecomposition(
                num_nodes=num_nodes,
                input_len=input_len,
                kernels=kernels,
                time_of_day_size=time_of_day_size,
                day_of_week_size=day_of_week_size,
                hidden_dim=max(16, d_model // 4),
                proto_scale_init=0.1,
                mode=decomp_mode,
            )

        self.recurring_branch = RecurringSTIDLightBranch(
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            d_model=d_model,
            dropout=dropout,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
            head_layers=rec_head_layers,
            identity_scale_init=0.3,
            identity_scale_max=0.8,
        )

        self.nonrecurring_branch = NonRecurringTemporalGraphSpatialMambaBranch(
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            spatial_topk=spatial_topk,
            spatial_node_dim=spatial_node_dim,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
            head_layers=non_head_layers,
            identity_scale_init=0.1,
            identity_scale_max=0.2,
            spatial_gate_max=0.5,
            disable_temporal_mamba=disable_temporal_mamba,
            disable_spatial_mamba=disable_spatial_mamba,
            disable_dynamic_graph=disable_dynamic_graph,
            spatial_mode=spatial_mode,
            spatial_time_mode=spatial_time_mode,
            spatial_bidirectional=spatial_bidirectional,
        )

        # Horizon-aware component gate. It uses branch states, not a separate STIDCore.
        self.horizon_emb = nn.Parameter(torch.empty(output_len, d_model))
        nn.init.xavier_uniform_(self.horizon_emb)

        self.component_gate = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.component_gate[-1].bias, float(residual_gate_bias_init))
        if self.use_component_energy_gate:
            energy_hidden = max(8, d_model // 8)
            self.component_energy_gate = nn.Sequential(
                nn.Linear(2, energy_hidden),
                nn.GELU(),
                nn.Linear(energy_hidden, 1),
            )
            nn.init.zeros_(self.component_energy_gate[-1].weight)
            nn.init.zeros_(self.component_energy_gate[-1].bias)
        else:
            self.component_energy_gate = None

        self.component_scale_max = 1.0
        self.component_horizon_scale_max = 1.0
        self.component_scale = nn.Parameter(_init_bounded_scale(residual_scale_init, self.component_scale_max))
        self.component_horizon_scale = nn.Parameter(
            _init_bounded_scale(0.75, self.component_horizon_scale_max).repeat(output_len)
        )

        # For optional diagnostics.
        self._last_stats = {}
        self._last_components = {}

    def _build_horizon_gate(self, z_rec: torch.Tensor, z_non: torch.Tensor) -> torch.Tensor:
        # z_rec/z_non: [B,N,D]
        B, N, D = z_rec.shape
        h_emb = self.horizon_emb.view(1, self.output_len, 1, D).expand(B, -1, N, -1)
        z_rec_h = z_rec.unsqueeze(1).expand(-1, self.output_len, -1, -1)
        z_non_h = z_non.unsqueeze(1).expand(-1, self.output_len, -1, -1)
        gate_input = torch.cat([z_rec_h, z_non_h, h_emb], dim=-1)  # [B,H,N,3D]
        gate = torch.sigmoid(self.component_gate(gate_input)).squeeze(-1)  # [B,H,N]
        return gate

    def _build_energy_gate(self, x_non: torch.Tensor) -> torch.Tensor:
        if self.component_energy_gate is None:
            return None
        mean_abs = x_non.abs().mean(dim=1)
        std = x_non.std(dim=1, unbiased=False)
        energy_feat = torch.stack([mean_abs, std], dim=-1)  # [B,N,2]
        return torch.sigmoid(self.component_energy_gate(energy_feat)).squeeze(-1).unsqueeze(1)

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor, A_phy: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.dim() != 3:
            raise ValueError(f"RNPMambaV6DualBranch expects x shape [B,T,N] or [B,T,N,1], got {tuple(x.shape)}")
        B, T, N = x.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        x_rec, x_non = self.decomposition(x, x_ts)
        if self.disable_rec:
            y_rec = x.new_zeros(B, self.output_len, N)
            z_rec = x.new_zeros(B, N, self.d_model)
        else:
            y_rec, z_rec = self.recurring_branch(x_rec, x_ts)

        if self.disable_nonrec:
            y_non = x.new_zeros(B, self.output_len, N)
            z_non = x.new_zeros(B, N, self.d_model)
        else:
            y_non, z_non = self.nonrecurring_branch(x_non, x_ts, A_phy=A_phy)

        base_gate = self._build_horizon_gate(z_rec, z_non)
        energy_gate = self._build_energy_gate(x_non)
        if energy_gate is None:
            gate = base_gate
        else:
            energy_weight = min(max(self.component_energy_weight, 0.0), 1.0)
            gate = (1.0 - energy_weight) * base_gate + energy_weight * energy_gate

        component_scale = _bounded_positive_scale(self.component_scale, self.component_scale_max).to(dtype=x.dtype)
        horizon_scale = _bounded_positive_scale(
            self.component_horizon_scale, self.component_horizon_scale_max
        ).to(dtype=x.dtype).view(1, self.output_len, 1)
        residual_gain = gate * component_scale * horizon_scale
        correction = residual_gain * y_non
        y_hat = y_rec + correction
        self._last_components = {
            "y_rec": y_rec,
            "y_non": y_non,
            "correction": correction,
            "residual_gain": residual_gain,
            "x_non": x_non,
        }

        if not self.training:
            horizon_scale_detached = horizon_scale.detach()
            base_gate_detached = base_gate.detach()
            energy_gate_detached = energy_gate.detach() if energy_gate is not None else None
            residual_gain_detached = residual_gain.detach()
            self._last_stats = {
                "fusion_mode": "rnp_v6_dual_branch",
                "gate_mean": float(gate.detach().mean().cpu()),
                "gate_std": float(gate.detach().std(unbiased=False).cpu()),
                "gate_min": float(gate.detach().min().cpu()),
                "gate_max": float(gate.detach().max().cpu()),
                "base_gate_mean": float(base_gate_detached.mean().cpu()),
                "energy_gate_mean": float(energy_gate_detached.mean().cpu()) if energy_gate_detached is not None else None,
                "component_scale": float(component_scale.detach().cpu()),
                "component_horizon_scale_mean": float(horizon_scale_detached.mean().cpu()),
                "component_horizon_scale_min": float(horizon_scale_detached.min().cpu()),
                "component_horizon_scale_max": float(horizon_scale_detached.max().cpu()),
                "residual_gain_mean": float(residual_gain_detached.mean().cpu()),
                "residual_gain_max": float(residual_gain_detached.max().cpu()),
                "y_rec_std": float(y_rec.detach().std(unbiased=False).cpu()),
                "y_non_std": float(y_non.detach().std(unbiased=False).cpu()),
                "y_hat_std": float(y_hat.detach().std(unbiased=False).cpu()),
                "correction_abs_mean": float(correction.detach().abs().mean().cpu()),
                "correction_std": float(correction.detach().std(unbiased=False).cpu()),
                "rec_identity_scale": float(
                    _bounded_positive_scale(
                        self.recurring_branch.identity_scale.detach(),
                        self.recurring_branch.identity_scale_max,
                    ).cpu()
                ),
                "disable_rec": float(self.disable_rec),
                "disable_nonrec": float(self.disable_nonrec),
            }

        return y_hat

    def get_component_outputs(self):
        return self._last_components

    def get_gate_stats(self):
        stats = {
            "fusion_mode": "rnp_v6_dual_branch",
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "gate_min": 0.0,
            "gate_max": 0.0,
            "base_gate_mean": None,
            "energy_gate_mean": None,
            "x_rec_std": None,
            "x_non_std": None,
            "rec_gate_mean": None,
            "rec_gate_min": None,
            "rec_gate_max": None,
            "scale_weight_mean": None,
            "proto_scale": None,
            "lambda_adp": 0.0,
            "lambda_dyn": 0.0,
            "lambda_self": 0.0,
            "lambda_phy": None,
            "ts_gate_mean": None,
            "ts_gate_min": None,
            "ts_gate_max": None,
            "component_scale": None,
            "component_horizon_scale_mean": None,
            "component_horizon_scale_min": None,
            "component_horizon_scale_max": None,
            "residual_gain_mean": None,
            "residual_gain_max": None,
            "y_rec_std": None,
            "y_non_std": None,
            "y_hat_std": None,
            "correction_abs_mean": None,
            "correction_std": None,
            "rec_identity_scale": None,
            "non_identity_scale": None,
            "spatial_gamma": None,
        }
        stats.update(self._last_stats)
        stats.update(getattr(self.decomposition, "_last_stats", {}))
        stats.update(getattr(self.nonrecurring_branch.graph_constructor, "_last_stats", {}))
        stats.update(getattr(self.nonrecurring_branch, "_last_stats", {}))
        return stats


class RNPMambaV62Lite(nn.Module):
    """
    RNP-Mamba-v6.2-lite.

    It keeps the v6 recurring branch, decomposition, and horizon-aware
    residual fusion, but replaces the non-recurring temporal Mamba path
    with a lightweight multi-scale temporal convolution path.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        d_model: int = 64,
        decomp_kernel: int = 5,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        node_dim: int = 32,
        temp_dim_tid: int = 32,
        temp_dim_diw: int = 32,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        spatial_topk: int = 8,
        spatial_node_dim: int = 16,
        rec_head_layers: int = 2,
        non_head_layers: int = 1,
        residual_scale_init: float = 0.2,
        residual_gate_bias_init: float = -0.6,
        decomp_mode: str = "st_embed",
        disable_nonrec: bool = False,
        disable_rec: bool = False,
        disable_temporal_mamba: bool = False,
        disable_spatial_mamba: bool = False,
        disable_dynamic_graph: bool = False,
        spatial_mode: str = "local_graph_mamba",
        spatial_time_mode: str = "summary",
        spatial_bidirectional: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.disable_nonrec = bool(disable_nonrec)
        self.disable_rec = bool(disable_rec)
        self.disable_temporal_mamba = bool(disable_temporal_mamba)

        if spatial_time_mode != "summary":
            raise ValueError("v6.2-lite only supports rnp_spatial_time_mode=summary.")

        k0 = int(decomp_kernel)
        if k0 % 2 == 0:
            k0 += 1
        kernels = sorted(set([3, k0, min(input_len if input_len % 2 == 1 else input_len - 1, 7)]))
        kernels = tuple(k for k in kernels if k > 0)

        if decomp_mode == "st_embed":
            self.decomposition = STEmbeddingGuidedRNPDecomposition(
                num_nodes=num_nodes,
                input_len=input_len,
                time_of_day_size=time_of_day_size,
                day_of_week_size=day_of_week_size,
                hidden_dim=max(32, d_model),
                gate_min=0.05,
                gate_max=0.95,
                init_rec_ratio=0.75,
                dropout=dropout,
            )
        else:
            self.decomposition = MultiScalePeriodicRNPDecomposition(
                num_nodes=num_nodes,
                input_len=input_len,
                kernels=kernels,
                time_of_day_size=time_of_day_size,
                day_of_week_size=day_of_week_size,
                hidden_dim=max(16, d_model // 4),
                proto_scale_init=0.1,
                mode=decomp_mode,
            )

        self.recurring_branch = RecurringSTIDLightBranch(
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            d_model=d_model,
            dropout=dropout,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
            head_layers=rec_head_layers,
            identity_scale_init=0.3,
            identity_scale_max=0.8,
        )

        self.nonrecurring_branch = LiteNonRecurringScaleGraphBranch(
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            d_model=d_model,
            dropout=dropout,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
            spatial_topk=spatial_topk,
            spatial_node_dim=spatial_node_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_layers=non_head_layers,
            identity_scale_init=0.1,
            identity_scale_max=0.2,
            spatial_alpha_init=0.1,
            spatial_alpha_max=0.3,
            kernels=(1, 3, 5),
            disable_spatial_mamba=disable_spatial_mamba,
            disable_dynamic_graph=disable_dynamic_graph,
            spatial_mode=spatial_mode,
            spatial_bidirectional=spatial_bidirectional,
        )

        self.horizon_emb = nn.Parameter(torch.empty(output_len, d_model))
        nn.init.xavier_uniform_(self.horizon_emb)

        self.component_gate = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.component_gate[-1].bias, float(residual_gate_bias_init))

        self.component_scale_max = 1.0
        self.component_horizon_scale_max = 1.0
        self.component_scale = nn.Parameter(_init_bounded_scale(residual_scale_init, self.component_scale_max))
        self.component_horizon_scale = nn.Parameter(
            _init_bounded_scale(0.75, self.component_horizon_scale_max).repeat(output_len)
        )

        self._last_stats = {}
        self._last_components = {}

    def _build_horizon_gate(self, z_rec: torch.Tensor, z_non: torch.Tensor) -> torch.Tensor:
        B, N, D = z_rec.shape
        h_emb = self.horizon_emb.view(1, self.output_len, 1, D).expand(B, -1, N, -1)
        z_rec_h = z_rec.unsqueeze(1).expand(-1, self.output_len, -1, -1)
        z_non_h = z_non.unsqueeze(1).expand(-1, self.output_len, -1, -1)
        gate_input = torch.cat([z_rec_h, z_non_h, h_emb], dim=-1)
        return torch.sigmoid(self.component_gate(gate_input)).squeeze(-1)

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor, A_phy: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.dim() != 3:
            raise ValueError(f"RNPMambaV62Lite expects x shape [B,T,N] or [B,T,N,1], got {tuple(x.shape)}")
        B, T, N = x.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        x_rec, x_non = self.decomposition(x, x_ts)
        if self.disable_rec:
            y_rec = x.new_zeros(B, self.output_len, N)
            z_rec = x.new_zeros(B, N, self.d_model)
        else:
            y_rec, z_rec = self.recurring_branch(x_rec, x_ts)

        if self.disable_nonrec:
            y_non = x.new_zeros(B, self.output_len, N)
            z_non = x.new_zeros(B, N, self.d_model)
        else:
            y_non, z_non = self.nonrecurring_branch(x_non, x_ts, A_phy=A_phy)

        gate = self._build_horizon_gate(z_rec, z_non)
        component_scale = _bounded_positive_scale(self.component_scale, self.component_scale_max).to(dtype=x.dtype)
        horizon_scale = _bounded_positive_scale(
            self.component_horizon_scale, self.component_horizon_scale_max
        ).to(dtype=x.dtype).view(1, self.output_len, 1)
        residual_gain = gate * component_scale * horizon_scale
        correction = residual_gain * y_non
        y_hat = y_rec + correction

        self._last_components = {
            "y_rec": y_rec,
            "y_non": y_non,
            "correction": correction,
            "residual_gain": residual_gain,
            "x_non": x_non,
        }

        if not self.training:
            horizon_scale_detached = horizon_scale.detach()
            residual_gain_detached = residual_gain.detach()
            self._last_stats = {
                "fusion_mode": "rnp_v62_lite",
                "gate_mean": float(gate.detach().mean().cpu()),
                "gate_std": float(gate.detach().std(unbiased=False).cpu()),
                "gate_min": float(gate.detach().min().cpu()),
                "gate_max": float(gate.detach().max().cpu()),
                "base_gate_mean": float(gate.detach().mean().cpu()),
                "energy_gate_mean": None,
                "component_scale": float(component_scale.detach().cpu()),
                "component_horizon_scale_mean": float(horizon_scale_detached.mean().cpu()),
                "component_horizon_scale_min": float(horizon_scale_detached.min().cpu()),
                "component_horizon_scale_max": float(horizon_scale_detached.max().cpu()),
                "residual_gain_mean": float(residual_gain_detached.mean().cpu()),
                "residual_gain_max": float(residual_gain_detached.max().cpu()),
                "y_rec_std": float(y_rec.detach().std(unbiased=False).cpu()),
                "y_non_std": float(y_non.detach().std(unbiased=False).cpu()),
                "y_hat_std": float(y_hat.detach().std(unbiased=False).cpu()),
                "correction_abs_mean": float(correction.detach().abs().mean().cpu()),
                "correction_std": float(correction.detach().std(unbiased=False).cpu()),
                "rec_identity_scale": float(
                    _bounded_positive_scale(
                        self.recurring_branch.identity_scale.detach(),
                        self.recurring_branch.identity_scale_max,
                    ).cpu()
                ),
                "disable_rec": float(self.disable_rec),
                "disable_nonrec": float(self.disable_nonrec),
                "disable_temporal_mamba": 1.0,
            }

        return y_hat

    def get_component_outputs(self):
        return self._last_components

    def get_gate_stats(self):
        stats = {
            "fusion_mode": "rnp_v62_lite",
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "gate_min": 0.0,
            "gate_max": 0.0,
            "base_gate_mean": None,
            "energy_gate_mean": None,
            "x_rec_std": None,
            "x_non_std": None,
            "rec_gate_mean": None,
            "rec_gate_min": None,
            "rec_gate_max": None,
            "scale_weight_mean": None,
            "proto_scale": None,
            "lambda_adp": 0.0,
            "lambda_dyn": 0.0,
            "lambda_self": 0.0,
            "lambda_phy": None,
            "ts_gate_mean": None,
            "ts_gate_min": None,
            "ts_gate_max": None,
            "component_scale": None,
            "component_horizon_scale_mean": None,
            "component_horizon_scale_min": None,
            "component_horizon_scale_max": None,
            "residual_gain_mean": None,
            "residual_gain_max": None,
            "y_rec_std": None,
            "y_non_std": None,
            "y_hat_std": None,
            "correction_abs_mean": None,
            "correction_std": None,
            "rec_identity_scale": None,
            "non_identity_scale": None,
            "spatial_gamma": None,
            "lite_scale_weight_mean": None,
            "lite_scale_weight_0": None,
            "lite_scale_weight_1": None,
            "lite_scale_weight_2": None,
            "lite_spatial_alpha": None,
            "lite_y_non_std": None,
            "lite_z_t_std": None,
            "lite_z_s_std": None,
            "lite_z_non_std": None,
            "lite_non_identity_scale": None,
        }
        stats.update(self._last_stats)
        stats.update(getattr(self.decomposition, "_last_stats", {}))
        stats.update(getattr(self.nonrecurring_branch.graph_constructor, "_last_stats", {}))
        stats.update(getattr(self.nonrecurring_branch, "_last_stats", {}))
        return stats
