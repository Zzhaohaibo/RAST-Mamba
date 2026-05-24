import math
from typing import Any, Dict

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only used when mamba_ssm is unavailable.
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc


def _init_bounded_scale(value: float, max_value: float, eps: float = 1e-4) -> torch.Tensor:
    ratio = float(value) / max(float(max_value), eps)
    ratio = min(max(ratio, eps), 1.0 - eps)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


def _bounded_positive_scale(raw: torch.Tensor, max_value: float) -> torch.Tensor:
    return float(max_value) * torch.sigmoid(raw)


def _as_float(value: Any):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        return float(value.detach().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


class SpatioTemporalEmbedding(nn.Module):
    """
    Reference:
    - STID: value embedding + node embedding + time-of-day/day-of-week identity.
    - Earlier local STID/RNP code: timestamp clamping and identity broadcast pattern.

    Migration:
    We migrate only the identity embedding idea, not the full STID encoder.
    Value, node, and temporal identities are summed into [B,T,N,D].

    Input:
        x:    [B,T,N]
        x_ts: [B,T,2]
    Output:
        H0:   [B,T,N,D]
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        dropout: float = 0.15,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)
        self.time_of_day_size = int(time_of_day_size)
        self.day_of_week_size = int(day_of_week_size)

        self.value_emb = nn.Linear(1, d_model)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        self.tod_emb = nn.Embedding(time_of_day_size, d_model)
        self.dow_emb = nn.Embedding(day_of_week_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.tod_emb.weight)
        nn.init.xavier_uniform_(self.dow_emb.weight)

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SpatioTemporalEmbedding expects x [B,T,N], got {tuple(x.shape)}")
        if x_ts.dim() != 3 or x_ts.shape[-1] < 2:
            raise ValueError(f"x_ts must be [B,T,2], got {tuple(x_ts.shape)}")

        B, T, N = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")
        if x_ts.shape[0] != B or x_ts.shape[1] != T:
            raise ValueError(f"x_ts must match x batch/time, got x={tuple(x.shape)}, x_ts={tuple(x_ts.shape)}")

        tod = x_ts[..., 0].long().clamp(0, self.time_of_day_size - 1)
        dow = x_ts[..., 1].long().clamp(0, self.day_of_week_size - 1)

        value = self.value_emb(x.unsqueeze(-1))
        node = self.node_emb.view(1, 1, N, self.d_model)
        tod = self.tod_emb(tod).unsqueeze(2).expand(-1, -1, N, -1)
        dow = self.dow_emb(dow).unsqueeze(2).expand(-1, -1, N, -1)

        out = value + node + tod + dow
        return self.dropout(self.norm(out))


class TemporalCausalMambaBranch(nn.Module):
    """
    Reference:
    - state-spaces/mamba: official Mamba block API.
    - MambaSL: lightweight single-layer Mamba usage in time-series tasks.
    - Time-Series-Library models/Mamba.py: forecasting model organization around one Mamba block.
    - Earlier local temporal Mamba code: [B,T,N,D] -> [B*N,T,D] reshape pattern.

    Migration:
    We only migrate the temporal scan pattern. Mamba is applied along T for each
    node independently. We do not scan raw node order.

    Input:
        H0:    [B,T,N,D]
    Output:
        H_tem: [B,T,N,D]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        fallback_mlp: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.fallback_mlp = bool(fallback_mlp)

        if self.fallback_mlp:
            self.mamba = None
            self.fallback = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            )
        else:
            if Mamba is None:
                raise ImportError(
                    "mamba_ssm is required for TemporalCausalMambaBranch. "
                    "Install mamba_ssm for training, or set fallback_mlp=True for CPU smoke tests."
                ) from _MAMBA_IMPORT_ERROR
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.fallback = None

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H0: torch.Tensor) -> torch.Tensor:
        if H0.dim() != 4:
            raise ValueError(f"TemporalCausalMambaBranch expects [B,T,N,D], got {tuple(H0.shape)}")

        B, T, N, D = H0.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")

        seq = H0.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        if self.fallback_mlp:
            out = self.fallback(seq.transpose(1, 2)).transpose(1, 2).contiguous()
        else:
            out = self.mamba(seq)
        out = self.dropout(out)
        out = self.norm(out + seq)

        return out.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()


class SpatialGraphFilteringBranch(nn.Module):
    """
    Reference:
    - TimeFilter: graph learner + top-k dependency filtering + graph convolution.
    - AGCRN: adaptive graph from learnable node embeddings.
    - Earlier local RNP graph code: static/dynamic/physical top-k graph construction and safe neighbor gather.

    Migration:
    We implement a lightweight static+dynamic top-k graph filtering module,
    with an optional physical prior when A_phy is provided. Spatial dependency
    is handled by graph message passing, not node-order Mamba.

    Input:
        H_tem: [B,T,N,D]
    Output:
        H_spa: [B,T,N,D]
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        spatial_topk: int = 8,
        spatial_node_dim: int = 16,
        dropout: float = 0.15,
        init_gamma: float = 0.05,
        max_gamma: float = 0.3,
    ):
        super().__init__()
        if spatial_topk < 0:
            raise ValueError("spatial_topk must be non-negative.")

        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)
        self.spatial_topk = int(spatial_topk)
        self.max_gamma = float(max_gamma)

        self.node_emb1 = nn.Parameter(torch.empty(num_nodes, spatial_node_dim))
        self.node_emb2 = nn.Parameter(torch.empty(num_nodes, spatial_node_dim))
        nn.init.xavier_uniform_(self.node_emb1)
        nn.init.xavier_uniform_(self.node_emb2)

        self.q_proj = nn.Linear(d_model, spatial_node_dim)
        self.k_proj = nn.Linear(d_model, spatial_node_dim)
        self.graph_logits = nn.Parameter(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32))

        self.message_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.raw_gamma = nn.Parameter(_init_bounded_scale(init_gamma, self.max_gamma))
        self._last_stats: Dict[str, Any] = {}

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

    def _prepare_physical_graph(self, A_phy: torch.Tensor, B: int, N: int, device, dtype):
        if A_phy is None:
            return None
        if A_phy.dim() == 2:
            A_phy = A_phy.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        elif A_phy.dim() == 3:
            A_phy = A_phy.to(device=device, dtype=dtype)
            if A_phy.shape[0] == 1:
                A_phy = A_phy.expand(B, -1, -1)
        else:
            raise ValueError(f"A_phy must be [N,N] or [B,N,N], got {tuple(A_phy.shape)}")
        if A_phy.shape != (B, N, N):
            raise ValueError(f"A_phy must be [{B},{N},{N}], got {tuple(A_phy.shape)}")
        return self._row_softmax_graph(A_phy, mask_self=True)

    def _topk_graph(self, A: torch.Tensor):
        B, N, _ = A.shape
        k = min(self.spatial_topk, max(N - 1, 0))
        if k == 0:
            topk_idx = torch.empty(B, N, 0, device=A.device, dtype=torch.long)
            topk_attn = torch.empty(B, N, 0, device=A.device, dtype=A.dtype)
            return topk_idx, topk_attn, k

        eye = torch.eye(N, device=A.device, dtype=torch.bool).view(1, N, N)
        score = A.masked_fill(eye, float("-inf"))
        topk_val, topk_idx = torch.topk(score, k=k, dim=-1)
        topk_attn = torch.softmax(topk_val, dim=-1)
        return topk_idx, topk_attn, k

    def forward(self, H_tem: torch.Tensor, A_phy: torch.Tensor = None) -> torch.Tensor:
        if H_tem.dim() != 4:
            raise ValueError(f"SpatialGraphFilteringBranch expects [B,T,N,D], got {tuple(H_tem.shape)}")

        B, T, N, D = H_tem.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")

        Z = H_tem.mean(dim=1)

        static_score = torch.relu(torch.matmul(self.node_emb1, self.node_emb2.transpose(0, 1)))
        A_static = self._row_softmax_graph(static_score, mask_self=True)

        q = self.q_proj(Z)
        k = self.k_proj(Z)
        dynamic_score = torch.einsum("bnd,bmd->bnm", q, k) / math.sqrt(max(q.shape[-1], 1))
        A_dynamic = self._row_softmax_graph(dynamic_score, mask_self=True)

        A_phy_norm = self._prepare_physical_graph(A_phy, B, N, H_tem.device, H_tem.dtype)
        if A_phy_norm is None:
            lambda_static, lambda_dynamic = torch.softmax(self.graph_logits[:2], dim=0)
            lambda_phy = None
            A = lambda_static * A_static.unsqueeze(0) + lambda_dynamic * A_dynamic
        else:
            lambda_static, lambda_dynamic, lambda_phy = torch.softmax(self.graph_logits, dim=0)
            A = (
                lambda_static * A_static.unsqueeze(0)
                + lambda_dynamic * A_dynamic
                + lambda_phy * A_phy_norm
            )

        topk_idx, topk_attn, effective_topk = self._topk_graph(A)
        if effective_topk == 0:
            neighbor_agg = torch.zeros_like(H_tem)
        else:
            batch_idx = torch.arange(B, device=H_tem.device).view(B, 1, 1, 1).expand(B, T, N, effective_topk)
            time_idx = torch.arange(T, device=H_tem.device).view(1, T, 1, 1).expand(B, T, N, effective_topk)
            node_idx = topk_idx.unsqueeze(1).expand(B, T, N, effective_topk)
            neighbor = H_tem[batch_idx, time_idx, node_idx]
            neighbor_agg = (topk_attn.unsqueeze(1).unsqueeze(-1) * neighbor).sum(dim=3)

        gamma = _bounded_positive_scale(self.raw_gamma, self.max_gamma).to(dtype=H_tem.dtype)
        message = self.dropout(self.message_proj(neighbor_agg))
        H_spa = self.norm(H_tem + gamma * message)

        self._last_stats = {
            "spatial_gamma": gamma.detach(),
            "spatial_topk": torch.tensor(float(effective_topk), device=H_tem.device),
            "lambda_static": lambda_static.detach(),
            "lambda_dynamic": lambda_dynamic.detach(),
            "lambda_phy": lambda_phy.detach() if lambda_phy is not None else None,
        }
        return H_spa

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._last_stats)


class PeriodicFrequencyBranch(nn.Module):
    """
    Reference:
    - TimeAlign: frequency mismatch correction and representation alignment ideas.
    - TimeFilter/TimeMixer-style code: lightweight filtering for low-frequency structure.
    - Earlier local RNP decomposition code: moving-average/periodic-prototype simplification.

    Migration:
    We keep only a lightweight low-pass temporal filter plus traffic time identities.
    No complex FFT stack or auxiliary alignment loss is added in this first version.

    Input:
        H0:   [B,T,N,D]
        x_ts: [B,T,2]
    Output:
        H_per: [B,T,N,D]
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.15,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        kernel_size: int = 3,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.d_model = int(d_model)
        self.time_of_day_size = int(time_of_day_size)
        self.day_of_week_size = int(day_of_week_size)

        self.low_pass = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=True,
        )
        self.tod_emb = nn.Embedding(time_of_day_size, d_model)
        self.dow_emb = nn.Embedding(day_of_week_size, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.tod_emb.weight)
        nn.init.xavier_uniform_(self.dow_emb.weight)

    def forward(self, H0: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        if H0.dim() != 4:
            raise ValueError(f"PeriodicFrequencyBranch expects H0 [B,T,N,D], got {tuple(H0.shape)}")
        if x_ts.dim() != 3 or x_ts.shape[-1] < 2:
            raise ValueError(f"x_ts must be [B,T,2], got {tuple(x_ts.shape)}")

        B, T, N, D = H0.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")
        if x_ts.shape[0] != B or x_ts.shape[1] != T:
            raise ValueError(f"x_ts must match H0 batch/time, got H0={tuple(H0.shape)}, x_ts={tuple(x_ts.shape)}")

        low = H0.permute(0, 2, 3, 1).contiguous().view(B * N, D, T)
        low = self.low_pass(low)
        low = low.view(B, N, D, T).permute(0, 3, 1, 2).contiguous()

        tod = x_ts[..., 0].long().clamp(0, self.time_of_day_size - 1)
        dow = x_ts[..., 1].long().clamp(0, self.day_of_week_size - 1)
        time_emb = self.tod_emb(tod) + self.dow_emb(dow)
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, N, -1)

        out = self.dropout(self.proj(low + time_emb))
        return self.norm(H0 + out)


class NodeDomainAdapter(nn.Module):
    """
    Reference:
    - AGCRN: node embeddings model node-specific traffic patterns.
    - STID: node identity embedding for spatial heterogeneity.
    - DST-Mamba: node-conditioned adapter idea in traffic forecasting code.

    Migration:
    We implement only a lightweight node-aware shared/private adapter.
    This is not full cross-city domain adaptation; it models per-node heterogeneity.

    Input:
        H_tem: [B,T,N,D]
    Output:
        H_dom: [B,T,N,D]
    """

    def __init__(self, num_nodes: int, d_model: int, dropout: float = 0.15, hidden_dim: int = None):
        super().__init__()
        hidden_dim = int(hidden_dim or d_model)
        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)

        self.node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        nn.init.xavier_uniform_(self.node_emb)

        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.private_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H_tem: torch.Tensor) -> torch.Tensor:
        if H_tem.dim() != 4:
            raise ValueError(f"NodeDomainAdapter expects [B,T,N,D], got {tuple(H_tem.shape)}")
        B, T, N, D = H_tem.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")

        node = self.node_emb.view(1, 1, N, D)
        shared = self.shared_mlp(H_tem)
        private = self.private_mlp(H_tem + node)
        gate = self.gate_mlp(self.node_emb).view(1, 1, N, D)
        return self.norm(H_tem + gate * private + (1.0 - gate) * shared)


class RoleAwareGatedFusion(nn.Module):
    """
    Reference:
    - Earlier local dual-branch code: gated component selection.
    - Common mixture-of-experts style softmax gating.

    Migration:
    The gate is role-aware over four aligned features: temporal, spatial,
    periodic, and domain. It is not a two-branch y_rec/y_non gate.

    Input:
        H_tem, H_spa, H_per, H_dom: [B,T,N,D]
    Output:
        H_fuse: [B,T,N,D]
    """

    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.d_model = int(d_model)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        self.norm = nn.LayerNorm(d_model)
        self._last_stats: Dict[str, Any] = {}

    def forward(self, H_tem: torch.Tensor, H_spa: torch.Tensor, H_per: torch.Tensor, H_dom: torch.Tensor) -> torch.Tensor:
        shape = H_tem.shape
        for name, tensor in (("H_spa", H_spa), ("H_per", H_per), ("H_dom", H_dom)):
            if tensor.shape != shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match H_tem {tuple(shape)}")
        if H_tem.dim() != 4:
            raise ValueError(f"RoleAwareGatedFusion expects [B,T,N,D], got {tuple(H_tem.shape)}")
        if H_tem.shape[-1] != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={H_tem.shape[-1]}")

        H_cat = torch.cat([H_tem, H_spa, H_per, H_dom], dim=-1)
        gate = torch.softmax(self.gate_mlp(H_cat), dim=-1)
        H_fuse = (
            gate[..., 0:1] * H_tem
            + gate[..., 1:2] * H_spa
            + gate[..., 2:3] * H_per
            + gate[..., 3:4] * H_dom
        )
        H_fuse = self.norm(H_fuse)

        self._last_stats = {
            "role_gate_tem_mean": gate[..., 0].detach().mean(),
            "role_gate_spa_mean": gate[..., 1].detach().mean(),
            "role_gate_per_mean": gate[..., 2].detach().mean(),
            "role_gate_dom_mean": gate[..., 3].detach().mean(),
        }
        return H_fuse

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._last_stats)


class ForecastHead(nn.Module):
    """
    Reference:
    - STID prediction head: per-node forecasting without outputting an extra channel.
    - Earlier local STID-style head: output contract [B,H,N].

    Migration:
    For stability, the first RAST-Mamba version uses a simple flatten-linear
    head instead of a deeper STID Conv2d head.

    Input:
        H_fuse: [B,T,N,D]
    Output:
        Y_base: [B,H,N]
    """

    def __init__(self, input_len: int, output_len: int, d_model: int):
        super().__init__()
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.d_model = int(d_model)
        self.proj = nn.Linear(input_len * d_model, output_len)

    def forward(self, H_fuse: torch.Tensor) -> torch.Tensor:
        if H_fuse.dim() != 4:
            raise ValueError(f"ForecastHead expects [B,T,N,D], got {tuple(H_fuse.shape)}")
        B, T, N, D = H_fuse.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")

        z = H_fuse.permute(0, 2, 1, 3).contiguous().view(B, N, T * D)
        y = self.proj(z)
        return y.permute(0, 2, 1).contiguous()


class ResidualCorrectionHead(nn.Module):
    """
    Reference:
    - PnP-Corrector style base prediction + residual correction idea.
    - Earlier local RNP code: bounded residual/component horizon scales.

    Migration:
    We use a weak, bounded residual head:
        Y_hat = Y_base + Delta_Y
    The correction projection is deliberately small at initialization.

    Input:
        H_fuse: [B,T,N,D]
        Y_base: [B,H,N]
    Output:
        Delta_Y: [B,H,N]
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        d_model: int,
        correction_scale_init: float = 0.1,
        correction_scale_max: float = 0.5,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.d_model = int(d_model)
        self.correction_scale_max = float(correction_scale_max)

        self.proj = nn.Linear(input_len * d_model, output_len)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

        self.raw_scale = nn.Parameter(_init_bounded_scale(correction_scale_init, correction_scale_max))
        self.raw_horizon_scale = nn.Parameter(torch.zeros(output_len))
        self._last_stats: Dict[str, Any] = {}

    def forward(self, H_fuse: torch.Tensor, Y_base: torch.Tensor) -> torch.Tensor:
        if H_fuse.dim() != 4:
            raise ValueError(f"ResidualCorrectionHead expects H_fuse [B,T,N,D], got {tuple(H_fuse.shape)}")
        B, T, N, D = H_fuse.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")
        if tuple(Y_base.shape) != (B, self.output_len, N):
            raise ValueError(f"Y_base must be [B,{self.output_len},N], got {tuple(Y_base.shape)}")

        z = H_fuse.permute(0, 2, 1, 3).contiguous().view(B, N, T * D)
        delta = self.proj(z).permute(0, 2, 1).contiguous()

        correction_scale = _bounded_positive_scale(self.raw_scale, self.correction_scale_max).to(dtype=H_fuse.dtype)
        horizon_scale = torch.sigmoid(self.raw_horizon_scale).to(dtype=H_fuse.dtype).view(1, self.output_len, 1)
        Delta_Y = correction_scale * horizon_scale * delta

        self._last_stats = {
            "correction_abs_mean": Delta_Y.detach().abs().mean(),
            "correction_std": Delta_Y.detach().std(unbiased=False),
            "correction_scale": correction_scale.detach(),
            "correction_horizon_scale_mean": horizon_scale.detach().mean(),
        }
        return Delta_Y

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._last_stats)


class RASTMamba(nn.Module):
    """
    RAST-Mamba: Role-Aware Spatio-Temporal Mamba for Traffic Forecasting.

    Reference:
    - state-spaces/mamba, MambaSL, Time-Series-Library: temporal Mamba usage.
    - TimeFilter and AGCRN: adaptive/dynamic graph filtering.
    - STID and earlier local STID code: traffic identity embeddings and [B,H,N] head.
    - TimeAlign: lightweight frequency/alignment motivation without adding an aux loss.
    - Earlier local RNP code: diagnostics, bounded residual scales, and shape discipline.

    Migration:
    Mamba only scans along the temporal axis T. Spatial dependency goes through
    graph filtering, periodic/low-frequency structure through a light conv branch,
    and node heterogeneity through a node-aware adapter. Four role features are
    fused by a softmax gate, followed by base forecast plus weak residual correction.

    Forward flow:
        x      -> [B,T,N]
        H0     -> [B,T,N,D]
        H_tem  -> [B,T,N,D]
        H_spa  -> [B,T,N,D]
        H_per  -> [B,T,N,D]
        H_dom  -> [B,T,N,D]
        H_fuse -> [B,T,N,D]
        Y_base -> [B,H,N]
        Delta  -> [B,H,N]
        Y_hat  -> [B,H,N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        spatial_topk: int = 8,
        spatial_node_dim: int = 16,
        correction_scale_init: float = 0.1,
        correction_scale_max: float = 0.5,
        fallback_mlp: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.d_model = int(d_model)

        self.embedding = SpatioTemporalEmbedding(
            num_nodes=num_nodes,
            d_model=d_model,
            dropout=dropout,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
        )
        self.temporal_branch = TemporalCausalMambaBranch(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            fallback_mlp=fallback_mlp,
        )
        self.spatial_branch = SpatialGraphFilteringBranch(
            num_nodes=num_nodes,
            d_model=d_model,
            spatial_topk=spatial_topk,
            spatial_node_dim=spatial_node_dim,
            dropout=dropout,
            init_gamma=0.05,
            max_gamma=0.3,
        )
        self.periodic_branch = PeriodicFrequencyBranch(
            d_model=d_model,
            dropout=dropout,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
            kernel_size=3,
        )
        self.domain_adapter = NodeDomainAdapter(num_nodes=num_nodes, d_model=d_model, dropout=dropout)
        self.fusion = RoleAwareGatedFusion(d_model=d_model, dropout=dropout)
        self.forecast_head = ForecastHead(input_len=input_len, output_len=output_len, d_model=d_model)
        self.correction_head = ResidualCorrectionHead(
            input_len=input_len,
            output_len=output_len,
            d_model=d_model,
            correction_scale_init=correction_scale_init,
            correction_scale_max=correction_scale_max,
        )

        self._last_stats: Dict[str, Any] = {"fusion_mode": "rast_mamba"}

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor, A_phy: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.dim() != 3:
            raise ValueError(f"RASTMamba expects x shape [B,T,N] or [B,T,N,1], got {tuple(x.shape)}")
        if x_ts.dim() != 3 or x_ts.shape[-1] < 2:
            raise ValueError(f"x_ts must be [B,T,2], got {tuple(x_ts.shape)}")

        B, T, N = x.shape
        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")
        if x_ts.shape[0] != B or x_ts.shape[1] != T:
            raise ValueError(f"x_ts must match x batch/time, got x={tuple(x.shape)}, x_ts={tuple(x_ts.shape)}")

        H0 = self.embedding(x, x_ts)
        H_tem = self.temporal_branch(H0)
        H_spa = self.spatial_branch(H_tem, A_phy=A_phy)
        H_per = self.periodic_branch(H0, x_ts)
        H_dom = self.domain_adapter(H_tem)
        H_fuse = self.fusion(H_tem, H_spa, H_per, H_dom)

        Y_base = self.forecast_head(H_fuse)
        Delta_Y = self.correction_head(H_fuse, Y_base)
        Y_hat = Y_base + Delta_Y

        self._last_stats = {
            "fusion_mode": "rast_mamba",
            **self.fusion.get_stats(),
            **self.correction_head.get_stats(),
            **self.spatial_branch.get_stats(),
        }
        return Y_hat

    def get_gate_stats(self) -> Dict[str, Any]:
        stats = {
            "fusion_mode": "rast_mamba",
            "role_gate_tem_mean": 0.0,
            "role_gate_spa_mean": 0.0,
            "role_gate_per_mean": 0.0,
            "role_gate_dom_mean": 0.0,
            "correction_abs_mean": 0.0,
            "correction_std": 0.0,
            "correction_scale": 0.0,
            "correction_horizon_scale_mean": 0.0,
            "spatial_gamma": 0.0,
            "spatial_topk": 0.0,
            "lambda_static": 0.0,
            "lambda_dynamic": 0.0,
            "lambda_phy": None,
        }
        stats.update(self._last_stats)
        return {key: _as_float(value) for key, value in stats.items()}
