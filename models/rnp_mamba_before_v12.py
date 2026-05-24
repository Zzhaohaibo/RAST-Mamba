import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class TemporalMovingAverage(nn.Module):
    """
    Fixed temporal smoothing module.

    Input:
        x: [B, T, N, D]
    Output:
        x_smooth: [B, T, N, D]

    This module gives a first-order recurring candidate.
    It is intentionally simple and stable for v1.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape

        # [B, T, N, D] -> [B*N*D, 1, T]
        x_ = x.permute(0, 2, 3, 1).contiguous().view(B * N * D, 1, T)

        # replicate padding avoids introducing zeros at boundaries
        x_ = F.pad(x_, (self.padding, self.padding), mode="replicate")
        x_ = F.avg_pool1d(x_, kernel_size=self.kernel_size, stride=1)

        # [B*N*D, 1, T] -> [B, T, N, D]
        x_ = x_.view(B, N, D, T).permute(0, 3, 1, 2).contiguous()
        return x_


class FeedForwardBranch(nn.Module):
    """
    Lightweight branch used for recurring traffic regularity.

    Input / output:
        [B, T, N, D]
    """

    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class NodeWiseMambaBranch(nn.Module):
    """
    Node-wise temporal Mamba branch.

    This is the key TimePro-inspired implementation pattern:
    each traffic node is treated as one temporal sequence.

    Input:
        x: [B, T, N, D]
    Output:
        y: [B, T, N, D]
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

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape

        # [B, T, N, D] -> [B, N, T, D] -> [B*N, T, D]
        x_seq = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)

        y_seq = self.mamba(x_seq)
        y_seq = self.dropout(y_seq)
        y_seq = self.norm(y_seq + x_seq)

        # [B*N, T, D] -> [B, T, N, D]
        y = y_seq.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return y


class RNPMambaV1(nn.Module):
    """
    RNP-Mamba-v1.

    Design:
        1. Value embedding keeps temporal dimension.
        2. Temporal smoothing gives recurring candidate.
        3. Residual part gives non-recurring candidate.
        4. Recurring branch uses lightweight FFN.
        5. Non-recurring branch uses node-wise Mamba.
        6. Fusion gate combines two candidates.
        7. STID-style node/time identity embeddings are concatenated.
        8. Linear head predicts [B, H, N].

    Input:
        x:    [B, T, N]
        x_ts: [B, T, 2]
    Output:
        y_hat: [B, H, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        d_model: int = 64,
        smooth_kernel: int = 3,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        if_node: bool = True,
        node_dim: int = 32,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
        temp_dim_tid: int = 32,
        temp_dim_diw: int = 32,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model

        self.if_node = if_node
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week

        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        # value embedding: [B,T,N,1] -> [B,T,N,D]
        self.value_proj = nn.Linear(1, d_model)

        # recurring / non-recurring candidate generator
        self.smoother = TemporalMovingAverage(kernel_size=smooth_kernel)

        # two complexity-matched branches
        self.recurring_branch = FeedForwardBranch(d_model=d_model, dropout=dropout)
        self.nonrecurring_branch = NodeWiseMambaBranch(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # feature-wise fusion gate
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # STID-style identity embeddings
        identity_dim = 0

        if if_node:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
            nn.init.xavier_uniform_(self.node_emb)
            identity_dim += node_dim
        else:
            self.node_emb = None

        if if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(time_of_day_size, temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            identity_dim += temp_dim_tid
        else:
            self.time_in_day_emb = None

        if if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(day_of_week_size, temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            identity_dim += temp_dim_diw
        else:
            self.day_in_week_emb = None

        self.identity_dim = identity_dim

        head_in_dim = d_model + identity_dim

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, head_in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in_dim, output_len),
        )

    def build_identity_embedding(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        """
        Build STID-style identity embedding.

        Args:
            x:    [B, T, N]
            x_ts: [B, T, 2]

        Returns:
            identity: [B, N, identity_dim]
        """
        B, T, N = x.shape
        emb_list = []

        if self.if_node:
            node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # [B,N,node_dim]
            emb_list.append(node_emb)

        if self.if_time_in_day:
            tid = x_ts[:, -1, 0].clamp(0, self.time_of_day_size - 1)  # [B]
            tid_emb = self.time_in_day_emb[tid]                       # [B,tid_dim]
            tid_emb = tid_emb.unsqueeze(1).expand(-1, N, -1)           # [B,N,tid_dim]
            emb_list.append(tid_emb)

        if self.if_day_in_week:
            diw = x_ts[:, -1, 1].clamp(0, self.day_of_week_size - 1)   # [B]
            diw_emb = self.day_in_week_emb[diw]                       # [B,diw_dim]
            diw_emb = diw_emb.unsqueeze(1).expand(-1, N, -1)           # [B,N,diw_dim]
            emb_list.append(diw_emb)

        if len(emb_list) == 0:
            return torch.empty(B, N, 0, device=x.device, dtype=x.dtype)

        return torch.cat(emb_list, dim=-1)

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, T, N]
            x_ts: [B, T, 2]

        Returns:
            prediction: [B, H, N]
        """
        if x.dim() != 3:
            raise ValueError(f"RNPMambaV1 expects x shape [B,T,N], got {tuple(x.shape)}")

        B, T, N = x.shape

        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        # [B,T,N] -> [B,T,N,D]
        h = self.value_proj(x.unsqueeze(-1))

        # candidate decomposition
        h_rec_candidate = self.smoother(h)
        h_non_candidate = h - h_rec_candidate

        # branch modeling
        z_rec = self.recurring_branch(h_rec_candidate)       # [B,T,N,D]
        z_non = self.nonrecurring_branch(h_non_candidate)    # [B,T,N,D]

        # aggregate temporal dimension.
        # recurring uses mean; non-recurring uses final state.
        z_rec_node = z_rec.mean(dim=1)       # [B,N,D]
        z_non_node = z_non[:, -1, :, :]      # [B,N,D]

        # simple adaptive fusion
        gate = self.fusion_gate(torch.cat([z_rec_node, z_non_node], dim=-1))  # [B,N,D]
        z = gate * z_non_node + (1.0 - gate) * z_rec_node                    # [B,N,D]

        # identity embedding
        identity = self.build_identity_embedding(x, x_ts)  # [B,N,D_id]

        z = torch.cat([z, identity], dim=-1)  # [B,N,D+D_id]

        # [B,N,D] -> [B,N,H] -> [B,H,N]
        pred = self.prediction_head(z).permute(0, 2, 1).contiguous()

        return pred


class RNPMambaV11(nn.Module):
    """
    RNP-Mamba-v1.1: Identity-aware RNP-Mamba.

    Difference from RNPMambaV1:
        1. Node/time identity embeddings are injected BEFORE decomposition.
        2. Recurring / non-recurring branches are identity-aware.
        3. Fusion gate output is directly sent to prediction head.
        4. No post-fusion identity concatenation.

    Input:
        x:    [B, T, N]
        x_ts: [B, T, 2]

    Output:
        y_hat: [B, H, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        d_model: int = 64,
        smooth_kernel: int = 3,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.15,
        if_node: bool = True,
        node_dim: int = 32,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
        temp_dim_tid: int = 32,
        temp_dim_diw: int = 32,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model

        self.if_node = if_node
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week

        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        # Value embedding keeps temporal dimension:
        # [B,T,N,1] -> [B,T,N,D]
        self.value_proj = nn.Linear(1, d_model)

        # STID-style identity embeddings, but injected before decomposition.
        identity_dim = 0

        if if_node:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
            nn.init.xavier_uniform_(self.node_emb)
            identity_dim += node_dim
        else:
            self.node_emb = None

        if if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(time_of_day_size, temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            identity_dim += temp_dim_tid
        else:
            self.time_in_day_emb = None

        if if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(day_of_week_size, temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            identity_dim += temp_dim_diw
        else:
            self.day_in_week_emb = None

        self.identity_dim = identity_dim

        if identity_dim > 0:
            self.identity_proj = nn.Linear(identity_dim, d_model)
            self.identity_norm = nn.LayerNorm(d_model)
        else:
            self.identity_proj = None
            self.identity_norm = None

        # Decomposition and branches
        self.smoother = TemporalMovingAverage(kernel_size=smooth_kernel)

        self.recurring_branch = FeedForwardBranch(
            d_model=d_model,
            dropout=dropout,
        )

        self.nonrecurring_branch = NodeWiseMambaBranch(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Fusion gate: no post-fusion identity concat.
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Prediction head: [B,N,D] -> [B,N,H] -> [B,H,N]
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len),
        )

    def build_temporal_identity_embedding(
        self,
        x: torch.Tensor,
        x_ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build identity embedding for every historical time step.

        Args:
            x:    [B, T, N]
            x_ts: [B, T, 2]

        Returns:
            identity: [B, T, N, identity_dim]
        """
        B, T, N = x.shape
        emb_list = []

        if self.if_node:
            # [N,node_dim] -> [B,T,N,node_dim]
            node_emb = self.node_emb.view(1, 1, N, -1).expand(B, T, -1, -1)
            emb_list.append(node_emb)

        if self.if_time_in_day:
            # tid: [B,T]
            tid = x_ts[:, :, 0].clamp(0, self.time_of_day_size - 1)
            tid_emb = self.time_in_day_emb[tid]  # [B,T,tid_dim]
            tid_emb = tid_emb.unsqueeze(2).expand(-1, -1, N, -1)  # [B,T,N,tid_dim]
            emb_list.append(tid_emb)

        if self.if_day_in_week:
            # diw: [B,T]
            diw = x_ts[:, :, 1].clamp(0, self.day_of_week_size - 1)
            diw_emb = self.day_in_week_emb[diw]  # [B,T,diw_dim]
            diw_emb = diw_emb.unsqueeze(2).expand(-1, -1, N, -1)  # [B,T,N,diw_dim]
            emb_list.append(diw_emb)

        if len(emb_list) == 0:
            return torch.empty(B, T, N, 0, device=x.device, dtype=x.dtype)

        return torch.cat(emb_list, dim=-1)

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, T, N]
            x_ts: [B, T, 2]

        Returns:
            prediction: [B, H, N]
        """
        if x.dim() != 3:
            raise ValueError(f"RNPMambaV11 expects x shape [B,T,N], got {tuple(x.shape)}")

        B, T, N = x.shape

        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        # Value embedding: [B,T,N] -> [B,T,N,D]
        h_value = self.value_proj(x.unsqueeze(-1))

        # Early identity injection.
        if self.identity_dim > 0:
            identity = self.build_temporal_identity_embedding(x, x_ts)  # [B,T,N,D_id]
            h_identity = self.identity_proj(identity)                   # [B,T,N,D]
            h = h_value + self.identity_norm(h_identity)
        else:
            h = h_value

        # Recurring / non-recurring candidate generation.
        h_rec_candidate = self.smoother(h)
        h_non_candidate = h - h_rec_candidate

        # Branch modeling.
        z_rec = self.recurring_branch(h_rec_candidate)       # [B,T,N,D]
        z_non = self.nonrecurring_branch(h_non_candidate)    # [B,T,N,D]

        # Temporal aggregation.
        z_rec_node = z_rec.mean(dim=1)       # [B,N,D]
        z_non_node = z_non[:, -1, :, :]      # [B,N,D]

        # Fusion.
        gate = self.fusion_gate(torch.cat([z_rec_node, z_non_node], dim=-1))
        z = gate * z_non_node + (1.0 - gate) * z_rec_node

        # Prediction.
        pred = self.prediction_head(z).permute(0, 2, 1).contiguous()

        return pred
