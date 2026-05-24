import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """
    Residual 1x1 Conv MLP block, following the core idea of STID.
    Input / output: [B, C, N, 1]
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()

        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out + residual


class STIDCore(nn.Module):
    """
    Lightweight extracted STID core.

    Input:
        x:     [B, T, N]
        x_ts:  [B, T, 2], where x_ts[..., 0] is time-of-day index,
                            x_ts[..., 1] is day-of-week index.

    Output:
        y_hat: [B, H, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        embed_dim: int = 32,
        num_layers: int = 3,
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

        self.if_node = if_node
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week

        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=input_len,
            out_channels=embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        hidden_dim = embed_dim

        if if_node:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
            nn.init.xavier_uniform_(self.node_emb)
            hidden_dim += node_dim
        else:
            self.node_emb = None

        if if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(time_of_day_size, temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            hidden_dim += temp_dim_tid
        else:
            self.time_in_day_emb = None

        if if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(day_of_week_size, temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            hidden_dim += temp_dim_diw
        else:
            self.day_in_week_emb = None

        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.regression_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def build_embedding(self, x: torch.Tensor, x_ts: torch.Tensor):
        """
        x:    [B, T, N]
        x_ts: [B, T, 2]
        return: [B, hidden_dim, N, 1]
        """
        B, T, N = x.shape

        if T != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={T}")
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        # [B, T, N] -> [B, T, N, 1], Conv2d treats T as channel
        time_series_emb = self.time_series_emb_layer(x.unsqueeze(-1))

        emb_list = [time_series_emb]

        if self.if_node:
            node_emb = self.node_emb.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
            node_emb = node_emb.expand(B, -1, -1, -1)
            emb_list.append(node_emb)

        if self.if_time_in_day:
            tid = x_ts[:, -1, 0].clamp(0, self.time_of_day_size - 1)
            tid_emb = self.time_in_day_emb[tid]  # [B, temp_dim]
            tid_emb = tid_emb.transpose(1, 2) if tid_emb.dim() == 3 else tid_emb
            # Our timestamps are shared by nodes, so expand to all nodes.
            tid_emb = tid_emb.unsqueeze(-1).unsqueeze(-1)  # [B, temp_dim, 1, 1]
            tid_emb = tid_emb.expand(-1, -1, N, 1)
            emb_list.append(tid_emb)

        if self.if_day_in_week:
            diw = x_ts[:, -1, 1].clamp(0, self.day_of_week_size - 1)
            diw_emb = self.day_in_week_emb[diw]  # [B, temp_dim]
            diw_emb = diw_emb.unsqueeze(-1).unsqueeze(-1)  # [B, temp_dim, 1, 1]
            diw_emb = diw_emb.expand(-1, -1, N, 1)
            emb_list.append(diw_emb)

        hidden = torch.cat(emb_list, dim=1)
        return hidden

    def forward(self, x: torch.Tensor, x_ts: torch.Tensor):
        hidden = self.build_embedding(x, x_ts)
        hidden = self.encoder(hidden)
        out = self.regression_layer(hidden)  # [B, H, N, 1]
        out = out.squeeze(-1)               # [B, H, N]
        return out
