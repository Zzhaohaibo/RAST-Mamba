import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rast_mamba import RASTMamba


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torch.manual_seed(2024)

    B = 2
    T = 12
    N = 20
    H = 12
    D = 64

    x = torch.randn(B, T, N)
    x_ts = torch.zeros(B, T, 2).long()
    x_ts[..., 0] = torch.arange(T).view(1, T).expand(B, T) % 288
    x_ts[..., 1] = 0

    model = RASTMamba(
        num_nodes=N,
        input_len=T,
        output_len=H,
        d_model=D,
        d_state=16,
        d_conv=2,
        expand=1,
        dropout=0.1,
        time_of_day_size=288,
        day_of_week_size=7,
        spatial_topk=4,
        spatial_node_dim=16,
        fallback_mlp=True,
    )

    y = model(x, x_ts)
    assert y.shape == (B, H, N), f"Expected output shape {(B, H, N)}, got {tuple(y.shape)}"

    loss = y.mean()
    loss.backward()

    print("output shape:", tuple(y.shape))
    print("parameter count:", count_parameters(model))
    print("backward OK")
    print("gate stats:", model.get_gate_stats())
    print("note: fallback_mlp=True is for CPU smoke testing only; formal training uses mamba_ssm.")


if __name__ == "__main__":
    main()
