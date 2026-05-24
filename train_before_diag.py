import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from data.pems_dataset import build_datasets
from models.stid_core import STIDCore
from models.rnp_mamba import RNPMambaV1, RNPMambaV11, RNPMambaV12
from utils.scaler import StandardScaler
from utils.metrics import masked_mae, masked_mae_with_raw_mask, compute_all_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_loaders(args, scaler):
    train_set, val_set, test_set = build_datasets(
        data_dir=args.data_dir,
        scaler=scaler,
        input_len=args.input_len,
        output_len=args.output_len,
        points_per_day=args.points_per_day,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, train_set


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()

    total_loss = 0.0
    total_batches = 0

    pbar = tqdm(loader, ncols=120)
    for x, y_norm, x_ts, y_ts, y_raw in pbar:
        x = x.to(device)
        y_norm = y_norm.to(device)
        x_ts = x_ts.to(device)
        y_raw = y_raw.to(device)

        optimizer.zero_grad()

        pred_norm = model(x, x_ts)

        # Loss in normalized space, but mask is built from raw labels.
        loss = masked_mae_with_raw_mask(
            pred_norm,
            y_norm,
            y_raw,
            null_val=args.null_val,
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, scaler, device, args):
    model.eval()

    preds_raw = []
    labels_raw = []

    for x, y_norm, x_ts, y_ts, y_raw in loader:
        x = x.to(device)
        x_ts = x_ts.to(device)
        y_norm = y_norm.to(device)

        pred_norm = model(x, x_ts)

        pred_raw = scaler.inverse_transform_tensor(pred_norm)

        # Use exact raw labels from dataset.
        # Do not use inverse_transform(y_norm), otherwise zeros may become tiny nonzero values
        # and MAPE will explode.
        label_raw = y_raw.to(device)

        preds_raw.append(pred_raw.detach().cpu())
        labels_raw.append(label_raw.detach().cpu())

    preds = torch.cat(preds_raw, dim=0)
    labels = torch.cat(labels_raw, dim=0)

    avg_metrics = compute_all_metrics(preds, labels, null_val=args.null_val)

    horizon_metrics = {}
    for h in args.eval_horizons:
        if h <= preds.shape[1]:
            horizon_metrics[f"h{h}"] = compute_all_metrics(
                preds[:, h - 1:h, :],
                labels[:, h - 1:h, :],
                null_val=args.null_val,
            )

    return avg_metrics, horizon_metrics


def print_metrics(prefix, avg_metrics, horizon_metrics=None):
    msg = (
        f"{prefix}: "
        f"MAE={avg_metrics['MAE']:.4f}, "
        f"RMSE={avg_metrics['RMSE']:.4f}, "
        f"MAPE={avg_metrics['MAPE']:.4f}"
    )
    print(msg)

    if horizon_metrics:
        for h, m in horizon_metrics.items():
            print(
                f"{prefix}@{h}: "
                f"MAE={m['MAE']:.4f}, "
                f"RMSE={m['RMSE']:.4f}, "
                f"MAPE={m['MAPE']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='stid_core', choices=['stid_core', 'rnp_mamba_v1', 'rnp_mamba_v11', 'rnp_mamba_v12'])

    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/BasicTS/datasets/PEMS08")
    parser.add_argument("--save_dir", type=str, default="checkpoints/stid_core_pems08")

    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--points_per_day", type=int, default=288)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--milestones", type=int, nargs="+", default=[1, 50, 80])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--rnp_d_model", type=int, default=64)
    parser.add_argument("--rnp_smooth_kernel", type=int, default=3)
    parser.add_argument("--rnp_d_state", type=int, default=16)
    parser.add_argument("--rnp_d_conv", type=int, default=2)
    parser.add_argument("--rnp_expand", type=int, default=1)
    parser.add_argument("--rnp_head_layers", type=int, default=3)
    parser.add_argument("--node_dim", type=int, default=32)
    parser.add_argument("--tid_dim", type=int, default=32)
    parser.add_argument("--diw_dim", type=int, default=32)

    parser.add_argument("--null_val", type=float, default=0.0)
    parser.add_argument("--eval_horizons", type=int, nargs="+", default=[3, 6, 12])

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Fallback to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    print("=" * 80)
    print("Lightweight traffic forecasting framework")
    print("model:", args.model)
    print("data_dir:", args.data_dir)
    print("device:", device)
    print("input_len:", args.input_len)
    print("output_len:", args.output_len)
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print("=" * 80)

    train_data_path = os.path.join(args.data_dir, "train_data.npy")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Cannot find {train_data_path}")

    train_data = np.load(train_data_path).astype(np.float32)
    if train_data.ndim == 3 and train_data.shape[-1] == 1:
        train_data = train_data[..., 0]
    if train_data.ndim != 2:
        raise ValueError(f"Expected train_data shape [T, N], got {train_data.shape}")

    scaler = StandardScaler().fit(train_data)
    num_nodes = train_data.shape[1]

    print("train_data shape:", train_data.shape)
    print("num_nodes:", num_nodes)
    print("scaler mean shape:", scaler.mean.shape)
    print("scaler std shape:", scaler.std.shape)

    train_loader, val_loader, test_loader, train_set = build_loaders(args, scaler)

    if args.model == "stid_core":
        model = STIDCore(
            num_nodes=num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            if_node=True,
            node_dim=args.node_dim,
            if_time_in_day=True,
            if_day_in_week=True,
            temp_dim_tid=args.tid_dim,
            temp_dim_diw=args.diw_dim,
            time_of_day_size=args.points_per_day,
            day_of_week_size=7,
        ).to(device)

    elif args.model == "rnp_mamba_v1":
        model = RNPMambaV1(
            num_nodes=num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            d_model=args.rnp_d_model,
            smooth_kernel=args.rnp_smooth_kernel,
            d_state=args.rnp_d_state,
            d_conv=args.rnp_d_conv,
            expand=args.rnp_expand,
            dropout=args.dropout,
            if_node=True,
            node_dim=args.node_dim,
            if_time_in_day=True,
            if_day_in_week=True,
            temp_dim_tid=args.tid_dim,
            temp_dim_diw=args.diw_dim,
            time_of_day_size=args.points_per_day,
            day_of_week_size=7,
        ).to(device)

    elif args.model == "rnp_mamba_v11":
        model = RNPMambaV11(
            num_nodes=num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            d_model=args.rnp_d_model,
            smooth_kernel=args.rnp_smooth_kernel,
            d_state=args.rnp_d_state,
            d_conv=args.rnp_d_conv,
            expand=args.rnp_expand,
            dropout=args.dropout,
            if_node=True,
            node_dim=args.node_dim,
            if_time_in_day=True,
            if_day_in_week=True,
            temp_dim_tid=args.tid_dim,
            temp_dim_diw=args.diw_dim,
            time_of_day_size=args.points_per_day,
            day_of_week_size=7,
        ).to(device)

    elif args.model == "rnp_mamba_v12":
        model = RNPMambaV12(
            num_nodes=num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            d_model=args.rnp_d_model,
            smooth_kernel=args.rnp_smooth_kernel,
            d_state=args.rnp_d_state,
            d_conv=args.rnp_d_conv,
            expand=args.rnp_expand,
            dropout=args.dropout,
            if_node=True,
            node_dim=args.node_dim,
            if_time_in_day=True,
            if_day_in_week=True,
            temp_dim_tid=args.tid_dim,
            temp_dim_diw=args.diw_dim,
            time_of_day_size=args.points_per_day,
            day_of_week_size=7,
            head_layers=args.rnp_head_layers,
        ).to(device)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(model)
    print("Trainable parameters:", count_parameters(model))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "best.pt"

    best_val_mae = float("inf")
    best_epoch = -1
    bad_count = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"lr: {optimizer.param_groups[0]['lr']:.6g}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_metrics, val_horizon_metrics = evaluate(model, val_loader, scaler, device, args)

        print(f"train/loss_norm={train_loss:.6f}")
        print_metrics("val", val_metrics, val_horizon_metrics)

        val_mae = val_metrics["MAE"]

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            bad_count = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "best_val_mae": best_val_mae,
                    "scaler_mean": scaler.mean,
                    "scaler_std": scaler.std,
                },
                best_ckpt_path,
            )
            print(f"Saved best checkpoint to {best_ckpt_path}")
        else:
            bad_count += 1
            print(f"No improvement. bad_count={bad_count}/{args.patience}")

        scheduler.step()

        if bad_count >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    print("\nLoading best checkpoint for final test:", best_ckpt_path)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_horizon_metrics = evaluate(model, test_loader, scaler, device, args)

    print("\nFinal test results:")
    print_metrics("test", test_metrics, test_horizon_metrics)

    result = {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test": test_metrics,
        "test_horizons": test_horizon_metrics,
    }

    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Saved metrics to:", save_dir / "test_metrics.json")


if __name__ == "__main__":
    main()
