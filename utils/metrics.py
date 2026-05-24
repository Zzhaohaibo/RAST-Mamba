import torch


def _get_mask(labels: torch.Tensor, null_val: float = 0.0):
    """
    Build mask from raw labels.

    labels: raw-scale ground truth, usually [B, H, N].
    """
    if null_val is None:
        mask = torch.ones_like(labels, dtype=torch.float32)
    else:
        mask = labels.ne(null_val).float()

    mask_mean = mask.mean()
    if mask_mean > 0:
        mask = mask / mask_mean

    return torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0):
    mask = _get_mask(labels, null_val)
    loss = torch.abs(preds - labels) * mask
    return torch.mean(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))


def masked_mae_with_raw_mask(
    preds: torch.Tensor,
    labels: torch.Tensor,
    raw_labels: torch.Tensor,
    null_val: float = 0.0,
):
    """
    MAE in normalized space, but mask is built from raw labels.
    This is safer for traffic data where 0 means missing value.
    """
    mask = _get_mask(raw_labels, null_val)
    loss = torch.abs(preds - labels) * mask
    return torch.mean(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0):
    mask = _get_mask(labels, null_val)
    loss = (preds - labels) ** 2 * mask
    loss = torch.mean(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.sqrt(loss)


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0):
    """
    MAPE on raw-scale values.

    Important:
    The denominator must not use near-zero pseudo labels produced by inverse scaling.
    Use exact raw labels whenever possible.
    """
    raw_mask = labels.ne(null_val)

    # Avoid division by zero on masked positions.
    safe_labels = torch.where(raw_mask, labels, torch.ones_like(labels))

    mask = raw_mask.float()
    mask_mean = mask.mean()
    if mask_mean > 0:
        mask = mask / mask_mean

    loss = torch.abs((preds - labels) / safe_labels) * mask
    return torch.mean(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))


@torch.no_grad()
def compute_all_metrics(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0):
    return {
        "MAE": masked_mae(preds, labels, null_val).item(),
        "RMSE": masked_rmse(preds, labels, null_val).item(),
        "MAPE": masked_mape(preds, labels, null_val).item(),
    }
