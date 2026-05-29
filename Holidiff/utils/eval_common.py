import numpy as np
import torch

EPS = 1e-8


def to_tensor(x, dtype=torch.float32):
    if x is None:
        return None
    return x.detach().to(dtype=dtype) if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=dtype)


def to_bool_mask(mask, shape=None):
    if mask is None:
        return None
    mask = to_tensor(mask).bool()
    while shape is not None and mask.dim() < len(shape):
        mask = mask.unsqueeze(-1)
    return mask.expand(shape) if shape is not None else mask


def merge_masks(*masks, shape=None):
    masks = [to_bool_mask(m, shape) for m in masks if m is not None]
    if not masks:
        return None
    merged = masks[0]
    for mask in masks[1:]:
        merged = merged & mask
    return merged


def scalarize(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item()) if value.numel() == 1 else value.detach().cpu().numpy().tolist()
    if isinstance(value, np.generic):
        return float(value)
    return value


def masked_mae(y_pred, y_true, mask=None, eps=EPS):
    y_pred, y_true = to_tensor(y_pred), to_tensor(y_true)
    err = torch.abs(y_pred - y_true)
    if mask is None:
        return err.mean()
    mask = to_bool_mask(mask, err.shape).to(err.dtype)
    return (err * mask).sum() / (mask.sum() + eps)


def masked_rmse(y_pred, y_true, mask=None, eps=EPS):
    y_pred, y_true = to_tensor(y_pred), to_tensor(y_true)
    err2 = (y_pred - y_true) ** 2
    if mask is None:
        return torch.sqrt(err2.mean() + eps)
    mask = to_bool_mask(mask, err2.shape).to(err2.dtype)
    return torch.sqrt((err2 * mask).sum() / (mask.sum() + eps) + eps)


def masked_mse(y_pred, y_true, mask=None, eps=EPS):
    y_pred, y_true = to_tensor(y_pred), to_tensor(y_true)
    err2 = (y_pred - y_true) ** 2
    if mask is None:
        return err2.mean()
    mask = to_bool_mask(mask, err2.shape).to(err2.dtype)
    return (err2 * mask).sum() / (mask.sum() + eps)


def masked_mape(y_pred, y_true, mask=None, eps=EPS):
    y_pred, y_true = to_tensor(y_pred), to_tensor(y_true)
    err = torch.abs((y_pred - y_true) / (y_true + eps))
    valid = merge_masks(mask, torch.abs(y_true) > eps, shape=y_true.shape)
    if valid is None:
        return err.mean()
    valid = valid.to(err.dtype)
    return (err * valid).sum() / (valid.sum() + eps)


def holiday_masks(holiday, n_nodes):
    holiday = to_tensor(holiday)
    hol_mask = holiday.bool().unsqueeze(-1).expand(-1, -1, n_nodes)
    reg_mask = ~hol_mask
    return hol_mask, reg_mask
