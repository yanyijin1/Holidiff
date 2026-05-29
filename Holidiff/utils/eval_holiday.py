import torch

from Holidiff.utils.eval_common import (
    EPS,
    holiday_masks,
    masked_mae,
    masked_mape,
    masked_mse,
    masked_rmse,
    merge_masks,
    scalarize,
    to_bool_mask,
    to_tensor,
)


def historical_state_compatibility(hist, z, h_hist=None, bandwidth_scale=0.5, eps=EPS):
    hist, z = to_tensor(hist), to_tensor(z)
    median = hist.median(dim=1, keepdim=True).values
    free_mask = (hist <= median).to(hist.dtype)
    cong_mask = (hist > median).to(hist.dtype)
    free_cnt = free_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    cong_cnt = cong_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    if h_hist is None:
        h_hist = hist.std(dim=1, keepdim=True).clamp_min(eps) * bandwidth_scale
    else:
        h_hist = to_tensor(h_hist)
        if h_hist.dim() == 0:
            h_hist = h_hist.view(1, 1, 1)
    diff = z.unsqueeze(2) - hist.unsqueeze(1)
    kernel = torch.exp(-0.5 * (diff / (h_hist.unsqueeze(1) + eps)) ** 2)
    l_free = (kernel * free_mask.unsqueeze(1)).sum(dim=2) / (free_cnt + eps)
    l_cong = (kernel * cong_mask.unsqueeze(1)).sum(dim=2) / (cong_cnt + eps)
    p_cong = l_cong / (l_free + l_cong + eps)
    return l_free, l_cong, p_cong


def build_congestion_mask(hist, y_true, gamma=0.5, **kwargs):
    _, _, p = historical_state_compatibility(hist, y_true, **kwargs)
    return p >= gamma, p


def build_pred_congestion_mask(hist, y_pred, gamma=0.5, **kwargs):
    _, _, p = historical_state_compatibility(hist, y_pred, **kwargs)
    return p >= gamma, p


def compute_holiday_metrics(y_pred, y_true, hist, holiday, gamma=0.5, eps=EPS, valid_mask=None, bandwidth_scale=0.5):
    y_pred, y_true, hist = to_tensor(y_pred), to_tensor(y_true), to_tensor(hist)
    valid_mask = merge_masks(valid_mask, shape=y_true.shape)
    hol_mask, reg_mask = holiday_masks(holiday, y_true.shape[-1])
    g_true, p_true = build_congestion_mask(hist, y_true, gamma=gamma, bandwidth_scale=bandwidth_scale, eps=eps)
    hol_cg_mask = hol_mask & g_true
    reg_cg_mask = reg_mask & g_true
    hol_eval_mask = merge_masks(hol_mask, valid_mask, shape=y_true.shape)
    reg_eval_mask = merge_masks(reg_mask, valid_mask, shape=y_true.shape)
    hol_cg_eval_mask = merge_masks(hol_cg_mask, valid_mask, shape=y_true.shape)
    reg_cg_eval_mask = merge_masks(reg_cg_mask, valid_mask, shape=y_true.shape)
    hol_mae = masked_mae(y_pred, y_true, hol_eval_mask, eps)
    reg_mae = masked_mae(y_pred, y_true, reg_eval_mask, eps)
    hol_cg_mae = masked_mae(y_pred, y_true, hol_cg_eval_mask, eps)
    reg_cg_mae = masked_mae(y_pred, y_true, reg_cg_eval_mask, eps)
    return {
        'MAE': masked_mae(y_pred, y_true, valid_mask, eps),
        'MSE': masked_mse(y_pred, y_true, valid_mask, eps),
        'RMSE': masked_rmse(y_pred, y_true, valid_mask, eps),
        'MAPE': masked_mape(y_pred, y_true, valid_mask, eps),
        'Hol-MAE': hol_mae,
        'Reg-MAE': reg_mae,
        'HDR': (hol_mae - reg_mae) / (reg_mae + eps),
        'Hol-CG-MAE': hol_cg_mae,
        'Reg-CG-MAE': reg_cg_mae,
        'CG-HDR': (hol_cg_mae - reg_cg_mae) / (reg_cg_mae + eps),
        'congestion_ratio': g_true.float().mean(),
        'holiday_congestion_ratio': hol_cg_eval_mask.float().mean() if hol_cg_eval_mask is not None else hol_cg_mask.float().mean(),
        'holiday_ratio': hol_mask.float().mean(),
        'p_cong_mean': p_true.mean(),
    }


def compute_peak_metrics(y_pred, y_true, holiday, delta_t=1.0, peak_radius=1, eps=EPS, valid_mask=None):
    y_pred, y_true, holiday = to_tensor(y_pred), to_tensor(y_true), to_tensor(holiday)
    valid_mask = merge_masks(valid_mask, shape=y_true.shape)
    _, horizon, n_nodes = y_true.shape
    h_true = y_true.argmax(dim=1)
    h_pred = y_pred.argmax(dim=1)
    holiday_3d = holiday.unsqueeze(-1).expand(-1, -1, n_nodes)
    holiday_at_peak = torch.gather(holiday_3d, dim=1, index=h_true.unsqueeze(1)).squeeze(1).bool()
    pte = torch.abs(h_true - h_pred).to(y_true.dtype) * delta_t
    hol_pte = (pte * holiday_at_peak.to(y_true.dtype)).sum() / (holiday_at_peak.sum() + eps)
    pve = torch.abs(y_pred.max(dim=1).values - y_true.max(dim=1).values)
    hol_pve = (pve * holiday_at_peak.to(y_true.dtype)).sum() / (holiday_at_peak.sum() + eps)
    h_index = torch.arange(horizon, device=y_true.device).view(1, horizon, 1)
    peak_mask = torch.abs(h_index - h_true.unsqueeze(1)) <= peak_radius
    hol_mask = holiday.bool().unsqueeze(-1).expand(-1, -1, n_nodes)
    return {
        'Hol-PTE': hol_pte,
        'Hol-PVE': hol_pve,
        'Hol-Peak-MAE': masked_mae(y_pred, y_true, merge_masks(peak_mask, hol_mask, valid_mask, shape=y_true.shape), eps),
    }


def compute_edge_std(hist, edge_index, eps=EPS):
    hist = to_tensor(hist)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=hist.device)
    src, dst = edge_index[0], edge_index[1]
    return (hist[:, :, dst] - hist[:, :, src]).std(dim=1).clamp_min(eps)


def compute_field_consistency_metrics(y_pred, y_true, hist, holiday, adj, g_true=None, eps=EPS):
    y_pred, y_true, hist, holiday, adj = to_tensor(y_pred), to_tensor(y_true), to_tensor(hist), to_tensor(holiday), to_tensor(adj)
    edge_index = (adj > 0).nonzero(as_tuple=False).t().long().to(y_true.device)
    if edge_index.numel() == 0:
        nan = torch.tensor(float('nan'))
        return {'EDE': nan, 'FCE': nan}
    src, dst = edge_index[0], edge_index[1]
    pred_diff = y_pred[:, :, dst] - y_pred[:, :, src]
    true_diff = y_true[:, :, dst] - y_true[:, :, src]
    edge_err = torch.abs(pred_diff - true_diff)
    edge_std = compute_edge_std(hist, edge_index, eps)
    norm_edge_err = edge_err / (edge_std[:, None, :] + eps)
    output = {'EDE': edge_err.mean(), 'FCE': norm_edge_err.mean()}
    if g_true is not None:
        g_true = to_bool_mask(g_true, y_true.shape)
        hol_edge = holiday.bool().unsqueeze(-1).expand(-1, -1, edge_index.shape[1])
        cg_edge = g_true[:, :, src] | g_true[:, :, dst]
        hol_cg_edge_mask = hol_edge & cg_edge
        output['Hol-CG-FCE'] = (norm_edge_err * hol_cg_edge_mask.to(norm_edge_err.dtype)).sum() / (hol_cg_edge_mask.sum() + eps)
    return output


def compute_congestion_event_metrics(y_pred, y_true, hist, holiday, gamma=0.5, eps=EPS, bandwidth_scale=0.5):
    y_pred, y_true, hist = to_tensor(y_pred), to_tensor(y_true), to_tensor(hist)
    hol_mask = to_tensor(holiday).bool().unsqueeze(-1).expand_as(y_true)
    g_true, _ = build_congestion_mask(hist, y_true, gamma=gamma, bandwidth_scale=bandwidth_scale, eps=eps)
    g_pred, _ = build_pred_congestion_mask(hist, y_pred, gamma=gamma, bandwidth_scale=bandwidth_scale, eps=eps)
    tp = (hol_mask & g_true & g_pred).sum().to(y_true.dtype)
    fp = (hol_mask & (~g_true) & g_pred).sum().to(y_true.dtype)
    fn = (hol_mask & g_true & (~g_pred)).sum().to(y_true.dtype)
    return {
        'CG-F1': 2 * tp / (2 * tp + fp + fn + eps),
        'CG-CSI': tp / (tp + fp + fn + eps),
        'TP': tp,
        'FP': fp,
        'FN': fn,
    }


def aggregate_mean(samples):
    return to_tensor(samples).mean(dim=0)


def aggregate_median(samples):
    return to_tensor(samples).median(dim=0).values


def dca_aggregate(samples, hist, h_kde=None, h_hist=None, kde_scale=0.5, hist_scale=0.5, eps=EPS):
    samples, hist = to_tensor(samples), to_tensor(hist)
    if h_kde is None:
        h_kde = samples.std(dim=0, keepdim=True).clamp_min(eps) * kde_scale
    else:
        h_kde = to_tensor(h_kde)
    diff = samples.unsqueeze(1) - samples.unsqueeze(0)
    rho_cloud = torch.exp(-0.5 * (diff / (h_kde.unsqueeze(0) + eps)) ** 2).mean(dim=1)
    rho_hist = []
    for sample_idx in range(samples.shape[0]):
        l_free, l_cong, _ = historical_state_compatibility(hist, samples[sample_idx], h_hist=h_hist, bandwidth_scale=hist_scale, eps=eps)
        rho_hist.append(l_free + l_cong)
    rho_hist = torch.stack(rho_hist, dim=0)
    weights = rho_cloud * rho_hist
    weights = weights / (weights.sum(dim=0, keepdim=True) + eps)
    return (weights * samples).sum(dim=0)


def compute_dca_gain(samples, y_true, hist, holiday, gamma=0.5, eps=EPS, valid_mask=None, bandwidth_scale=0.5):
    y_mean = aggregate_mean(samples)
    y_median = aggregate_median(samples)
    y_dca = dca_aggregate(samples, hist)
    mean_metrics = compute_holiday_metrics(y_mean, y_true, hist, holiday, gamma, eps, valid_mask, bandwidth_scale)
    median_metrics = compute_holiday_metrics(y_median, y_true, hist, holiday, gamma, eps, valid_mask, bandwidth_scale)
    dca_metrics = compute_holiday_metrics(y_dca, y_true, hist, holiday, gamma, eps, valid_mask, bandwidth_scale)
    return {
        'Mean': {k: scalarize(v) for k, v in mean_metrics.items()},
        'Median': {k: scalarize(v) for k, v in median_metrics.items()},
        'DCA': {k: scalarize(v) for k, v in dca_metrics.items()},
        'DCA-Gain-Hol-CG': scalarize((mean_metrics['Hol-CG-MAE'] - dca_metrics['Hol-CG-MAE']) / (mean_metrics['Hol-CG-MAE'] + eps)),
    }


def evaluate_all(y_pred, y_true, hist, holiday, adj=None, samples=None, gamma=0.5, delta_t=1.0, peak_radius=1, eps=EPS, valid_mask=None, bandwidth_scale=0.5):
    y_pred, y_true, hist, holiday = to_tensor(y_pred), to_tensor(y_true), to_tensor(hist), to_tensor(holiday)
    valid_mask = merge_masks(valid_mask, shape=y_true.shape)
    results = {}
    results.update(compute_holiday_metrics(y_pred, y_true, hist, holiday, gamma, eps, valid_mask, bandwidth_scale))
    results.update(compute_peak_metrics(y_pred, y_true, holiday, delta_t, peak_radius, eps, valid_mask))
    results.update(compute_congestion_event_metrics(y_pred, y_true, hist, holiday, gamma, eps, bandwidth_scale))
    if adj is not None:
        g_true, _ = build_congestion_mask(hist, y_true, gamma=gamma, bandwidth_scale=bandwidth_scale, eps=eps)
        results.update(compute_field_consistency_metrics(y_pred, y_true, hist, holiday, adj, g_true, eps))
    if samples is not None:
        results['DCA-Analysis'] = compute_dca_gain(samples, y_true, hist, holiday, gamma, eps, valid_mask, bandwidth_scale)
    return {k: (scalarize(v) if k != 'DCA-Analysis' else v) for k, v in results.items()}
