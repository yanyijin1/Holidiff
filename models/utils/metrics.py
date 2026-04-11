import numpy as np
from .dtw_metric import dtw as _dtw, accelerated_dtw as _accelerated_dtw
from sklearn.metrics.pairwise import manhattan_distances


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def dtw_metric(pred, true, use_accelerated=False):
    """
    计算 DTW (Dynamic Time Warping) 距离
    
    Args:
        pred: 预测值，shape (batch, seq_len, n_vars) 或 (seq_len,)
        true: 真实值，shape (batch, seq_len, n_vars) 或 (seq_len,)
        use_accelerated: 是否使用加速版本
    
    Returns:
        DTW 距离的平均值
    """
    if pred.ndim == 3:
        pred = pred.reshape(-1, pred.shape[-1])
    if true.ndim == 3:
        true = true.reshape(-1, true.shape[-1])
    
    dtw_func = _accelerated_dtw if use_accelerated else _dtw
    dist_fun = manhattan_distances
    
    dtw_values = []
    for i in range(len(pred)):
        try:
            dist, _, _, _ = dtw_func(pred[i], true[i], dist_fun)
            dtw_values.append(dist)
        except Exception:
            dtw_values.append(np.nan)
    
    return np.nanmean(dtw_values)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def metric_with_dtw(pred, true, use_dtw=False, use_accelerated_dtw=True):
    """
    计算所有指标，包括可选的 DTW
    
    Args:
        pred: 预测值
        true: 真实值
        use_dtw: 是否计算 DTW 指标
        use_accelerated_dtw: 是否使用加速的 DTW 算法
    
    Returns:
        包含所有指标的字典
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    result = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe
    }
    
    if use_dtw:
        dtw_val = dtw_metric(pred, true, use_accelerated=use_accelerated_dtw)
        result['dtw'] = dtw_val
    
    return result
