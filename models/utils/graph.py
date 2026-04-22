"""
图结构构建工具

提供多种图邻接矩阵构建方法：
1. 从 CSV 构建（有向/无向）
2. 从距离阈值构建
3. 从相关性矩阵构建
4. 全连接图
"""
import torch
import pandas as pd
import numpy as np


# ============== 1. 从 CSV 构建（有向图/无向图） ==============

def build_directed_adjacency(adj_path, num_nodes):
    """
    从 CSV 构建有向邻接矩阵
    
    Args:
        adj_path: adjacent_gantry.csv 路径，包含 src_FID, nbr_FID 列
        num_nodes: 节点数量
    
    Returns:
        A: torch.Tensor, shape (N, N), 有向邻接矩阵
    """
    df = pd.read_csv(adj_path)
    
    A = torch.zeros(num_nodes, num_nodes)
    
    for _, row in df.iterrows():
        src = int(row['src_FID'])
        nbr = int(row['nbr_FID'])
        if src < num_nodes and nbr < num_nodes:
            A[src, nbr] = 1.0
    
    return A


def build_upstream_downstream_adjacency(adj_path, num_nodes):
    """
    从 CSV 构建上下游邻接矩阵（有向）
    
    假设 CSV 中的 (src_FID -> nbr_FID) 表示下游方向
    
    Args:
        adj_path: adjacent_gantry.csv 路径
        num_nodes: 节点数量
    
    Returns:
        A_down: torch.Tensor, shape (N, N), 下游邻接（src -> nbr）
        A_up: torch.Tensor, shape (N, N), 上游邻接（nbr -> src）
    """
    df = pd.read_csv(adj_path)
    
    A_down = torch.zeros(num_nodes, num_nodes)
    A_up = torch.zeros(num_nodes, num_nodes)
    
    for _, row in df.iterrows():
        src = int(row['src_FID'])
        nbr = int(row['nbr_FID'])
        if src < num_nodes and nbr < num_nodes:
            # src -> nbr 是下游
            A_down[src, nbr] = 1.0
            # nbr -> src 是上游
            A_up[nbr, src] = 1.0
    
    return A_down, A_up


def build_undirected_adjacency(adj_path, num_nodes, symmetric=True):
    """
    从 CSV 构建无向邻接矩阵
    
    Args:
        adj_path: adjacent_gantry.csv 路径
        num_nodes: 节点数量
        symmetric: 是否强制对称（如果 CSV 只记录单向关系）
    
    Returns:
        A: torch.Tensor, shape (N, N), 无向邻接矩阵
    """
    df = pd.read_csv(adj_path)
    
    A = torch.zeros(num_nodes, num_nodes)
    
    for _, row in df.iterrows():
        src = int(row['src_FID'])
        nbr = int(row['nbr_FID'])
        if src < num_nodes and nbr < num_nodes:
            A[src, nbr] = 1.0
            if symmetric:
                A[nbr, src] = 1.0
    
    return A


# ============== 2. 从距离矩阵构建 ==============

def build_distance_threshold_adjacency(coords, threshold, directed=False):
    """
    从坐标构建距离阈值邻接矩阵
    
    Args:
        coords: numpy array, shape (N, 2) 或 (N, 3)，每个节点的坐标
        threshold: 距离阈值，超过则不连边
        directed: 是否为有向图（默认无向）
    
    Returns:
        A: torch.Tensor, shape (N, N)
    """
    n = len(coords)
    A = torch.zeros(n, n)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= threshold:
                A[i, j] = 1.0
                if not directed:
                    A[j, i] = 1.0
    
    return A


def build_distance_weighted_adjacency(coords, sigma=0.1, directed=False):
    """
    从坐标构建距离加权邻接矩阵（高斯核）
    
    Args:
        coords: numpy array, shape (N, 2) 或 (N, 3)
        sigma: 高斯核带宽参数
        directed: 是否为有向图
    
    Returns:
        A: torch.Tensor, shape (N, N), 边权重为 exp(-dist^2 / sigma^2)
    """
    n = len(coords)
    A = torch.zeros(n, n)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            A[i, j] = np.exp(-dist ** 2 / sigma ** 2)
            if not directed:
                A[j, i] = A[i, j]
    
    return A


# ============== 3. 从相关性矩阵构建 ==============

def build_correlation_adjacency(data, threshold=0.5, method='pearson'):
    """
    从时序数据构建相关性邻接矩阵
    
    Args:
        data: numpy array, shape (T, N)，T 个时间步，N 个节点
        threshold: 相关性阈值，超过则连边
        method: 'pearson' 或 'spearman'
    
    Returns:
        A: torch.Tensor, shape (N, N)
    """
    n = data.shape[1]
    
    if method == 'pearson':
        corr = np.corrcoef(data.T)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr, _ = spearmanr(data)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 处理 NaN
    corr = np.nan_to_num(corr, nan=0.0)
    
    # 二值化
    A = (corr > threshold).astype(float)
    np.fill_diagonal(A, 0)  # 去掉自环
    
    return torch.tensor(A)


# ============== 4. 全连接图 ==============

def build_fully_connected_adjacency(num_nodes, exclude_self=True):
    """
    构建全连接邻接矩阵
    
    Args:
        num_nodes: 节点数量
        exclude_self: 是否排除自环
    
    Returns:
        A: torch.Tensor, shape (N, N)
    """
    A = torch.ones(num_nodes, num_nodes)
    if exclude_self:
        A = A - torch.eye(num_nodes)
    return A


# ============== 5. 归一化方法 ==============

def normalize_adjacency(A, method='sym'):
    """
    归一化邻接矩阵
    
    Args:
        A: torch.Tensor, shape (N, N)
        method: 'sym' (对称归一化) 或 'row' (行归一化)
    
    Returns:
        A_norm: torch.Tensor, shape (N, N)
    """
    if method == 'sym':
        # 对称归一化: D^{-1/2} A D^{-1/2}
        d = A.sum(dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt
    
    elif method == 'row':
        # 行归一化: D^{-1} A
        d = A.sum(dim=1)
        d_inv = torch.pow(d, -1)
        d_inv[torch.isinf(d_inv)] = 0.0
        D_inv = torch.diag(d_inv)
        return D_inv @ A
    
    else:
        raise ValueError(f"Unknown method: {method}")


def add_self_loops(A):
    """为邻接矩阵添加自环"""
    n = A.shape[0]
    return A + torch.eye(n)


# ============== 便捷函数 ==============

def get_phys_adjacency(adj_path, num_nodes, directed=True):
    """
    获取物理邻接矩阵的便捷函数
    
    Args:
        adj_path: 邻接关系 CSV 路径
        num_nodes: 节点数量
        directed: 是否返回有向图（True 返回 A_down, A_up；False 返回单个 A）
    
    Returns:
        如果 directed=True: (A_down, A_up)
        如果 directed=False: A (无向)
    """
    if directed:
        return build_upstream_downstream_adjacency(adj_path, num_nodes)
    else:
        return build_undirected_adjacency(adj_path, num_nodes)


def get_d_matrix(adj_path, num_nodes, device='cpu'):
    """
    获取 D 矩阵的便捷函数（D = I - S^T）
    
    Args:
        adj_path: adjacent_gantry.csv 路径
        num_nodes: 节点数量
        device: torch device
    
    Returns:
        D: torch.Tensor, shape (N, N)
    """
    A_down, _ = build_upstream_downstream_adjacency(adj_path, num_nodes)
    D = torch.eye(num_nodes) - A_down
    return D.to(device)
