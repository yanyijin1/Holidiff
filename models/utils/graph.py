import torch
import pandas as pd
import os


def build_laplacian_from_adjacency(adj_path, num_nodes=30):
    """
    从邻接关系构建零阶拉普拉斯矩阵（L = D_deg - A）
    
    参数:
        adj_path:   adjacent_gantry.csv 路径
        num_nodes:  节点数量
    
    返回:
        L: torch.Tensor, shape (N, N)
    """
    df = pd.read_csv(adj_path)
    
    # 构建下游移位矩阵 S^T
    ST = torch.zeros(num_nodes, num_nodes)
    
    for _, row in df.iterrows():
        src = int(row['src_FID'])
        nbr = int(row['nbr_FID'])
        ST[src, nbr] = 1.0
    
    # D = I - S^T （后向差分算子）
    D = torch.eye(num_nodes) - ST
    return D


def build_d_matrix_from_adjacency(adj_path, num_nodes=30):
    """
    从邻接关系构建 D 矩阵（D = I - S^T，后向差分算子）
    
    参数:
        adj_path:   adjacent_gantry.csv 路径
        num_nodes:  节点数量
    
    返回:
        D: torch.Tensor, shape (N, N)
    """
    return build_laplacian_from_adjacency(adj_path, num_nodes)


def build_d_matrix_linear(num_nodes, device='cpu'):
    """
    线性拓扑的 D 矩阵（D = I - S^T）
    
    显式形式（N=4）：
    D = [[1, -1,  0,  0],
         [0,  1, -1,  0],
         [0,  0,  1, -1],
         [0,  0,  0,  1]]
    
    参数:
        num_nodes: 节点数量
        device: 设备
    
    返回:
        D: torch.Tensor, shape (N, N)
    """
    # 构建下游移位矩阵 S^T
    ST = torch.zeros(num_nodes, num_nodes, device=device)
    for i in range(num_nodes - 1):
        ST[i, i + 1] = 1.0  # 节点i指向i+1
    
    # D = I - S^T
    D = torch.eye(num_nodes, device=device) - ST
    return D


def get_d_matrix(device='cuda'):
    """
    获取 D 矩阵的便捷函数（默认使用邻接关系构建）
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adj_path = os.path.join(current_dir, 'data', 'adjacent_gantry.csv')
    
    D = build_d_matrix_from_adjacency(adj_path, num_nodes=30)
    return D.to(device)
