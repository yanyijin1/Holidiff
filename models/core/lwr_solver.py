"""
core/lwr_solver.py - LWR 物理骨架

LWR (Lighthill-Whitham-Richards) 空间推演算子

This module is LOCKED - do not modify
"""

import torch
import torch.nn as nn
from torch import einsum
import math


class PatchLWROperatorLayer(nn.Module):
    """
    Patch-level LWR Operator Layer
    
    Implements the LWR (Lighthill-Whitham-Richards) conservation equation
    in the latent space for traffic flow prediction.
    
    Physics: The conservation equation states that the change in density
    is driven by the spatial gradient of flow.
    """
    def __init__(self, d_model):
        super().__init__()
        self.decode = nn.Linear(d_model, 2)
        self.project = nn.Linear(1, d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(1)

    def forward(self, x, d_op, eps=1e-5):
        """
        Args:
            x: tensor of shape (B, N, P, d_model)
            d_op: spatial operator matrix (N, N)
            eps: numerical stability constant
        Returns:
            Updated tensor with LWR physics applied
        """
        _, _, p, _ = x.shape
        dv_dq = self.decode(x)
        dv, dq = dv_dq[..., 0], dv_dq[..., 1]
        
        # LWR conservation: density change = spatial gradient of flow
        drho = dq / (dv.abs() + eps)
        
        # Propagate spatial information using D operator
        drho_next = torch.zeros_like(drho)
        drho_next[:, :, 0] = drho[:, :, 0]
        for i in range(p - 1):
            drho_next[:, :, i + 1] = drho[:, :, i] + torch.einsum('nm,bm->bn', d_op, dq[:, :, i])
        
        # Project back to model dimension
        proj = self.project(self.norm(drho_next.unsqueeze(-1)))
        return x + torch.sigmoid(self.gate) * proj


class LWRGraphOperator(nn.Module):
    """
    Graph-based LWR operator for arbitrary road network topologies.
    
    Uses the adjacency matrix D to model upstream-downstream relationships.
    """
    def __init__(self, d_model, num_nodes):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.v_decode = nn.Linear(d_model, 1)
        self.q_decode = nn.Linear(d_model, 1)
        self.proj = nn.Linear(1, d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        
    def build_d_matrix(self, edges):
        """
        Build the D matrix from edge list.
        
        Args:
            edges: tensor of shape (E, 2) with (upstream, downstream) pairs
        Returns:
            D matrix of shape (N, N)
        """
        N = self.num_nodes
        D = torch.zeros(N, N, device=edges.device)
        
        for i in range(edges.shape[0]):
            src, dst = edges[i]
            D[src, dst] = 1.0   # upstream -> downstream
            D[dst, src] = -1.0  # downstream <- upstream
            
        return D
        
    def forward(self, h, edges=None, d_matrix=None):
        """
        Apply LWR operator on graph structure.
        
        Args:
            h: hidden states (B, N, d_model)
            edges: optional edge list for building D
            d_matrix: pre-computed D matrix
        """
        if d_matrix is None and edges is not None:
            d_matrix = self.build_d_matrix(edges)
            
        v = self.v_decode(h).squeeze(-1)  # (B, N)
        q = self.q_decode(h).squeeze(-1)  # (B, N)
        
        # Compute density change from flow gradient
        dq = q  # flow
        drho = dq / (v.abs() + 1e-5)  # density from fundamental diagram
        
        # Spatial propagation
        drho_out = torch.zeros_like(drho)
        drho_out = drho + torch.matmul(d_matrix, drho) * 0.1
        
        proj = self.proj(drho_out.unsqueeze(-1))
        return h + torch.sigmoid(self.gate) * proj
