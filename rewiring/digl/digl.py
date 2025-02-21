import numpy as np
import torch
from torch_geometric.data import Data

def get_adj_matrix(dataset) -> np.ndarray:
    num_nodes = dataset.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.edge_index[0], dataset.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

"""
- alpha (0< alpha<1) is the personalized PR teleport probability, it controls the balance between local connections and long range dependencies
    If alpha is small (e.g., alpha=0.1)
        The PPR method spreads information more globally.
        Nodes are more likely to connect to distant nodes.
        Better for long-range dependencies in deep GNNs.
        
    If alpha is large (e.g., alpha=0.85)

        The PPR method focuses more on local neighborhood.
        Nodes mostly connect to their close neighbors.
        Better for preserving local structure.

- k selects the top-K strongest edges per node.
    Controls: The graph sparsity after PPR rewiring.
    Value Range: k > 0 (default values: k=64 or k=128).
    
    If k is small (e.g., k=32)

        Each node keeps only its 32 strongest edges.
        The graph remains sparse (good for efficiency).
        Some long-range connections may be lost.
    
    If k is large (e.g., k=256)
        Each node keeps more edges, making the graph denser.
        More long-range dependencies are preserved.
        The graph becomes computationally expensive.
""" 
def rewire_digl(base, alpha, k=None, eps=None):
    # generate adjacency matrix from sparse representation
    adj_matrix = get_adj_matrix(base)
    # obtain exact PPR matrix
    ppr_matrix = get_ppr_matrix(adj_matrix, alpha=alpha)

    if k != None:
            #print(f'Selecting top {k} edges per node.')
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)
    elif eps != None:
            #print(f'Selecting edges with weight greater than {eps}.')
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)
    else:
        raise ValueError

        # create PyG Data object
    edges_i = []
    edges_j = []
    edge_attr = []
    for i, row in enumerate(ppr_matrix):
        for j in np.where(row > 0)[0]:
            edges_i.append(i)
            edges_j.append(j)
            edge_attr.append(ppr_matrix[i, j])
    edge_index = [edges_i, edges_j]

    data = Data(
        x=base.x,
        edge_index=torch.LongTensor(edge_index),
        y=base.y
    )        
    return data
