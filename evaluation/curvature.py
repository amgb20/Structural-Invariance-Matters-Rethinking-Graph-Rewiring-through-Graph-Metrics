# from GraphraphRicciCurvature.FormanRicci import FormanRicci
import networkx as nx
import numpy as np
from GraphRicciCurvature import FormanRicci


"""
Forman-Ricci curvature F(i,j)

A computationally simpler alternative to Ollivier-Ricci, used for edge-based curvature analysis. While F(i,j) is given in terms of combinatorial quantities, results are scarce and the definition is biased towards negative curvature.
"""
def get_forman_curve(Graphraph):
    # Initialize the Forman-Ricci curvature calculator
    frc = FormanRicci(Graphraph)
    
    # Compute the Forman-Ricci curvature
    frc.compute_ricci_curvature()
    
    # Extract the curvature values for each edge
    curvatures = [data['formanCurvature'] for _, _, data in frc.Graphraph.edges(data=True)]
    
    # Calculate the average curvature
    avg_frc = np.mean(curvatures)
    
    return avg_frc

"""
Forman-Ricci curvature F(i,j)

A computationally simpler alternative to Ollivier-Ricci, used for edge-based curvature analysis. While F(i,j) is given in terms of combinatorial quantities, results are scarce and the definition is biased towards negative curvature.
"""
def get_Forman_curve(Graph: nx.Graph):
    curvature = {}
    for u, v in Graph.edges():
        k_u = Graph.degree[u]
        k_v = Graph.degree[v]
        curvature[(u, v)] = 4 - (k_u + k_v)

        avg_curvature = np.mean(list(curvature.values()))
        return avg_curvature