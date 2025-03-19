from scipy.sparse.csgraph import laplacian
from scipy.linalg import pinv, eigvalsh
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import pandas as pd
import numpy as np
from grakel.kernels import GraphletSampling
from grakel import Graph
from scipy.sparse.csgraph import laplacian
from scipy.linalg import pinv, eigvalsh
from scipy.stats import wasserstein_distance
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

class GraphMetrics:
    def __init__(self, G: nx.Graph, G_name: str):
        self.G = G
        self.G_name = G_name
    """
    Effective resistance

    The effective resistance is measures how easily information flows between pairs of nodes. If rewiring reduces effective resistance, it improves communication efficiency:

    $$[ R_{uv} = L^+{uu} + L^+{vv} - 2L^+_{uv} ]$$

    where $R_{ij}$ is the resistance between nodes $i$ and $j$ in the graph.

    """
    # TODO: might be worth checking if this implementation is correct compared: https://github.com/Fedzbar/laser-release/blob/main/laser/rewiring/dynamic/effective_resistance.py
    def get_eff_res(self):
        nodes = list(self.G.nodes())
        u = nodes[0]
        v = nodes[1]

        L = laplacian(nx.to_numpy_array(self.G), normed=False)
        L_pinv = pinv(L)
        return L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]
    
    def get_eff_res_v2(self):
        eff_res = nx.distance_measures.effective_graph_resistance(self.G)
        return eff_res
        
    
    """
    Forman curvature
    
    The Forman curvature measures how well the graph approximates a sphere. If rewiring changes the curvature, it may indicate a change in the global structure of the graph.
    """
    #Function to get curvature
    def get_Forman_curve(self):
        curvature = {}
        for u, v in self.G.edges():
            k_u = self.G.degree[u]
            k_v = self.G.degree[v]
            curvature[(u, v)] = 4 - (k_u + k_v)

            avg_curvature = np.mean(list(curvature.values()))
            return avg_curvature
        
    """
    A class to compute Forman-Ricci curvature for all nodes and edges in G
    """
    def get_Forman_curve_v2(self):
        forman = FormanRicci(self.G) 
        forman.compute_ricci_curvature() # Compute Forman-ricci curvature for all nodes and edges in G. Node curvature is defined as the average of all it’s adjacency edge
        
        forman_curvatures = [
        forman.G[u][v].get("formanCurvature", None) for u, v in forman.G.edges()
        ]
        
        # Remove None values in case some edges do not have curvature computed
        forman_curvatures = [c for c in forman_curvatures if c is not None]
        
        # Compute the average Forman Curvature
        avg_forman_curvature = sum(forman_curvatures) / len(forman_curvatures) if forman_curvatures else None
        
        return avg_forman_curvature, forman.G
    
    """
    Get Olliver Ricci curvature
    """
    def get_Olliver_Ricci_cuve(self):
        # how much mass remains at the original node when computing optimal transport for curvature estimation
        # When α = 0 → All mass is distributed to the neighboring nodes.
        # When α = 1 → No mass is moved, meaning the curvature is not meaningful (stays at its original node).
        # When α = 0.5 → The mass is evenly split: half remains at the node, and half is distributed to neighbors.
        orc = OllivierRicci(self.G, alpha=0) # set it to 0 as BORF method is doing... why? i don't know
        orc.compute_ricci_curvature()
        
        orc_curvatures = [
        orc.G[u][v].get("ricciCurvature", None) for u, v in orc.G.edges()
        ]
        
        # remove None values in case some edges do not have curvature computed
        orc_curvatures = [c for c in orc_curvatures if c is not None]
        
        # compute the average Forman Curvature
        avg_orc_curvature = sum(orc_curvatures) / len(orc_curvatures) if orc_curvatures else None
                
        return avg_orc_curvature, orc.G

    """
    Modularity

    Quantifies how well the graph divides into clusters or communities. If rewiring disrupts modularity, it may indicate loss of local structure.

    $$Q = \sum_{c=1}^{n}
        \left[ \frac{L_c}{m} - \gamma\left( \frac{k_c}{2m} \right) ^2 \right]$$

    where the sum iterates over all communities $c$, $m$ is the number of edges, $L_c$ is the number of intra-community links for community $c$, $k_c$ is the sum of degrees of the nodes in community $c$, and $\gamma$ is the resolution parameter.
    """
    def get_modularity(self):
        communities = list(greedy_modularity_communities(self.G))
        modularity = nx.algorithms.community.modularity(self.G, communities)
        return modularity

    """
    Graph Assortativity

    Measures if nodes tend to connect to others with similar degree. A change in assortativity indicates whether high/low-degree nodes are rewired differently.

    $$r = \frac{\sum_{ij} ij (e_{ij} - q_i q_j)}{\sigma^2}$$

    where $e_{ij}$ is the fraction of edges connecting nodes of degree $i$ and $j$, $q_i$ is the fraction of edges connected to nodes of degree $i$, and $\sigma^2$ is the variance of the degree distribution.
    """
    def get_assort(self):
        assortativity = nx.degree_assortativity_coefficient(self.G)
        return assortativity

    """
    Clustering Coefficient

    Measures how likely a node’s neighbors are to be connected to each other. Increasing clustering after rewiring may improve local information sharing.

    $$C = \frac{1}{n}\sum_{v \in G} c_v$$

    where :math:`n` is the number of nodes in `G`.

    """
    def get_clust_coeff(self):
        clustering_coeff = nx.average_clustering(self.G)
        return clustering_coeff

    """
    Graph Laplacian Eigenvalues (Spectral Gap):

    The second smallest eigenvalue (λ1\lambda_1λ1​) of the Laplacian indicates how well-connected the graph is. A larger spectral gap after rewiring suggests better robustness and connectivity.

    ADD formula

    where $f$ is a non-zero vector orthogonal to the all-ones vector $\mathbf{1}$, $L$ is the Laplacian matrix, and $f^T$ denotes the transpose of $f$.
    """
    def get_spec_gap(self):
        L = laplacian(nx.to_numpy_array(self.G), normed=True)
        eigenvalues = eigvalsh(L)
        spectral_gap = eigenvalues[1]
        return spectral_gap

    """
    Average Betweenness Centrality


    """
    def get_bet_cent(self):
        bet_cent = nx.betweenness_centrality(self.G)
        avg_bet = sum(bet_cent.values()) / len(bet_cent)
        return avg_bet
    
    """
    Graph Diameter

    ....
    """
    def get_diameter(self):
        if nx.is_connected(self.G):
            return nx.diameter(self.G)
        else:
            return max(nx.diameter(self.G.subgraph(c)) for c in nx.connected_components(self.G))
        
    # -----------------------------------------------------

    def get_all_metrics(self):
        metrics = {}  
        
        metrics["Diameter"] = self.get_diameter()
        metrics["Modularity"] = self.get_modularity()
        metrics["Assortativity"] = self.get_assort()
        metrics["Clustering Coefficient"] = self.get_clust_coeff()
        metrics["Spectral Gap"] = self.get_spec_gap()
        metrics["Average Betweenness Centrality"] = self.get_bet_cent()
        metrics["Forman Curvature"] = self.get_Forman_curve()

        if self.G_name != "MUTAG":
            metrics["Effective Resistance"] = self.get_eff_res()

        return metrics
    
    def get_forman(self):
        metrics = {}
        metrics["Forman Curvature"] = self.get_Forman_curve()
        avg_forman_curvature, forman_G = self.get_Forman_curve_v2()
        metrics["Forman Curvature V2"] = avg_forman_curvature
        return metrics, forman_G
    
    def get_orc(self):
        metrics = {}
        
        avg_orc_curvature, orc_G = self.get_Olliver_Ricci_cuve()
        metrics["Ollivier Ricci Curvature"] = avg_orc_curvature
        return metrics, orc_G
    
    def get_eff_res_compa(self):
        metrics = {}
        metrics["Effective Resistance"] = self.get_eff_res()
        metrics["Effective Resistance_v2"] = self.get_eff_res_v2()
        return metrics

    def get_metrics_dataframe(self):
        metrics = self.get_all_metrics()
        df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        df["Dataset"] = self.G_name
        return df
    





