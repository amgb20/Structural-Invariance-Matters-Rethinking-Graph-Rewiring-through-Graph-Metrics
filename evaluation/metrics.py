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

        if self.G_name != "MUTAG":
            metrics["Effective Resistance"] = self.get_eff_res()

        return metrics

    def get_metrics_dataframe(self):
        metrics = self.get_all_metrics()
        df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        df["Dataset"] = self.G_name
        return df
    





