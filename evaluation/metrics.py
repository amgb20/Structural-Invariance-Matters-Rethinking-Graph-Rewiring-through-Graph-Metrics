from scipy.sparse.csgraph import laplacian
from scipy.linalg import pinv, eigvalsh
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


"""
Graph Diameter

....
"""
def get_diameter(G):
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        diameter = max(nx.diameter(G.subgraph(c)) for c in nx.connected_components(G))

    return diameter

"""
Effective resistance

The effective resistance is measures how easily information flows between pairs of nodes. If rewiring reduces effective resistance, it improves communication efficiency:

$$[ R_{uv} = L^+{uu} + L^+{vv} - 2L^+_{uv} ]$$

where $R_{ij}$ is the resistance between nodes $i$ and $j$ in the graph.

"""
def get_eff_res(Graph: nx.Graph):
    nodes = list(Graph.nodes())
    u = nodes[0]
    v = nodes[1]

    L = laplacian(nx.to_numpy_array(Graph), normed=False)
    L_pinv = pinv(L)
    return L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]


"""
Modularity

Quantifies how well the graph divides into clusters or communities. If rewiring disrupts modularity, it may indicate loss of local structure.

$$Q = \sum_{c=1}^{n}
       \left[ \frac{L_c}{m} - \gamma\left( \frac{k_c}{2m} \right) ^2 \right]$$

where the sum iterates over all communities $c$, $m$ is the number of edges, $L_c$ is the number of intra-community links for community $c$, $k_c$ is the sum of degrees of the nodes in community $c$, and $\gamma$ is the resolution parameter.
"""
def get_modularity(G):
    communities = list(greedy_modularity_communities(G))
    modularity = nx.algorithms.community.modularity(G, communities)
    return modularity

"""
Graph Assortativity

Measures if nodes tend to connect to others with similar degree. A change in assortativity indicates whether high/low-degree nodes are rewired differently.

$$r = \frac{\sum_{ij} ij (e_{ij} - q_i q_j)}{\sigma^2}$$

where $e_{ij}$ is the fraction of edges connecting nodes of degree $i$ and $j$, $q_i$ is the fraction of edges connected to nodes of degree $i$, and $\sigma^2$ is the variance of the degree distribution.
"""
def get_assort(G):
    assortativity = nx.degree_assortativity_coefficient(G)
    return assortativity

"""
Clustering Coefficient

Measures how likely a node’s neighbors are to be connected to each other. Increasing clustering after rewiring may improve local information sharing.

$$C = \frac{1}{n}\sum_{v \in G} c_v$$

where :math:`n` is the number of nodes in `G`.

"""
def get_clust_coeff(G):
    clustering_coeff = nx.average_clustering(G)
    return clustering_coeff

"""
Graph Laplacian Eigenvalues (Spectral Gap):

The second smallest eigenvalue (λ1\lambda_1λ1​) of the Laplacian indicates how well-connected the graph is. A larger spectral gap after rewiring suggests better robustness and connectivity.

ADD formula

where $f$ is a non-zero vector orthogonal to the all-ones vector $\mathbf{1}$, $L$ is the Laplacian matrix, and $f^T$ denotes the transpose of $f$.
"""
def get_spec_gap(G):
    L = laplacian(nx.to_numpy_array(G), normed=True)
    eigenvalues = eigvalsh(L)
    spectral_gap = eigenvalues[1]
    return spectral_gap

"""
Average Betweenness Centrality


"""
def get_bet_cent(G):
    bet_cent = nx.betweenness_centrality(G)
    avg_bet = sum(bet_cent.values()) / len(bet_cent)
    return avg_bet