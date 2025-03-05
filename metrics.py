import torch
import importlib
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from rewiring_files import PrecomputeGTREdges, AddPrecomputedGTREdges 
import sys
import networkx as nx
import pandas as pd
from IPython.display import display
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import pinv, eigvalsh
from scipy.stats import wasserstein_distance
from grakel.kernels import GraphletSampling
from grakel import Graph

#Function to convert dataset to a NetworkX Representation
def make_G(dataset, num_graph):
    graph = dataset[num_graph]
    edge_index = graph.edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(edge_index)

    return G

#Function to get diameter
def get_diameter(G):
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        diameter = max(nx.diameter(G.subgraph(c)) for c in nx.connected_components(G))

    return diameter

#Function to get effective resistance
def get_eff_res(G):
    nodes = list(G.nodes())
    u = nodes[0]
    v = nodes[1]

    L = laplacian(nx.to_numpy_array(G), normed=False)
    L_pinv = pinv(L)
    return L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]

#Function to get modularity
from networkx.algorithms.community import greedy_modularity_communities

def get_modularity(G):
    communities = list(greedy_modularity_communities(G))
    modularity = nx.algorithms.community.modularity(G, communities)
    return modularity

#Function to get Graph Assortativity
def get_assort(G):
    assortativity = nx.degree_assortativity_coefficient(G)
    return assortativity

#Function to get clustering coefficient
def get_clust_coeff(G):
    clustering_coeff = nx.average_clustering(G)
    return clustering_coeff

#Function to get Spectral Gap
def get_spec_gap(G):
    L = laplacian(nx.to_numpy_array(G), normed=True)
    eigenvalues = eigvalsh(L)
    spectral_gap = eigenvalues[1]
    return spectral_gap

#Function to get curvature
def get_Forman_curve(G):
    curvature = {}
    for u, v in G.edges():
        k_u = G.degree[u]
        k_v = G.degree[v]
        curvature[(u, v)] = 4 - (k_u + k_v)

        avg_curvature = np.mean(list(curvature.values()))
        return avg_curvature
    
#Function to get average betweenness centrality
def get_bet_cent(G):
    bet_cent = nx.betweenness_centrality(G)
    avg_bet = sum(bet_cent.values()) / len(bet_cent)
    return avg_bet


#Overall function to complete all metrics for a specific dataset for just the first graph
def get_metrics(dataset):
    G = make_G(dataset, 0)

    print("Diameter: ", get_diameter(G))
    print("Effective Resistance: ", get_eff_res(G))
    print("Modularity: ", get_modularity(G))
    print("Assortativity: ", get_assort(G))
    print("Clustering Coefficient:", get_clust_coeff(G))
    print("Spectral Gap:", get_spec_gap(G))
    print("Forman Curvature:", get_Forman_curve(G))
    print("Average Betweenness Centrality:", get_bet_cent(G))


def get_avg_metrics(dataset):
    metrics = {
        "Diameter" : [],
        "Effective Resistance" : [],
        "Modularity" : [],
        "Assortativity" : [],
        "Clustering Coefficient" : [],
        "Spectral Gap" : [],
        "Forman Curvature" : [],
        "Average Betweenness Centrality" : []
    }
    diams, effres, moduls, assorts, clustcoeffs, specgaps, curves, betweens = [], [], [], [], [], [], [], []
    for i, _ in enumerate(dataset):
        G = make_G(dataset, i)

        metrics["Diameter"].append(get_diameter(G))
        metrics["Effective Resistance"].append(get_eff_res(G))
        metrics["Modularity"].append(get_modularity(G))
        #Sometimes Assortativity returns as nan
        if not np.isnan(get_assort(G)):
            metrics["Assortativity"].append(get_assort(G))
        metrics["Clustering Coefficient"].append(get_clust_coeff(G))
        metrics["Spectral Gap"].append(get_spec_gap(G))
        metrics["Forman Curvature"].append(get_Forman_curve(G))
        metrics["Average Betweenness Centrality"].append(get_bet_cent(G))

    for metric_name, values in metrics.items():
        cur_mean = np.mean(values)
        cur_std = np.std(values)
        print(f"{metric_name}: Mean: {cur_mean} Std Dev: {cur_std}")

    
def get_metrics_table(dataset, name):
    G = make_G(dataset)

    metrics = {
        "Diameter": get_diameter(G),
        "Effective Resistance": get_eff_res(G),
        "Modularity": get_modularity(G),
        "Assortativity": get_assort(G),
        "Clustering Coefficient": get_clust_coeff(G),
        "Spectral Gap": get_spec_gap(G),
        "Forman Curvature": get_Forman_curve(G),
        "Average Betweenness Centrality": get_bet_cent(G),
    }

    # Convert dictionary to a pandas DataFrame
    df = pd.DataFrame(metrics.items(), columns=["Metric", name])

    # Round to 3 decimal places
    df[name] = df[name].round(5)

    # Display the table
    display(df)

#Functions to measure structural changes specifically (not on individual graph, but pair of original graph and rewired graph)
#Graph Edit Distance
def get_graph_edit_distance(orig_G, rewired_G):
    return nx.graph_edit_distance(orig_G, rewired_G)

def get_jaccard_sim(orig_G, rewired_G):
    orig_edge = set(orig_G.edges())
    rew_edge = set(rewired_G.edges())

    intersection = len(orig_edge & rew_edge)
    union = len(orig_edge | rew_edge)

    return intersection / union if union != 0 else 1.0

def get_laplac_spec_dist(G1, G2, p=2):
    L1 = nx.laplacian_matrix(G1).toarray()
    L2 = nx.laplacian_matrix(G2).toarray()

    # Compute eigenvalues
    eigvals1 = np.sort(np.linalg.eigvalsh(L1))
    eigvals2 = np.sort(np.linalg.eigvalsh(L2))

    # Pad the smaller eigenvalue array to match size
    max_len = max(len(eigvals1), len(eigvals2))
    eigvals1 = np.pad(eigvals1, (0, max_len - len(eigvals1)), 'constant')
    eigvals2 = np.pad(eigvals2, (0, max_len - len(eigvals2)), 'constant')

    # Compute the p-norm distance
    return np.linalg.norm(eigvals1 - eigvals2, ord=p)

#Adjacency Spectrum Distance
def get_adj_spec_dist(G1, G2, p=2):
    A1 = nx.adjacency_matrix(G1).toarray()
    A2 = nx.adjacency_matrix(G2).toarray()

    eigvals1 = np.sort(np.linalg.eigvalsh(A1))
    eigvals2 = np.sort(np.linalg.eigvalsh(A2))

    # Pad the eigenvalue arrays to the same length
    max_len = max(len(eigvals1), len(eigvals2))
    eigvals1 = np.pad(eigvals1, (0, max_len - len(eigvals1)), 'constant')
    eigvals2 = np.pad(eigvals2, (0, max_len - len(eigvals2)), 'constant')

    return np.linalg.norm(eigvals1 - eigvals2, ord=p)

#Spectral Norm Adjacency Difference
def get_spec_norm_adj_diff(G1, G2):
    A1 = nx.adjacency_matrix(G1).toarray()
    A2 = nx.adjacency_matrix(G2).toarray()

    # Pad smaller matrix if graphs have different sizes
    max_nodes = max(A1.shape[0], A2.shape[0])
    A1 = np.pad(A1, ((0, max_nodes - A1.shape[0]), (0, max_nodes - A1.shape[1])), 'constant')
    A2 = np.pad(A2, ((0, max_nodes - A2.shape[0]), (0, max_nodes - A2.shape[1])), 'constant')

    return np.linalg.norm(A1 - A2, ord=2)  # Spectral norm (largest singular value)

#Degree Distribution Difference using Wassertein Distance (????) Will look up
def deg_dist_diff(G1, G2):
    degrees_G1 = np.array([d for _, d in G1.degree()])
    degrees_G2 = np.array([d for _, d in G2.degree()])

    return wasserstein_distance(degrees_G1, degrees_G2)

#Graphlet Kernel Distance
def get_graph_kern_dist(G1, G2):
    G1_grakel = Graph(list(G1.edges()))
    G2_grakel = Graph(list(G2.edges()))

    kernel = GraphletSampling()
    similarity = kernel.fit_transform([G1_grakel, G2_grakel])

    return 1 - similarity[0, 1]  # Distance = 1 - similarity

#Shortest Path Length Distribution Difference
def get_short_path_diff(G1, G2):
    def short_path_dist(G):
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        distribution = []
        for _, dist in lengths.items():
            distribution.extend(dist.values())  # Collect all shortest path lengths
        return np.array(distribution)
    
    dist_G1 = short_path_dist(G1)
    dist_G2 = short_path_dist(G2)

    return wasserstein_distance(dist_G1, dist_G2)

#Function to go through a rewired dataset and calculate average comparison metrics
def comparison_metrics(original_dataset, rewired_dataset):
    if len(original_dataset) != len(rewired_dataset):
        print("Not the same dataset")
    else:
        metrics = {
        "Graph Edit Distance" : [],
        "Jaccard Similarity" : [],
        "Laplacian Spectrum Distance" : [],
        "Adjacency Spectrum Distance" : [],
        "Spectral Norm of Adjacency Difference" : [],
        "Degree Distribution Distance" : [],
        "Graphlet Kernel Distance" : [],
        "Shortest Path Length Distribution Difference" : []
        }
        
        for i in range(len(original_dataset)):
            orig_G = make_G(original_dataset, i)
            rewired_G = make_G(rewired_dataset, i)

            metrics["Graph Edit Distance"].append(get_graph_edit_distance(orig_G, rewired_G))
            metrics["Jaccard Similarity"].append(get_jaccard_sim(orig_G, rewired_G))
            metrics["Laplacian Spectrum Distance"].append(get_laplac_spec_dist(orig_G, rewired_G))
            metrics["Adjacency Spectrum Distance"].append(get_adj_spec_dist(orig_G, rewired_G))
            metrics["Spectral Norm of Adjacency Difference"].append(get_spec_norm_adj_diff(orig_G, rewired_G))
            metrics["Degree Distribution Distance"].append(deg_dist_diff(orig_G, rewired_G))
            metrics["Graphlet Kernel Distance"].append(get_graph_kern_dist(orig_G, rewired_G))
            metrics["Shortest Path Length Distribution Difference"].append(get_short_path_diff(orig_G, rewired_G))

        for metric_name, values in metrics.items():
            cur_mean = np.mean(values)
            cur_std = np.std(values)
            print(f"{metric_name}: Mean: {cur_mean} Std Dev: {cur_std}")