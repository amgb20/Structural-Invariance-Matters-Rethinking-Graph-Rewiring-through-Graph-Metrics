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

class GraphDistanceMetrics:
    def __init__(self, G: nx.Graph, G_rewired: nx.Graph, G_name: str):
        self.G = G
        self.G_rewired = G_rewired
        self.G_name = G_name
        
    def get_graph_edit_distance(self):
        return nx.graph_edit_distance(self.G, self.G_rewired)
        
    def get_jaccard_sim(self):
        orig_edge = set(self.G.edges())
        rew_edge = set(self.G_rewired.edges())

        intersection = len(orig_edge & rew_edge)
        union = len(orig_edge | rew_edge)

        return intersection / union if union != 0 else 1.0

    def get_laplac_spec_dist(self, p=2):
        G1 = self.G
        G2 = self.G_rewired
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
    def get_adj_spec_dist(self, p=2):
        G1 = self.G
        G2 = self.G_rewired
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
    def get_spec_norm_adj_diff(self):
        G1 = self.G
        G2 = self.G_rewired
        A1 = nx.adjacency_matrix(G1).toarray()
        A2 = nx.adjacency_matrix(G2).toarray()

        # Pad smaller matrix if graphs have different sizes
        max_nodes = max(A1.shape[0], A2.shape[0])
        A1 = np.pad(A1, ((0, max_nodes - A1.shape[0]), (0, max_nodes - A1.shape[1])), 'constant')
        A2 = np.pad(A2, ((0, max_nodes - A2.shape[0]), (0, max_nodes - A2.shape[1])), 'constant')

        return np.linalg.norm(A1 - A2, ord=2)  # Spectral norm (largest singular value)

    #Degree Distribution Difference using Wassertein Distance (????) Will look up
    def deg_dist_diff(self):
        G1 = self.G
        G2 = self.G_rewired
        degrees_G1 = np.array([d for _, d in G1.degree()])
        degrees_G2 = np.array([d for _, d in G2.degree()])

        return wasserstein_distance(degrees_G1, degrees_G2)

    #Graphlet Kernel Distance
    def get_graph_kern_dist(self):
        G1 = self.G
        G2 = self.G_rewired
        G1_grakel = Graph(list(G1.edges()))
        G2_grakel = Graph(list(G2.edges()))

        kernel = GraphletSampling()
        similarity = kernel.fit_transform([G1_grakel, G2_grakel])

        return 1 - similarity[0, 1]  # Distance = 1 - similarity

    #Shortest Path Length Distribution Difference
    def get_short_path_diff(self):
        G1 = self.G
        G2 = self.G_rewired
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
    def comparison_metrics(self):
        # # if they have the same # of graphs
        # if len(self.G) != len(self.G_rewired):
        #     print("Not the same dataset")
        # else:
        metrics = {
        #"Graph Edit Distance" : [],
        "Jaccard Similarity" : [],
        "Laplacian Spectrum Distance" : [],
        #"Adjacency Spectrum Distance" : [],
        "Spectral Norm of Adjacency Difference" : [],
        "Degree Distribution Distance" : [],
        "Graphlet Kernel Distance" : [],
        "Shortest Path Length Distribution Difference" : []
        }

        #metrics["Graph Edit Distance"](self.get_graph_edit_distance())
        metrics["Jaccard Similarity"]=(self.get_jaccard_sim())
        metrics["Laplacian Spectrum Distance"]=(self.get_laplac_spec_dist())
        #metrics["Adjacency Spectrum Distance"]=(self.get_adj_spec_dist(self))
        metrics["Spectral Norm of Adjacency Difference"]=(self.get_spec_norm_adj_diff())
        metrics["Degree Distribution Distance"]=(self.deg_dist_diff())
        metrics["Graphlet Kernel Distance"]=(self.get_graph_kern_dist())
        metrics["Shortest Path Length Distribution Difference"]=(self.get_short_path_diff())
        
        df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        df["Dataset"] = self.G_name
        return df