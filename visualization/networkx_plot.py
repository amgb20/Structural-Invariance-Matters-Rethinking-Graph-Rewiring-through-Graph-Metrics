import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde
import open3d as o3d
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# for ARI computation
from sklearn import preprocessing, metrics


def plot_dataset_net(Graph: nx.Graph, dataset_name: str):
    plt.figure(figsize=(12, 6))
    subax1 = plt.subplot(121)
    nx.draw(Graph, with_labels=True, font_weight='bold', node_size=50, font_size=8)
    subax1.set_title(f"{dataset_name} Graph")

    subax2 = plt.subplot(122)
    nx.draw_shell(Graph, with_labels=True, font_weight='bold', node_size=50, font_size=8)
    subax2.set_title(f"{dataset_name} Graph (Shell Layout)")

    plt.show()
    
def plot_dataset_net_3D(Graph: nx.Graph, dataset_name: str):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    pos = nx.spring_layout(Graph, dim=3)

    x_vals = [pos[node][0] for node in Graph.nodes()]
    y_vals = [pos[node][1] for node in Graph.nodes()]
    z_vals = [pos[node][2] for node in Graph.nodes()]

    ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o', s=50)

    for edge in Graph.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, c='r')

    ax.set_title(f"{dataset_name} Graph in 3D")
    plt.show()
    



def degree_dist(G: nx.Graph):
    if not isinstance(G, nx.Graph):
        raise ValueError("Input graph must be a NetworkX graph.")
    return np.array([deg for _, deg in G.degree()])

def plot_degree_dist(graphs_dict, dataset_name):

    plt.figure(figsize=(10, 6))
    x_vals_min, x_vals_max = float("inf"), float("-inf")
    
    if "UNREWIRED" not in graphs_dict or len(graphs_dict["UNREWIRED"]) == 0:
        raise ValueError("No unrewired graphs found. Cannot compute baseline degree distribution.")
    
    degrees_og = np.concatenate([degree_dist(G) for G in graphs_dict["UNREWIRED"]])
    degrees_og_log = np.log2(degrees_og + 1)  
    kde_og = gaussian_kde(degrees_og_log)
    
    for rewiring_method, graph_list in graphs_dict.items():
        all_degrees = np.concatenate([degree_dist(G) for G in graph_list])
        all_degrees_log = np.log2(all_degrees + 1)
        x_vals_min = min(x_vals_min, all_degrees_log.min())
        x_vals_max = max(x_vals_max, all_degrees_log.max())

    x_vals = np.linspace(x_vals_min, x_vals_max, 100)

    plt.fill_between(x_vals, kde_og(x_vals), alpha=0.3, color="blue")
    plt.plot(x_vals, kde_og(x_vals), label="Original Graph", color="blue")

    for rewiring_method, graph_list in graphs_dict.items():
        if rewiring_method == "UNREWIRED":
            continue  # Skip original, already plotted
        
        all_degrees = np.concatenate([degree_dist(G) for G in graph_list])
        all_degrees_log = np.log2(all_degrees + 1)
        kde_rw = gaussian_kde(all_degrees_log)
        W1_distance = wasserstein_distance(degrees_og_log, all_degrees_log)

        plt.fill_between(x_vals, kde_rw(x_vals), alpha=0.3, label=f"{rewiring_method} (W1={W1_distance:.4f})")
        plt.plot(x_vals, kde_rw(x_vals))

    # Labels & Titles
    plt.xlabel("Node Degree (log2 scale)")
    plt.ylabel("Density (KDE)")
    plt.title(f"Degree Distribution of {dataset_name}")

    # Legend
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Degree_Distribution of {dataset_name}")
    plt.show()
    
def plot_curv_histogram_compare(curv_dict, name_1, name_2):
    plt.figure(figsize=(9, 6))
    
    curv1 = curv_dict[name_1]
    curv2 = curv_dict[name_2]

    bins = np.linspace(
        min(min(curv1), min(curv2)),
        max(max(curv1), max(curv2)),
        30
    )

    plt.hist(curv1, bins=bins, alpha=0.6, label=name_1, color='blue', edgecolor='black')
    plt.hist(curv2, bins=bins, alpha=0.6, label=name_2, color='red', edgecolor='black')

    plt.xlabel('Edge Curvature')
    plt.ylabel('Frequency')
    plt.title(f"Curvature Distribution: {name_1} vs {name_2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    

def plot_curv_histogram(curvature_values, name: str):
    """Plots the histogram of Ricci curvatures for multiple graphs."""
    
    plt.figure(figsize=(8, 6))
    
    plt.hist(curvature_values, bins=20, edgecolor='black', alpha=0.75)
    plt.xlabel('Ricci curvature')
    plt.ylabel('Frequency')
    plt.title(f"Histogram of {name} (Aggregated Across Graphs)")
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"Histogram of {name} (Aggregated Across Graphs)")
    plt.show()

def plot_graph_curvature_subplots(
    G: nx.Graph,
    curvature_name: str = "formanCurvature",
    title: str = "Graph Curvature Comparison"
    ):
    edge_curvs = nx.get_edge_attributes(G, curvature_name)
    if not edge_curvs:
        print(f"[WARNING] No edge attribute '{curvature_name}' found in the graph.")
        return

    curv_values = np.array(list(edge_curvs.values()))
    vmin, vmax = curv_values.min(), curv_values.max()
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    cmap = plt.cm.get_cmap("RdBu_r")

    def curvature_to_color(c):
        frac = (c - vmin) / (vmax - vmin)
        return cmap(frac)


    fig = plt.figure(figsize=(14, 6))

    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    pos2d = nx.kamada_kawai_layout(G)

    edges_2d = list(G.edges())
    edge_colors_2d = [curvature_to_color(G[u][v][curvature_name]) for (u, v) in edges_2d]

    nx.draw_networkx_nodes(G, pos2d, node_size=80, node_color='lightgray', ax=ax2d)
    nx.draw_networkx_edges(G, pos2d, edgelist=edges_2d, edge_color=edge_colors_2d, ax=ax2d)

    ax2d.set_title("2D Kamada–Kawai Layout")
    ax2d.axis("off")

    pos3d = nx.spring_layout(G, dim=3, seed=42)

    for (u, v) in G.edges():
        c = G[u][v][curvature_name]
        color = curvature_to_color(c)
        xline = [pos3d[u][0], pos3d[v][0]]
        yline = [pos3d[u][1], pos3d[v][1]]
        zline = [pos3d[u][2], pos3d[v][2]]
        ax3d.plot(xline, yline, zline, color=color, alpha=0.9)

    node_xyz = np.array([pos3d[n] for n in G.nodes()])
    ax3d.scatter(
        node_xyz[:,0],
        node_xyz[:,1],
        node_xyz[:,2],
        s=50,
        c='lightgray',
        edgecolors='black'
    )

    ax3d.set_title("3D Spring Layout")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    cax = fig.add_axes([0.02, 0.2, 0.02, 0.6])  

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(curvature_name)

    plt.suptitle(title)
    plt.tight_layout(rect=[0.07, 0, 1, 1])  
    plt.show()
    
def plot_graph_manifold(
    G: nx.Graph,
    curvature_name: str = "formanCurvature",
    title: str = "Graph Manifold Plot",
    alpha_val: float = 1.5,
    smooth_iterations: int = 20
):
    edge_curvs = nx.get_edge_attributes(G, curvature_name)
    node_curv = {n: [] for n in G.nodes()}
    for (u, v), curv in edge_curvs.items():
        node_curv[u].append(curv)
        node_curv[v].append(curv)
    node_curv = {n: np.mean(vals) if vals else 0 for n, vals in node_curv.items()}
    
    pos = nx.spring_layout(G, dim=3, seed=42)
    nodes = list(G.nodes())
    points = np.array([pos[n] for n in nodes])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_val)
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    tree = cKDTree(points)
    _, vertex_node_indices = tree.query(vertices)
    curv_values = np.array([node_curv[nodes[i]] for i in vertex_node_indices])
    
    vmin, vmax = np.min(curv_values), np.max(curv_values)
    cmap = plt.cm.RdBu_r  # Blue for negative, Red for positive
    colors = cmap((curv_values - vmin) / (vmax - vmin))[:, :3]  # RGBA to RGB
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    triangles = np.asarray(mesh.triangles)
    mesh_polygons = [vertices[t] for t in triangles]
    face_colors = np.mean(colors[triangles], axis=1)  # Average vertex colors per face
    
    collection = Poly3DCollection(mesh_polygons, facecolors=face_colors, linewidths=0.1, edgecolors='k')
    ax.add_collection3d(collection)
    
    ax.auto_scale_xyz(vertices[:,0], vertices[:,1], vertices[:,2])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(f"Curvature ({curvature_name})")
    
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

