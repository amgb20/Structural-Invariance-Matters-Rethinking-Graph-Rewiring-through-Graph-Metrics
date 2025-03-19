import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde

"""
This module provides functions to visualize network graphs using NetworkX and Matplotlib.
Functions:
    plot_dataset_net(Graph: nx.Graph, dataset_name: str):
        Plots a 2D visualization of the given graph using two different layouts: default and shell layout.
    plot_dataset_net_3D(Graph: nx.Graph, dataset_name: str):
        Plots a 3D visualization of the given graph using a spring layout.
"""

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
    # graph in 3D
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 3D positions for the nodes
    pos = nx.spring_layout(Graph, dim=3)

    # Extract node positions
    x_vals = [pos[node][0] for node in Graph.nodes()]
    y_vals = [pos[node][1] for node in Graph.nodes()]
    z_vals = [pos[node][2] for node in Graph.nodes()]

    # nodes
    ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o', s=50)

    # edges
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


def plot_graph_with_curvature(G: nx.Graph, title: str, curvature_name="formanCurvature" ):

    edge_curvs = nx.get_edge_attributes(G, curvature_name)
    if not edge_curvs:
        print(f"Warning: No edge attribute '{curvature_name}' found.")
        return

    curv_values = np.array(list(edge_curvs.values()))
    vmin, vmax = curv_values.min(), curv_values.max()
    if vmin == vmax:  # if all curvatures identical, offset a bit
        vmin, vmax = vmin - 0.5, vmax + 0.5

    fig = plt.figure(figsize=(14, 6))
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection='3d')


    pos2d = nx.kamada_kawai_layout(G)  

    edges_2d = []
    edge_colors_2d = []
    for (u, v) in G.edges():
        c = G[u][v][curvature_name]
        frac = (c - vmin)/(vmax - vmin)
        color = plt.cm.coolwarm(frac)
        edges_2d.append((u, v))
        edge_colors_2d.append(color)

    nx.draw_networkx_nodes(
        G, 
        pos2d, 
        node_size=80, 
        node_color='lightgray', 
        ax=ax2d
    )
    nx.draw_networkx_edges(
        G,
        pos2d,
        edgelist=edges_2d,
        edge_color=edge_colors_2d,
        ax=ax2d
    )
    ax2d.set_title(f"2D: {curvature_name}")
    ax2d.axis("off")


    pos3d = nx.spring_layout(G, dim=3, seed=42)  
    for (u, v) in G.edges():
        c = G[u][v][curvature_name]
        frac = (c - vmin)/(vmax - vmin)
        color = plt.cm.coolwarm(frac)
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

    ax3d.set_title(f"3D: {curvature_name}")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")


    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm, 
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])  # avoids a matplotlib warning
    cbar = fig.colorbar(sm, ax=[ax2d, ax3d], fraction=0.03, pad=0.08)
    cbar.set_label(curvature_name)

    plt.suptitle(f"Edge Curvature in 2D & 3D: {title}")
    plt.savefig(f"Edge Curvature in 2D & 3D: {title}")
    plt.show()