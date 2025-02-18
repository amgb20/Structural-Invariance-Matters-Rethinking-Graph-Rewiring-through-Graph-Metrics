import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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