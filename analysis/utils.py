import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import pandas as pd
from evaluation.metrics import *


def removed_added_edges_eval(G: nx.Graph, G_rw: nx.Graph, loaded_datasets, rewiring_method, edge_stats_dict, eval= False):
       
    if eval:
        rows = []
        for dataset_name in loaded_datasets:
            for method in rewiring_method or [rewiring_method]:
                vals = edge_stats_dict[dataset_name][method]  # list of (pct_added, pct_removed)
                if len(vals) > 0:
                    arr = np.array(vals)  # shape (N,2)
                    mean_added   = arr[:,0].mean()
                    mean_removed = arr[:,1].mean()
                else:
                    mean_added = mean_removed = 0.0
                rows.append({
                    "Dataset": dataset_name,
                    "Method": method,
                    "% edges added (avg)": f"{mean_added:.2f}%",
                    "% edges removed (avg)": f"{mean_removed:.2f}%"
                })
        
        df = pd.DataFrame(rows)        
        return df
        
    else:
        G_unrewired = to_networkx(G, to_undirected=True)
        E0 = set((min(u,v), max(u,v)) for u,v in G_unrewired.edges())
        
        E1 = set((min(u,v), max(u,v)) for u,v in G_rw.edges())
                
        added   = E1 - E0
        removed = E0 - E1
        if len(E0) > 0:
            pct_added   = 100.0 * len(added)   / len(E0)
            pct_removed = 100.0 * len(removed) / len(E0)
        else:
            pct_added = pct_removed = 0.0
        
        return pct_added, pct_removed
        
    

def extract_curvature(curv_G_rewired: nx.Graph, og_graph: nx.Graph, dataset_name: str ):
    edges = list(curv_G_rewired.edges())
    if not edges:
        raise ValueError("[ERROR] Rewired graph has no edges, cannot extract curvature.")
    if edges:
        
        u, v = edges[0]  # First edge in the og_graph
        if "ricciCurvature" in curv_G_rewired[u][v]:
            curvatures = nx.get_edge_attributes(curv_G_rewired, "ricciCurvature").values()
            curvature_name = "ricciCurvature"
            
            G_nx = to_networkx(og_graph, to_undirected=True)  # Convert PyG og_graph to NetworkX
            metric_unrewired = GraphMetrics(G_nx, dataset_name)
            _, curv_G_unrewired = metric_unrewired.get_orc()
            curvature_unrewired = nx.get_edge_attributes(curv_G_unrewired, curvature_name).values()
        elif "formanCurvature" in curv_G_rewired[u][v]:
            curvatures = nx.get_edge_attributes(curv_G_rewired, "formanCurvature").values()
            curvature_name = "formanCurvature"
            
            G_nx = to_networkx(og_graph, to_undirected=True)  # Convert PyG og_graph to NetworkX
            metric_unrewired = GraphMetrics(G_nx, dataset_name)
            _, curv_G_unrewired = metric_unrewired.get_forman()
            curvature_unrewired = nx.get_edge_attributes(curv_G_unrewired, curvature_name).values()
        else:
            curvatures = []
    else:
        assert False, "No curvature found in the og_graph"
        
    unrew_arr = np.array(list(curvature_unrewired))
    rew_arr = np.array(list(curvatures))
    if len(unrew_arr) == len(rew_arr) and len(unrew_arr) > 0:
        curv_diff = np.mean(np.abs(unrew_arr - rew_arr))
    else:
        curv_diff = 0.0
        
    return curv_diff, curvatures, curv_G_unrewired, curvature_unrewired, curvature_name