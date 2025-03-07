#Test file to apply GTR Rewiring

from rewiring_files import AddPrecomputedGTREdges, PrecomputeGTREdges
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset

datasets = ["REDDIT-BINARY", "IMDB-BINARY", "MUTAG", "ENZYMES", "PROTEINS", "COLLAB"]

#data_dict = {name: TUDataset(root=f'./tmp/'+name, name=name) for name in datasets}

def rewire_gtr(name, data_w_edge, num_edges, add_edges):
    #pre_transform = T.Compose([PrecomputeGTREdges(num_edges=num_edges)])
    #transform = T.Compose([AddPrecomputedGTREdges(num_edges=add_edges)])
    pre_transform = PrecomputeGTREdges(num_edges=num_edges)
    transform = AddPrecomputedGTREdges(num_edges=add_edges)

    #dataset = TUDataset(
    #    root="./tmp/",
    #    name=name,
    #    transform=transform,
    #    pre_transform=pre_transform
    #)
    #Adding precomputed edges if it doesn't have any
    #for data in dataset:
    #    if not hasattr(data, "precomputed_gtr_edges"):
    #        data.precomputed_gtr_edges = PrecomputeGTREdges(num_edges=30)(data).precomputed_gtr_edges

    #Trying just having the dataset sent in
    for i in range(len(data_w_edge)):
        if not hasattr(data_w_edge[i], "precomputed_gtr_edges"):
            #print(f"⚠️ Adding precomputed GTR edges for dataset: {name}")
            data_w_edge[i].precomputed_gtr_edges = pre_transform(data_w_edge[i]).precomputed_gtr_edges
            data_w_edge[i]["precomputed_gtr_edges"] = data_w_edge[i].precomputed_gtr_edges


    #print(type(data_w_edge))

    dataset = data_w_edge
    for data in data_w_edge:
        if not hasattr(data, "precomputed_gtr_edges"):
            print("Edges got lost")

    for data in dataset:
        data = transform(data)

    if all([
    hasattr(data, "precomputed_gtr_edges") and data.precomputed_gtr_edges.shape[1] == (2*num_edges)
    for data in dataset
    ]):
        print("Edges succesfully precomputed!")

    dataset_wo_edges = TUDataset(
    root="./tmp/",
    name=name,
    pre_transform=pre_transform
    )
    # Check that 40 edges have been added to each graph in the dataset
    if all([ 
        (data.edge_index.shape[1]-data_wo_edges.edge_index.shape[1]) == (2*add_edges)
        for data, data_wo_edges 
        in zip(dataset, dataset_wo_edges) 
    ]):
        print("Edges succesfully added!")

    return dataset

