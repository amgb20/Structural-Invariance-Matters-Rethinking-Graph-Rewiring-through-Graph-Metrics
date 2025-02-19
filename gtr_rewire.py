#Test file to apply GTR Rewiring

from gtr import gtr_rewire
from torch_geometric.datasets import TUDataset

datasets = ["REDDIT-BINARY", "IMDB-BINARY", "MUTAG", "ENZYMES", "PROTEINS", "COLLAB"]

data_dict = {name: TUDataset(root=f'./tmp/'+name, name=name) for name in datasets}

gtr_rewired_graphs = {
    name: [gtr_rewire(data) for data in dataset]
    for name, dataset in data_dict.items()
}

for name, rewired_graphs in gtr_rewired_graphs.items():
    print(f"Rewired {len(rewired_graphs)} graphs in {name}")