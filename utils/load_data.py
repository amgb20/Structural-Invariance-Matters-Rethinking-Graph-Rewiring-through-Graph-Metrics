import networkx as nx
import torch
from torch_geometric.datasets import TUDataset
# from utils.tu_dataset import TUDataset
from torch_geometric.utils import to_networkx
import difflib
import os
from torch_geometric.datasets.zinc import ZINC


class GraphDatasetLoader:
    """
    A class to load multiple PyTorch Geometric datasets and convert them into NetworkX representations.
    It also provides automatic dataset name correction in case of misspellings.
    """

    # Define available datasets (MACRO)
    DATASET_BENCHMARK = {"REDDIT-BINARY", "IMDB-BINARY", "MUTAG", "ENZYMES", "PROTEINS", "COLLAB", "ZINC"}

    def __init__(self, dataset_names, edge_attr=False):
        """
        Initializes the loader with multiple datasets.
        :param dataset_names: List of dataset names (str)
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]  # Convert single string to list
        
        self.datasets = {}  # Stores PyG dataset objects
        self.networkx_graphs = {}  # Stores NetworkX graph lists
        self.first_graphs = {}  # Stores first graph per dataset
        self.edge_attr = edge_attr

        # Load each dataset
        for name in dataset_names:
            corrected_name = self.get_closest_dataset_name(name)
            self.datasets[corrected_name], self.first_graphs[corrected_name] = self.load_dataset(corrected_name)
            self.networkx_graphs[corrected_name] = self.convert_to_networkx(self.datasets[corrected_name])

    def get_closest_dataset_name(self, name):
        """
        If the dataset name is misspelled, suggest the closest valid dataset name.
        """
        closest_match = difflib.get_close_matches(name, self.DATASET_BENCHMARK, n=1, cutoff=0.6)
        if closest_match:
            print(f"⚠️ Warning: '{name}' not found. Did you mean '{closest_match[0]}'?")
            return closest_match[0]
        elif name in self.DATASET_BENCHMARK:
            return name
        else:
            raise ValueError(f"❌ Dataset '{name}' is not in the benchmark list: {self.DATASET_BENCHMARK}")
        
    def dataset_exists(self, name):
        """
        Checks if the dataset has already been downloaded in the `./data` directory.
        """
        dataset_path = f"./data/{name}"
        return os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0

    def load_dataset(self, name):
        """
        Load dataset from TUDataset, but only download if necessary.
        """
        if self.dataset_exists(name):
            print(f"✅ Dataset {name} already exists. Loading from disk...")
        else:
            print(f"📂 Downloading dataset: {name}...")
        
        if name == "ZINC":
            dataset = ZINC(root="datasets-test/ZINC/", subset=True, split="test")
        else:
            dataset = TUDataset(root='./data', name=name, use_edge_attr= self.edge_attr)
        
        return dataset, dataset[0]

    def convert_to_networkx(self, dataset):
        """
        Convert all graphs in a PyG dataset into NetworkX format.
        """
        nx_graphs = [to_networkx(data, to_undirected=True) for data in dataset]
        print(f"✅ Converted {len(nx_graphs)} graphs from {dataset.name} into NetworkX format.")
        return nx_graphs

    def get_graph_info(self, dataset_name, index=0):
        """
        Print details of a specific graph in a dataset.
        :param dataset_name: Name of the dataset (str)
        :param index: Index of the graph (default: 0)
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' is not loaded.")

        dataset = self.datasets[dataset_name]
        data = dataset[index]
        G = self.networkx_graphs[dataset_name][index]

        num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else "Unknown"

        print(f"\n📊 **Graph {index} from {dataset_name}**:")
        print(f"- Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"- Number of classes: {num_classes}")
        print(f"- Features per Node: {data.num_node_features}")
        print(f"- Graph Class Label: {data.y.item()}")

        return G

    def get_loaded_dataset_names(self):
        """
        Return the names of the loaded datasets.
        """
        return list(self.datasets.keys())

# dataset_loader = GraphDatasetLoader(["REDDIT-BINARY"])  # Misspelled "REDDIT-BINARY"

# loaded_datasets = dataset_loader.get_loaded_dataset_names()
# print("Loaded datasets:", loaded_datasets)

# G = dataset_loader.get_graph_info(loaded_datasets[0], index=0)
# G2 = dataset_loader.get_graph_info("IMDB-BINARY", index=1)