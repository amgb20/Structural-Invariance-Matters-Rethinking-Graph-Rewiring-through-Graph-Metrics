
from rewiring.borf.borf import *
from rewiring.sdrf.sdrf import *
from rewiring.laser.laser import *
from rewiring.fosr.fosr import *
from rewiring.digl.digl import *

class rewiring_call:
    def __init__(self, G: nx.Graph, dataset_name: str):
        self.G = G
        self.dataset_name = dataset_name
    

    def borf_rewiring(self):
        # Get first graph from dataset
        # Ensure node features exist
        if not hasattr(self.G, 'x') or self.G.x is None:
            self.G.x = torch.ones((self.G.num_nodes, 1))
        # print(f"📊 First graph from {self.dataset_name}: {self.G}")
        # print(self.G) ## --> REDDIT-BINARY: Data(edge_index=[2, 480], y=[1], num_nodes=218)
        
        # # Apply BORF rewiring using borf3
        # print(f"🔄 Applying BORF on {self.dataset_name}...")
        edge_index, edge_type = borf3(
            data=self.G,
            loops=50,
            batch_add=4,
            batch_remove=2,
            is_undirected=True,
            device=None,  
            save_dir='rewired_graphs',
            dataset_name= self.dataset_name,
            graph_index=0,
            debug=True
        )

        # Convert the edge_index to NumPy and then NetworkX graph
        rewired_edge_index = edge_index.numpy().T
        rewired_G_borf = nx.Graph()
        rewired_G_borf.add_edges_from(rewired_edge_index)

        # print(f"✅ Rewiring complete! {self.dataset_name} now has {rewired_G_borf.number_of_edges()} edges.")
        
        return rewired_G_borf

    def sdrf_rewiring(self):
        
        # Apply SDRF rewiring
        print(f"🔄 Applying SDRF on {self.dataset_name}...")
        sdrf = SDRFTransform(num_iterations=50, dataset=self.dataset_name)
        
        rewired_graph_sdrf = sdrf.transform(self.G)
        
        # Extract rewired edges
        rewired_edge_sdrf_index = rewired_graph_sdrf.edge_index.numpy().T

        # Convert to NetworkX for evaluation
        rewired_G_sdrf = nx.Graph()
        rewired_G_sdrf.add_edges_from(rewired_edge_sdrf_index)

        # print(f"✅ Rewiring complete! {self.dataset_name} now has {rewired_G_sdrf.number_of_edges()} edges.")
        
        return rewired_G_sdrf

    def fosr_rewiring(self):
        
        # Apply FOSR rewiring
        print(f"🔄 Applying FOSR on {self.dataset_name}...")
        fosr = FOSRTransform(num_snapshots=1, num_iterations=50, initial_power_iters=5, dataset=self.dataset_name)
        rewired_graph_fosr = fosr.transform(self.G)

        # Extract rewired edges
        rewired_edge_fosr_index = rewired_graph_fosr.edge_index.numpy().T

        # Convert to NetworkX for evaluation
        rewired_G_fosr = nx.Graph()
        rewired_G_fosr.add_edges_from(rewired_edge_fosr_index)

        print(f"✅ Rewiring complete! {self.dataset_name} now has {rewired_G_fosr.number_of_edges()} edges.")
        
        return rewired_G_fosr

    def des_rewiring(self, dataset_loader):
        
        # Get first graph (ensure fresh copy)
        original_G = dataset_loader.get_graph_info(self.dataset_name, 0)

        # Ensure the graph is copied before applying rewiring (DES modifies in-place)
        rewired_G_des = original_G.copy()
        
        # Apply Double Edge Swap rewiring
        print(f"🔄 Applying Double Edge Swap on {self.dataset_name}...")
        
        # checker for the number of edges
        if rewired_G_des.number_of_edges() < 100:
            print(f"❌ {self.dataset_name} has less than 100 edges. Adapting to its max number of edge")
            nswap = rewired_G_des.number_of_edges()
        else:
            nswap = 100
        rewired_G_des = nx.double_edge_swap(rewired_G_des, nswap=nswap, max_tries=500)

        print(f"✅ Rewiring complete! {self.dataset_name} now has {rewired_G_des.number_of_edges()} edges.")
        
        return rewired_G_des

    def ppr_rewiring(self):
        # Get first graph from dataset
        # Ensure node features exist
        if not hasattr(self.G, 'x') or self.G.x is None:
            self.G.x = torch.ones((self.G.num_nodes, 1))
        print(f"📊 First graph from {self.dataset_name}: {self.G}")
        print(self.G) ## --> REDDIT-BINARY: Data(edge_index=[2, 480], y=[1], num_nodes=218)
        
        # Apply PPR rewiring
        print(f"🔄 Applying PPR on {self.ataset_name}...")
        rewired_graph_ppr = rewire_digl(self.G, alpha=0.1, k=128) # returns data.edge_index
            
        # Extract rewired edges
        rewired_edge_ppr_index = rewired_graph_ppr.edge_index.numpy().T

        # Convert to NetworkX for evaluation
        rewired_G_ppr = nx.Graph()
        rewired_G_ppr.add_edges_from(rewired_edge_ppr_index)

        print(f"✅ Rewiring complete! {self.dataset_name} now has {rewired_G_ppr.number_of_edges()} edges.")
        
        return rewired_G_ppr
    
    def laser_rewiring(self):
        # add the dataset_name to the self.G object
        self.G.dataset_name = self.dataset_name
        
        # Apply LASER rewiring
        print(f"🔄 Applying LASER on {self.dataset_name}...")
        laser = LaserGlobalTransform(dataset=self.dataset_name)
        
        rewired_graph_laser = laser.transform(self.G)
        
        # Extract rewired edges
        rewired_edge_laser_index = rewired_graph_laser.edge_index.numpy().T

        # Convert to NetworkX for evaluation
        rewired_G_laser = nx.Graph()
        rewired_G_laser.add_edges_from(rewired_edge_laser_index)

        print(f"✅ Rewiring complete! {self.dataset_name} now has {rewired_G_laser.number_of_edges()} edges.")


