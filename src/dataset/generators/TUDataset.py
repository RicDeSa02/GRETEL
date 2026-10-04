import torch, numpy as np
from os.path import join,exists
from os import makedirs

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

from torch_geometric.datasets import TUDataset as downloader

class TUDataset(Generator):

    def prepare_data(self):
        base_path = join(self.context.working_store_path,self.dataset_name)
        pkl_path = join(base_path, f'{self.dataset_name}.pkl')
        self.context.logger.info("Dataset Data Path:\t" + base_path)

        if not exists(pkl_path):
            # self.context.logger.info("Downloading " + self.dataset_name + "...")
            makedirs(base_path, exist_ok=True)
            if self.dataset_name == "DBLP_v1":
                dataset = downloader(base_path, name=self.dataset_name, use_node_attr=False, use_edge_attr=True)
                # Convert one-hot node labels → single integer feature BEFORE saving
                compact = []
                for i in range(len(dataset)):
                    data = dataset[i]   # this gives a Data object
                    # if one-hot was created, reduce it
                    if hasattr(data, 'x') and data.x is not None and data.x.dim() == 2 and data.x.size(1) > 1000:
                        idx = data.x.argmax(dim=1)
                        data.x = idx.unsqueeze(1).to(torch.float32)
                    if hasattr(data, 'node_labels'):
                        del data.node_labels
                    compact.append(data)
                torch.save(compact, join(base_path, f'{self.dataset_name}.pkl'))
                self.context.logger.info(f"Saved dataset {self.dataset_name} in {join(base_path, f'{self.dataset_name}.pkl')}.")
                return base_path
            else:
                dataset = downloader(base_path, name=self.dataset_name, use_node_attr=True, use_edge_attr=True)
                # dataset = TUDataset(base_path, name=self.dataset_name, use_node_attr=True, use_edge_attr=True)
                torch.save(dataset, pkl_path)
                self.context.logger.info(f"Saved dataset {self.dataset_name} in {pkl_path}.")
        return base_path        
       
    
    def init(self):
        self.dataset_name = self.local_config['parameters']['alias']
        base_path = self.prepare_data()
        # read the dataset and process it
        self.read_file = join(base_path, f'{self.dataset_name}.pkl')
        self.generate_dataset()

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.populate()

    def populate(self):
        # input(self.dataset.__dict__)
        DATASETS_WITOHOUT_NODE_ATTR = {"COLLAB", "IMDB-MULTI", "IMDB-BINARY"}
        CHEMICAL_DATASETS = {"MUTAG", "Mutagenicity", "AIDS", "BZR", "COX2", "BBBP", "DHFR", "PTC_MR", "PTC_FM", "PTC_FR", "PTC_MM", "NCI1", "NCI109"}
        data = torch.load(self.read_file)
        # features_map = {f'attribute_{i}': i for i in range(data[0].x.size(1))} if self.dataset_name not in DATASETS_WITOHOUT_NODE_ATTR else {f'attribute_{i}': i for i in range(data[0].num_nodes)}
        features_map = {f'attribute_{i}': i for i in range(data[0].x.size(1))} if self.dataset_name not in DATASETS_WITOHOUT_NODE_ATTR else {}
        self.dataset.node_features_map = features_map

        # TODO edge_map, graph_map

        for id, instance in enumerate(data):
            if self.dataset_name in DATASETS_WITOHOUT_NODE_ATTR:
                # print(f"instance.num_nodes.shape: {instance.num_nodes}")
                # input()
                # adj_matrix = torch.zeros((instance.num_nodes, instance.num_nodes), dtype=torch.float)

                # instance.x = torch.zeros((instance.num_nodes, 1))
                # instance.x = torch.zeros((instance.num_nodes))
                # instance.x = torch.tensor([])
                instance.x = torch.empty((instance.num_nodes, 0))
            
            adj_matrix = torch.zeros((instance.num_nodes, instance.num_nodes), dtype=torch.float)
            adj_matrix[instance.edge_index[0], instance.edge_index[1]] = 1.0 

            if adj_matrix.shape[0] == 0:
                print(f"Skipping instance {id} with empty adjacency matrix.")
                continue
                
            edge_features = None
            try:
                edge_features = instance.edge_weights.numpy()
            except AttributeError:
                self.context.logger.info(f'Instance id = {id} does not have edge features.')

            # if self.dataset_name == "TRIANGLES": print("TRIANGLES dataset detected, adjusting labels by -1")
            label = instance.y.item()-1 if self.dataset_name == "TRIANGLES" else instance.y.item()

            if self.dataset_name in CHEMICAL_DATASETS:
                atom_types = np.argmax(instance.x.numpy(), axis=1).astype(float)  # shape (N,)
                n_atom_types = instance.x.shape[1]
            else:
                atom_types = None
                n_atom_types = None
            # input(atom_types)

            self.dataset.instances.append(GraphInstance(id=id, 
                                                        label=label, 
                                                        data=adj_matrix.numpy(),
                                                        graph_features=None,
                                                        node_features=instance.x.numpy(),
                                                        edge_features=edge_features,
                                                        atom_types=atom_types,
                                                        n_atom_types = n_atom_types
                                                        ))