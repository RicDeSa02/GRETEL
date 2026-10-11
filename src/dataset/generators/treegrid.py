import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

class TreeGrid(Generator):
    
    def init(self):

        self.dataset_name = "TreeGrid"
        self.num_instances = self.local_config['parameters'].get('num_instances', 1000)
        self.tree_height = self.local_config['parameters'].get('tree_height', 5)
        self.grid_size = self.local_config['parameters'].get('grid_size', (3, 3))
        self.seed = self.local_config['parameters'].get('seed', 0)
        self.rng = np.random.default_rng(self.seed)

        self.grid_size = (self.grid_size, self.grid_size) if isinstance(self.grid_size, int) else self.grid_size # Create a 2D grid if given a 1D value

        assert ((isinstance(self.num_instances, int)) and self.num_instances >= 1)
        assert ((isinstance(self.tree_height, float) or isinstance(self.tree_height, int)) and self.tree_height >= 1)
        assert ((isinstance(self.grid_size, float) or isinstance(self.grid_size, int)) and self.grid_size >= 1 or isinstance(self.grid_size, tuple))
        assert ((2**(self.tree_height + 1) - 1) > self.grid_size[0]*self.grid_size[1])

        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 1000)
        local_config['parameters']['tree_height'] = local_config['parameters'].get('tree_height', 5)
        local_config['parameters']['grid_size'] = local_config['parameters'].get('grid_size', (3, 3))

        

    def generate_dataset(self):

        nodes_count, edges_count = 0, 0

        for i in range(self.num_instances):
            # Randomly determine if the graph is going to contain a TreeGrid or just a tree
            has_treegrid = self.rng.integers(0, 2)  # 2 excluded
            graph_seed = int(self.rng.integers(0, 2**32 - 1))
            num_nodes = 2**(self.tree_height + 1) - 1  # Number of nodes in the tree
            
            if has_treegrid: # Generate grid and tree

                grid = nx.grid_graph(dim=self.grid_size)
                nx.set_node_attributes(grid, 1, 'grid')  # Mark grid nodes
                
                # Convert grid to a format where nodes are numbered consecutively
                grid = nx.convert_node_labels_to_integers(grid)

                # Generate tree
                tree_nodes = num_nodes - len(grid.nodes)  # Ensure tree has enough nodes to attach to grid
                tree = nx.random_tree(n=tree_nodes, seed=graph_seed)
                nx.set_node_attributes(tree, 0, 'grid')  # Mark tree nodes
                
                # Combine grid and tree
                combined = nx.disjoint_union(tree, grid)    
                
                # Connect grid to tree by adding edges between the two components
                grid_start = len(tree.nodes)
                tree_root = 0  # Connect to root of the tree
                combined.add_edge(tree_root, grid_start)  # Connect first node of grid to root
                
                adj_matrix = nx.to_numpy_array(combined)
                label = 1  # Graph contains a TreeGrid
                features = np.ones(len(combined.nodes))  # Example feature: all ones
                
                draw_graph = combined
            
            else: # Generate a random tree
                tree = nx.random_tree(n=num_nodes, seed=graph_seed)
                adj_matrix = nx.to_numpy_array(tree)
                label = 0  # Graph is a random tree
                features = np.ones(num_nodes)
                draw_graph = tree

            if False: # Visualize the graph
                nx.draw(draw_graph, with_labels=True, node_color=features, cmap=plt.cm.cool, edge_color='gray')
                plt.show()
            
            # Count nodes and edges
            graph = nx.from_numpy_array(adj_matrix)
            # print("Nodes:", graph.number_of_nodes(), "Edges:", graph.number_of_edges())
            nodes_count += graph.number_of_nodes()
            edges_count += graph.number_of_edges()
            # Append the instance to the dataset
            self.dataset.instances.append(GraphInstance(id=i, data=adj_matrix, label=label, node_features=features))
            # self.context.logger.info(f"Generated instance with id {i} and label={label}")    
    
        nodes_avg = nodes_count / self.num_instances
        edges_avg = edges_count / self.num_instances
        # print("Nodes avg:", nodes_avg, "Edges avg:", edges_avg)
        self.context.logger.info(f"Generated dataset - instances: {self.num_instances} - tree height: {self.tree_height} - grid size: {self.grid_size} - Nodes avg: {nodes_avg:.2f} - Edges avg: {edges_avg:.2f}")    


    def get_num_instances(self):
        return len(self.dataset.instances)