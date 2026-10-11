import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

class BAShapes(Generator):
    
    def init(self):

        self.dataset_name = "BAShapes"
        self.num_instances = self.local_config['parameters'].get('num_instances', 1000)
        self.nodes_num = self.local_config['parameters'].get('nodes_num', 32) # Number of nodes in the graph
        self.edges_per_node = self.local_config['parameters'].get('edges_per_node', 1) # Number of edges to attach from new node to existing nodes
        self.seed = self.local_config['parameters'].get('seed', 0)
        self.rng = np.random.default_rng(self.seed)

        assert ((isinstance(self.num_instances, int)) and self.num_instances >= 1)
        assert ((isinstance(self.nodes_num, int)) and self.nodes_num >= 5)
        assert ((isinstance(self.edges_per_node, int)) and self.edges_per_node >= 0)

        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 1000)
        local_config['parameters']['nodes_num'] = local_config['parameters'].get('nodes_num', 32)
        local_config['parameters']['edges_per_node'] = local_config['parameters'].get('edges_per_node', 1)    

        
    def generate_dataset(self):
        def create_house_motif():
            """Create a house motif and assign node features based on position."""
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (0, 4)]  # House shape
            house = nx.Graph(edges)

            # Assign node features: 4 (rooftop), 3 (middle), 2 (bottom), 1 (ba-shape default node)
            features = {0: 3, # Middle node
                        1: 2, # Bottom node
                        2: 2, # Bottom node
                        3: 3, # Middle node
                        4: 4} # Roof node
            nx.set_node_attributes(house, features, "feat")
            return house

        nodes_count, edges_count = 0, 0

        for i in range(self.num_instances):
            # Randomly determine if the graph is going to contain a motif
            has_motif = self.rng.integers(0, 2)  # 2 excluded
            graph_seed = int(self.rng.integers(0, 2**32 - 1))
            
            if has_motif: # Generate BA graph with house motif
                
                ba_graph = nx.barabasi_albert_graph(n=self.nodes_num-5, m=self.edges_per_node, seed=graph_seed)

                # Generate the house motif
                house = create_house_motif()
                house = nx.convert_node_labels_to_integers(house, first_label=ba_graph.number_of_nodes())  # Re-label motif nodes

                # Randomize the house node to attach
                house_attachment_node = self.rng.choice(list(house.nodes))
                # Select a random node in BA graph to attach the house motif
                ba_attachment_node = self.rng.choice(list(ba_graph.nodes))
                
                # Combine BA graph and house motif
                combined = nx.disjoint_union(ba_graph, house)
                combined.add_edge(ba_attachment_node, house_attachment_node)  # Connect motif to the BA graph

                # Initialize features array for the combined graph
                features = np.ones(len(combined.nodes))  # Default feature: 1 for BA graph nodes

                # Assign features to nodes in the house motif
                for node in house.nodes:
                    features[node] = house.nodes[node].get('feat', 1)
                
                adj_matrix = nx.to_numpy_array(combined)
                label = 1  # Graph contains a house motif
                
                draw_graph = combined
            
            else: # Generate a plain BA graph
                
                ba_graph = nx.barabasi_albert_graph(n=self.nodes_num, m=self.edges_per_node, seed=graph_seed)
                
                adj_matrix = nx.to_numpy_array(ba_graph)
                label = 0  # Plain BA graph
                features = np.ones(len(ba_graph.nodes))  # Default feature: 1
                
                draw_graph = ba_graph
            
            # Visualize if desired  
            if False:  # Change to True to visualize
                nx.draw(draw_graph, with_labels=True, node_color=features, cmap=plt.cm.cool, edge_color='gray')
                plt.title(f"Graph with ID={i}, Label={label}")
                plt.show()
            
            graph = nx.from_numpy_array(adj_matrix)
            # print("Nodes:", graph.number_of_nodes(), "Edges:", graph.number_of_edges())
            nodes_count += graph.number_of_nodes()
            edges_count += graph.number_of_edges()
            # Append the instance to the dataset
            self.dataset.instances.append(GraphInstance(id=i, data=adj_matrix, label=label, node_features=features))
            # self.context.logger.info(f"Generated instance with id {i}, label={label}")

        nodes_avg = nodes_count / self.num_instances
        edges_avg = edges_count / self.num_instances
        # print("Nodes avg:", nodes_avg, "Edges avg:", edges_avg)
        self.context.logger.info(f"Generated dataset - instances: {self.num_instances} - nodes: {self.nodes_num} - edges per node: {self.edges_per_node} - Nodes avg: {nodes_avg:.2f} - Edges avg: {edges_avg:.2f}")
        #input()
            
    def get_num_instances(self):
        return len(self.dataset.instances)