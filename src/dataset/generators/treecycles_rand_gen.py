import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


class TreeCyclesRandGen(Generator):
    
    def init(self):
        self.dataset_name = "TreeCyclesRandGen"    
        self.num_instances = self.local_config['parameters']['num_instances']
        self.num_nodes_per_instance = self.local_config['parameters']['num_nodes_per_instance']
        self.ratio_nodes_in_cycles = self.local_config['parameters']['ratio_nodes_in_cycles']
        self.generate_dataset()
        
    def check_configuration(self):
        super().check_configuration
        local_config=self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 50000)
        local_config['parameters']['num_nodes_per_instance'] = local_config['parameters'].get('num_nodes_per_instance', 16)
        local_config['parameters']['ratio_nodes_in_cycles'] = local_config['parameters'].get('ratio_nodes_in_cycles', 0.5)

    def generate_dataset(self):     
                  
        nodes_count, edges_count = 0, 0

        for i in range(self.num_instances):
            
            n_nodes = np.random.randint(8, self.num_nodes_per_instance) # Create a graph with a random number of nodes
            # print(f"n_nodes: {n_nodes}")
            ratio = max(3 / n_nodes, self.ratio_nodes_in_cycles)
            # print(f"ratio: {ratio}")

            # Randomly determine if the graph is going to contain cycles or just be a tree
            has_cycles = np.random.randint(0,2) # 2 excluded
            # If the graph will contain cycles
            if(has_cycles):
                full = False #or np.random.randint(0,2) # 2 excluded
                if full:
                    tc_graph = nx.to_numpy_array(nx.complete_graph(n_nodes))
                else:
                    cycles = []

                    budget = int( ratio * n_nodes )
                    budget = np.random.randint(3, budget+1) if budget > 3 else budget
                    # print(f"budget: {budget}")
                    left = n_nodes - budget
                    n_cycles = 0

                    while budget > 2: 
                        num_nodes = np.random.randint(3,budget+1)
                        cycles.append(nx.cycle_graph(num_nodes))
                        budget -= num_nodes
                        n_cycles += 1
                        # print(f"budget after cycle appended: {budget}")

                    
                    left += budget
                    tc_graph = nx.from_numpy_array(self._join_graphs_as_adj(nx.random_tree(n=left), cycles))
                    tc_graph.add_edges_from([(n, n) for n in tc_graph.nodes() if np.random.rand() < 0.3]) # add some self-loops

                # print(tc_graph); input()
                label = 1
                # self.dataset.instances.append(GraphInstance(id=i, data=tc_graph, node_features=np.zeros(n_nodes), label=label))
                self.dataset.instances.append(GraphInstance(id=i, data=nx.to_numpy_array(tc_graph), label=label))
                graph = tc_graph
            else:
                empty = False #or np.random.randint(0,2) # 2 excluded
                if empty:
                    t_graph = nx.empty_graph(n_nodes)
                else:
                    # Generating a random tree containing all the nodes of the instance
                    t_graph = nx.random_tree(n=n_nodes)
                    t_graph.add_edges_from([(n, n) for n in t_graph.nodes() if np.random.rand() < 0.3]) # add some self-loops
                
                # print(nx.to_numpy_array(t_graph)); input()
                label = 0
                # self.dataset.instances.append(GraphInstance(id=i, data=nx.to_numpy_array(t_graph), node_features=np.zeros(n_nodes), label=label))
                self.dataset.instances.append(GraphInstance(id=i, data=nx.to_numpy_array(t_graph), label=label))
                graph = t_graph

            if False:
                fig, axes = plt.subplots()
                nx.draw(graph, with_labels=True, edge_color='gray')
                fig.suptitle(f"label: {f"Cycle ({n_cycles})" if label else "Tree"}")
                # plt.text(0.5, 1.05, f"label: {label}", ha="center", transform=plt.gca().transAxes)
                plt.show()

            # print("Nodes:", graph.number_of_nodes(), "Edges:", graph.number_of_edges())
            nodes_count += graph.number_of_nodes()
            edges_count += graph.number_of_edges()
            # self.context.logger.info("Generated instance with id:"+str(i))
        
        nodes_avg = nodes_count / self.num_instances
        edges_avg = edges_count / self.num_instances
        # print("Nodes avg:", nodes_avg, "Edges avg:", edges_avg)   
        self.context.logger.info(f"Generated dataset - instances: {self.num_instances} - nodes per instance: {self.num_nodes_per_instance} - ratio nodes in cycles: {self.ratio_nodes_in_cycles} - Nodes avg: {nodes_avg:.2f} - Edges avg: {edges_avg:.2f}")    

        # input()
    
    
    
    def _join_graphs_as_adj(self, base, others):
        Ab = nx.to_numpy_array(base)
        A = Ab
        for other in others:
            Ao = nx.to_numpy_array(other)
            t_node = np.random.randint(0,len(Ab))
            s_node = len(A) + np.random.randint(0,len(Ao))
            A = np.block([[A,np.zeros((len(A),len(Ao)))],[np.zeros((len(Ao),len(A))), Ao]])            
            A[t_node,s_node]=1
            A[s_node,t_node]=1

        return A
    

    def count_graph_elements(graph):
        # If graph is an adjacency matrix, convert it to a networkx graph
        if isinstance(graph, np.ndarray):
            graph = nx.from_numpy_array(graph)
        return graph.number_of_nodes(), graph.number_of_edges()
