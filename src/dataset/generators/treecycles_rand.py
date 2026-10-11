import networkx as nx
import numpy as np

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


class TreeCyclesRand(Generator):
    
    def init(self):
        self.dataset_name = "TreeCyclesRand"    
        self.num_instances = self.local_config['parameters']['num_instances']
        self.num_nodes_per_instance = self.local_config['parameters']['num_nodes_per_instance']
        self.ratio_nodes_in_cycles = self.local_config['parameters']['ratio_nodes_in_cycles']
        self.seed = self.local_config['parameters'].get('seed', 0)
        self.rng = np.random.default_rng(self.seed)

        self.generate_dataset()
        
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config

        # set defaults
        local_config['parameters']['num_instances'] = local_config['parameters'].get('num_instances', 1000)
        local_config['parameters']['num_nodes_per_instance'] = local_config['parameters'].get('num_nodes_per_instance', 300)
        local_config['parameters']['ratio_nodes_in_cycles'] = local_config['parameters'].get('ratio_nodes_in_cycles', 0.3)

    def generate_dataset(self):     
                  
        nodes_count, edges_count = 0, 0

        for i in range(self.num_instances):
            # Randomly determine if the graph is going to contain cycles or just be a tree
            has_cycles = self.rng.integers(0, 2)  # 2 excluded
            graph_seed = int(self.rng.integers(0, 2**32 - 1))
            # If the graph will contain cycles

            if(has_cycles):
                full = False # or self.rng.integers(0,2) # 2 excluded
                if full:
                    tc_graph = nx.to_numpy_array(nx.complete_graph(self.num_nodes_per_instance))
                else:
                    cycles = []
                    budget = int( self.ratio_nodes_in_cycles * self.num_nodes_per_instance )
                    budget = self.rng.integers(3, budget+1) if budget > 3 else budget
                    left = self.num_nodes_per_instance - budget
                                
                    while budget > 2: 
                        num_nodes = self.rng.integers(3,budget+1)
                        cycles.append(nx.cycle_graph(num_nodes))
                        budget -= num_nodes
                    
                    left += budget
                    tc_graph = self._join_graphs_as_adj(nx.random_tree(n=left, seed=graph_seed), cycles)

                # print(tc_graph); input()
                self.dataset.instances.append(GraphInstance(id=i, data=tc_graph, node_features=np.ones(self.num_nodes_per_instance), label=1))
                graph = nx.from_numpy_array(tc_graph)
            else:
                empty = False # or self.rng.integers(0,2) # 2 excluded
                if empty:
                    t_graph = nx.empty_graph(self.num_nodes_per_instance)
                else:
                    # Generating a random tree containing all the nodes of the instance
                    t_graph = nx.random_tree(n=self.num_nodes_per_instance, seed=graph_seed)
                
                # print(nx.to_numpy_array(t_graph)); input()
                self.dataset.instances.append(GraphInstance(id=i, data=nx.to_numpy_array(t_graph), node_features=np.ones(self.num_nodes_per_instance), label=0))
                graph = t_graph

            # print("Nodes:", graph.number_of_nodes(), "Edges:", graph.number_of_edges())
            nodes_count += graph.number_of_nodes()
            edges_count += graph.number_of_edges()
            # self.context.logger.info("Generated instance with id:"+str(i))
        
        nodes_avg = nodes_count / self.num_instances
        edges_avg = edges_count / self.num_instances
        # print("Nodes avg:", nodes_avg, "Edges avg:", edges_avg)      
        self.context.logger.info(f"Generated dataset - instances: {self.num_instances} - nodes per instance: {self.num_nodes_per_instance} - ratio nodes in cycles: {self.ratio_nodes_in_cycles} - Nodes avg: {nodes_avg:.2f} - Edges avg: {edges_avg:.2f}")    
    
    
    
    def _join_graphs_as_adj(self, base, others):
        Ab = nx.to_numpy_array(base)
        A = Ab
        for other in others:
            Ao = nx.to_numpy_array(other)
            t_node = self.rng.integers(0,len(Ab))
            s_node = len(A) + self.rng.integers(0,len(Ao))
            A = np.block([[A,np.zeros((len(A),len(Ao)))],[np.zeros((len(Ao),len(A))), Ao]])            
            A[t_node,s_node]=1
            A[s_node,t_node]=1

        return A
    

    def count_graph_elements(graph):
        # If graph is an adjacency matrix, convert it to a networkx graph
        if isinstance(graph, np.ndarray):
            graph = nx.from_numpy_array(graph)
        return graph.number_of_nodes(), graph.number_of_edges()
