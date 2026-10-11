from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.instances.graph import GraphInstance
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
import numpy as np
import torch

class GraphEditDistanceFullMetric(EvaluationMetric):
    """Provides a graph edit distance function for graphs where nodes are already matched, 
    thus eliminating the need of performing an NP-Complete graph matching.
    """

    def __init__(self, node_insertion_cost=1.0, node_deletion_cost=1.0, edge_insertion_cost=1.0,
                 edge_deletion_cost=1.0, undirected=True, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Graph_Edit_Distance'
        self._node_insertion_cost = node_insertion_cost
        self._node_deletion_cost = node_deletion_cost
        self._edge_insertion_cost = edge_insertion_cost
        self._edge_deletion_cost = edge_deletion_cost
        self.undirected = undirected
        

    def evaluate(self, instance_1:(GraphInstance), instance_2:(GraphInstance), oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        """ It counts the edge weight instead of adjacency matrix, but in practice it is the same as standard GED

        Args:
            instance_1 (GraphInstance) :
                Original instance.
            instance_2 (GraphInstance):
                Counterfactual instance.
        """
        
        self.data_1 = torch.tensor(instance_1.data)
        adj_1 = torch.ones_like(self.data_1)
        edge_indices_1 = torch.where(self.data_1 != 0) # (integer tensor)
        edge_weights_1 = torch.tensor(self.data_1.clone().detach()[edge_indices_1])
        edge_weights_full_1 = torch.zeros_like(self.data_1)
        edge_weights_full_1[edge_indices_1] = edge_weights_1 # weights having also 0s for missing edges
        edge_weights_full_1 = edge_weights_full_1.flatten().clone().detach().cpu().numpy()
        # edge_index_full_1 = adj_1.nonzero(as_tuple=False).T

        self.data_2 = torch.tensor(instance_2.data)
        adj_2 = torch.ones_like(self.data_2)
        edge_indices_2 = torch.where(self.data_2 != 0) # (integer tensor)
        edge_weights_2 = torch.tensor(self.data_2.clone().detach()[edge_indices_2])
        edge_weights_full_2 = torch.zeros_like(self.data_2)
        edge_weights_full_2[edge_indices_2] = edge_weights_2 # weights having also 0s for missing edges
        edge_weights_full_2 = edge_weights_full_2.flatten().clone().detach().cpu().numpy()
        # edge_index_full_2 = adj_2.nonzero(as_tuple=False).T
    
       # Get the difference in the number of nodes
        nodes_diff_count = abs(adj_1.shape[0] - adj_2.shape[0])

        # Get the shape of the matrices
        shape_A_g1 = adj_1.shape
        shape_A_g2 = adj_2.shape

        edges_diff_count = np.sum((edge_weights_full_1 > 0) != (edge_weights_full_2 > 0))
        # print(edges_diff_count)

        if self.undirected:
            edges_diff_count /= 2
        # input()

        return nodes_diff_count + edges_diff_count
    
    
    def aggregate(self, measure_list, instances_correctness_list=None):
        # If no correctness list is provided aggregate all the measures
        if instances_correctness_list is None:
            return super().aggregate(measure_list, instances_correctness_list)
        else: # If correctness list is provided then aggregate only the measures of the correct instances
            filtered_measure_list = [item for item, flag in zip(measure_list, instances_correctness_list) if flag == 1]

            # Avoid aggregating an empty list
            if len(filtered_measure_list) > 0:
                return np.mean(filtered_measure_list), np.std(filtered_measure_list)
            else:
                return 0.0, 0.0

    