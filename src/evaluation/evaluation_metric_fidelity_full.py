from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.instances.graph import GraphInstance
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
import torch


class FidelityFullMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'FidelityFull'

    def evaluate_full(self, instance_1:GraphInstance, instance_2:GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        """
        Args:
            instance_1 (GraphInstance) :
                Original instance.
            instance_2 (GraphInstance):
                Counterfactual instance.
        """

        self.device = oracle.device

        self.data_1 = torch.tensor(instance_1.data, device=self.device)
        adj_1 = torch.ones_like(self.data_1, device=self.device)
        edge_indices_1 = torch.where(self.data_1 != 0) # (int tensor)
        edge_weights_1 = self.data_1.detach().clone()[edge_indices_1[0], edge_indices_1[1]]
        edge_weights_full_1 = torch.zeros_like(self.data_1, device=self.device)
        edge_weights_full_1[edge_indices_1] = edge_weights_1 # weights having also 0s for missing edges
        edge_weights_full_1 = edge_weights_full_1.flatten()
        edge_index_full_1 = adj_1.nonzero(as_tuple=False).T
        self.batch = torch.zeros(instance_1.node_features.shape[0], dtype=torch.long, device=self.device)
        label_instance_1 = torch.argmax(oracle.model(torch.tensor(instance_1.node_features, dtype=torch.double, device=self.device), edge_index_full_1, edge_weights_full_1, self.batch).clone().detach(), dim=-1).squeeze(-1)

        self.data_2 = torch.tensor(instance_2.data, device=self.device)
        adj_2 = torch.ones_like(self.data_2, device=self.device)
        edge_indices_2 = torch.where(self.data_2 != 0) # (int tensor)
        edge_weights_2 = self.data_2.detach().clone()[edge_indices_2[0], edge_indices_2[1]]
        edge_weights_full_2 = torch.zeros_like(self.data_2, device=self.device)
        edge_weights_full_2[edge_indices_2] = edge_weights_2 # weights having also 0s for missing edges
        edge_weights_full_2 = edge_weights_full_2.flatten()
        edge_index_full_2 = adj_2.nonzero(as_tuple=False).T
        label_instance_2 = torch.argmax(oracle.model(torch.tensor(instance_2.node_features, dtype=torch.double, device=self.device), edge_index_full_2, edge_weights_full_2, self.batch).clone().detach(), dim=-1).squeeze(-1) 

        prediction_fidelity = 1 if (label_instance_1 == instance_1.label) else 0
        
        counterfactual_fidelity = 1 if (label_instance_2 == instance_1.label) else 0

        result = prediction_fidelity - counterfactual_fidelity
        
        return result
    
    def evaluate(self, instance_1:GraphInstance, instance_2:GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        """
        Args:
            instance_1 (GraphInstance) :
                Original instance.
            instance_2 (GraphInstance):
                Counterfactual instance.
        """
        self.device = oracle.device

        label_instance_1 = oracle.predict(instance_1)
        label_instance_2 = oracle.predict(instance_2)
        oracle._call_counter -= 2
        
        prediction_fidelity = 1 if (label_instance_1 == instance_1.label) else 0
        
        counterfactual_fidelity = 1 if (label_instance_2 == instance_1.label) else 0

        result = prediction_fidelity - counterfactual_fidelity
        
        return result