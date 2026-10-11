from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.instances.graph import GraphInstance
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
# from src.evaluation.evaluation_metric_ged_full import GraphEditDistanceFullMetric
import torch




class CorrectnessFullMetric(EvaluationMetric):
    """Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'CorrectnessFull'
        self._ged = GraphEditDistanceMetric() # GraphEditDistanceFullMetric()

    def evaluate_full(self, instance_1:GraphInstance, instance_2:GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        """
        Args:
            instance_1 (GraphInstance) :
                Original instance.
            instance_2 (GraphInstance):
                Counterfactual instance.
        """

        self.device = oracle.device
        # print(f"metric device: {self.device}")

        # print(instance_1.data.shape)
        # print(instance_1.edge_weights.shape)
        # print(instance_2.edge_weights.shape)
        # print(instance_1.edge_weights)
        # print(instance_2.edge_weights)

        # label_instance_1 = oracle.predict(instance_1)
        # print(f"label instance_1: {label_instance_1}")
        # label_instance_2 = oracle.predict(instance_2)
        # print(f"label instance_2: {label_instance_2}")
        # oracle._call_counter -= 2

        # ged = self._ged.evaluate(instance_1, instance_2, oracle)

        # result = 1 if (label_instance_1 != label_instance_2) and (ged != 0) else 0
        # print(f"validity: {result}")

        self.data_1 = torch.tensor(instance_1.data, device=self.device)
        adj_1 = torch.ones_like(self.data_1, device=self.device)
        edge_indices_1 = torch.where(self.data_1 != 0) # (int tensor)
        edge_weights_1 = self.data_1.detach().clone()[edge_indices_1[0], edge_indices_1[1]]
        edge_weights_full_1 = torch.zeros_like(self.data_1, device=self.device)
        edge_weights_full_1[edge_indices_1] = edge_weights_1 # weights having also 0s for missing edges
        edge_weights_full_1 = edge_weights_full_1.flatten()
        edge_index_full_1 = adj_1.nonzero(as_tuple=False).T
        # print(f"edge_weights_full.shape: {edge_weights_full_1.shape}")
        # print(f"edge_index_full.shape: {edge_index_full_1.shape}")

        self.batch = torch.zeros(instance_1.node_features.shape[0], dtype=torch.long, device=self.device)
        # label_instance_1 = torch.argmax(oracle.model(torch.tensor(instance_1.node_features, dtype=torch.double, device=self.device), edge_index_full_1, edge_weights_full_1, self.batch).clone().detach(), dim=-1).squeeze(-1) 
        label_instance_1 = oracle.predict(instance_1)
        # print(f"label instance_1: {label_instance_1}")

        self.data_2 = torch.tensor(instance_2.data, device=self.device)
        adj_2 = torch.ones_like(self.data_2, device=self.device)
        edge_indices_2 = torch.where(self.data_2 != 0) # (int tensor)
        edge_weights_2 = self.data_2.detach().clone()[edge_indices_2[0], edge_indices_2[1]]
        edge_weights_full_2 = torch.zeros_like(self.data_2, device=self.device)
        edge_weights_full_2[edge_indices_2] = edge_weights_2 # weights having also 0s for missing edges
        edge_weights_full_2 = edge_weights_full_2.flatten()
        edge_index_full_2 = adj_2.nonzero(as_tuple=False).T
        # print(f"edge_weights_full.shape: {edge_weights_full_2.shape}")
        # print(f"edge_index_full.shape: {edge_index_full_2.shape}")
        # print(f"edge_weights_full: {edge_weights_full_2}")
        # print(instance_2.edge_weights)
        # print(instance_2.node_features)

        
        # print(instance_2.node_features.shape, edge_index_full_2.shape, instance_2.edge_weights.shape)
        # label_instance_2 = torch.argmax(oracle.model(torch.tensor(instance_2.node_features, dtype=torch.double, device=self.device), edge_index_full_2, edge_weights_full_2, self.batch).clone().detach(), dim=-1).squeeze(-1) 
        label_instance_2 = oracle.predict(instance_2)
        # print(f"label instance_2: {label_instance_2}")

        ged = self._ged.evaluate(instance_1, instance_2, oracle)

        result = 1 if (label_instance_1 != label_instance_2) and (ged != 0) else 0
        # print(f"validity: {result}")
        
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

        ged = self._ged.evaluate(instance_1, instance_2, oracle)

        result = 1 if (label_instance_1 != label_instance_2) and (ged != 0) else 0
        # print(f"validity: {result}")
        
        return result