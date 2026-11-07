from src.evaluation.evaluation_metric_chemical_validity import ChemicalValidityMetric
from src.dataset.instances.graph import GraphInstance
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer

class OriginalChemicalValidityMetric(ChemicalValidityMetric):
    """1.0 if the original instance is a chemically valid molecule, else 0.0."""

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Original_Chemical_Validity'
        self.num_structural_feat = None

    def evaluate(self, instance_1: GraphInstance, instance_2: GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        """
        Args:
            instance_1 (GraphInstance) :
                Original instance.
            instance_2 (GraphInstance):
                Counterfactual instance.
        """
        return float(self._is_valid(instance_1.data, instance_1, instance_1, dataset))