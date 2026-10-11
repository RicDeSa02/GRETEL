from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.instances.graph import GraphInstance
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class IterationsCountMetric(EvaluationMetric):
    """Provides the number of iterations the explainer has to perform in order to generate a counterfactual example
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Iterations_Count'

    def evaluate(self, instance_1:GraphInstance, instance_2:GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        """
        Args:
            instance_1 (GraphInstance) :
                Original instance.
            instance_2 (GraphInstance):
                Counterfactual instance.
        """
        result = instance_2.iteration if hasattr(instance_2, "iteration") else None
        return result