from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.dataset.instances.graph import GraphInstance
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.core.embedder_base import Embedder

from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityFullMetric(EvaluationMetric):
    """
    Computes the cosine similarity between the Instance and its counterfactual.
    This is done for all the embedders fitted on the dataset.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Cosine_Similarity'          

    def evaluate(self, instance_1:GraphInstance, instance_2:GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None, embedders:dict[Embedder]=None):
        """
        Evaluates the cosine similarity between the embeddings of instance_1 and instance_2.
        
        Parameters:
            instance_1 (GraphInstance): The original graph.
            instance_2 (GraphInstance): The counterfactual graph.
            oracle, explainer, dataset: Additional parameters (unused in this snippet).
        
        Returns:
            A dictionary mapping model names to the cosine similarity score between the two embeddings.
        """
        similarity_results = {}

        for embedder in embedders.values():
            embeddings = embedder.infer([instance_1, instance_2])
            
            # Reshape the embeddings if necessary to be 2D arrays
            emb1 = embeddings[0].reshape(1, -1)
            emb2 = embeddings[1].reshape(1, -1)
            
            # Compute cosine similarity between the two embeddings
            cos_sim = cosine_similarity(emb1, emb2)[0][0]
            similarity_results[embedder.__class__.__name__] = float(cos_sim)

        return similarity_results