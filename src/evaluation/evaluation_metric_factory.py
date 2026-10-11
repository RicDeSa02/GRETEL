from src.evaluation.evaluation_metric_fidelity_node import FidelityNodeMetric
from src.evaluation.evaluation_metric_oracle_accuracy_node_classification import OracleAccuracyNodeMetric
from src.evaluation.evaluation_metric_correctness import CorrectnessMetric
from src.evaluation.evaluation_metric_correctness_nodes import CorrectnessNodesMetric
from src.evaluation.evaluation_metric_correctness_full import CorrectnessFullMetric
from src.evaluation.evaluation_metric_node_correctness import NodeCorrectnessMetric
from src.evaluation.evaluation_metric_fidelity import FidelityMetric
from src.evaluation.evaluation_metric_fidelity_nodes import FidelityNodesMetric
from src.evaluation.evaluation_metric_fidelity_full import FidelityFullMetric
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.evaluation.evaluation_metric_ged_nodes import GraphEditDistanceNodesMetric
from src.evaluation.evaluation_metric_ged_full import GraphEditDistanceFullMetric
from src.evaluation.evaluation_metric_runtime import RuntimeMetric
from src.evaluation.evaluation_metric_runtime_nodes import RuntimeMetricNodes
from src.evaluation.evaluation_metric_sparsity import SparsityMetric
from src.evaluation.evaluation_metric_sparsity_nodes import SparsityNodesMetric
from src.evaluation.evaluation_metric_sparsity_full import SparsityFullMetric
from src.evaluation.evaluation_metric_oracle_calls import OracleCallsMetric
from src.evaluation.evaluation_metric_oracle_calls_nodes import OracleCallsNodesMetric
from src.evaluation.evaluation_metric_oracle_accuracy import OracleAccuracyMetric
from src.evaluation.evaluation_metric_oracle_accuracy_nodes import OracleAccuracyNodesMetric
from src.evaluation.evaluation_metric_oracle_accuracy_full import OracleAccuracyFullMetric
from src.evaluation.evaluation_metric_cosine_similarity import CosineSimilarityMetric
from src.evaluation.evaluation_metric_cosine_similarity_nodes import CosineSimilarityNodesMetric
from src.evaluation.evaluation_metric_cosine_similarity_full import CosineSimilarityFullMetric
from src.evaluation.evaluation_metric_smiles_levenshtein import SmilesLevenshteinMetric
from src.evaluation.evaluation_metric_dumper import InstancesDumper
from src.evaluation.evaluation_metric_embed_dataset import EmbedDatasetMetric
from src.evaluation.evaluation_metric_suppl import SupplMetric
from src.evaluation.evaluation_metric_labels_correctness_nodes import LabelsCorrectnessNodesMetric
from src.evaluation.evaluation_metric_iterations_count import IterationsCountMetric
from src.evaluation.evaluation_metric_chemical_validity import ChemicalValidityMetric
from src.evaluation.evaluation_metric_original_chemical_validity import OriginalChemicalValidityMetric



class EvaluationMetricFactory:

    def __init__(self,config_dict) -> None:
        self._config_dict = config_dict
        self._evaluation_metric_id_counter = 0

    def get_evaluation_metric_by_name(self, metric_dict) -> EvaluationMetric:
        metric_name = metric_dict['name']
        metric_parameters = metric_dict['parameters']

        if(metric_name == 'graph_edit_distance'):
            return self.get_graph_edit_distance_metric(config_dict=metric_dict)
        
        elif(metric_name == 'graph_edit_distance_nodes'):
            return self.get_graph_edit_distance_nodes_metric(config_dict=metric_dict)
        
        elif(metric_name == 'graph_edit_distance_full'):
            return self.get_graph_edit_distance_full_metric(config_dict=metric_dict)

        elif metric_name == 'oracle_calls':
            return self.get_oracle_calls_metric(config_dict=metric_dict)

        elif metric_name == 'oracle_calls_nodes':
            return self.get_oracle_calls_nodes_metric(config_dict=metric_dict)

        elif metric_name == 'sparsity':
            return self.get_sparsity_metric(config_dict=metric_dict)

        elif metric_name == 'sparsity_nodes':
            return self.get_sparsity_nodes_metric(config_dict=metric_dict)
        
        elif metric_name == 'sparsity_full':
            return self.get_sparsity_full_metric(config_dict=metric_dict)

        elif metric_name == 'correctness':
            return self.get_correctness_metric(config_dict=metric_dict)

        elif metric_name == 'correctness_nodes':
            return self.get_correctness_nodes_metric(config_dict=metric_dict)
        
        elif metric_name == 'correctness_full':
            return self.get_correctness_full_metric(config_dict=metric_dict)
        
        elif metric_name == 'node_correctness':
            return self.get_node_correctness_metric(config_dict=metric_dict)

        elif metric_name == 'fidelity':
            return self.get_fidelity_metric(config_dict=metric_dict)
        
        elif metric_name == 'fidelity_full':
            return self.get_fidelity_full_metric(config_dict=metric_dict)
        
        elif metric_name == 'fidelity_nodes':
            return self.get_fidelity_nodes_metric(config_dict=metric_dict)
        
        elif metric_name == 'fidelity_node':
            return self.get_fidelity_node_metric(config_dict=metric_dict)

        elif metric_name == 'oracle_accuracy':
            return self.get_oracle_accuracy_metric(config_dict=metric_dict)
        
        elif metric_name == 'oracle_accuracy_nodes':
            return self.get_oracle_accuracy_nodes_metric(config_dict=metric_dict)
        
        elif metric_name == 'oracle_accuracy_full':
            return self.get_oracle_accuracy_full_metric(config_dict=metric_dict)

        elif metric_name == 'smiles_levenshtein':
            return self.get_smiles_levenshtein_metric(config_dict=metric_dict)
        
        elif metric_name == 'oracle_accuracy_node':
            return self.get_oracle_accuracy_node_metric(config_dict=metric_dict)
        
        elif metric_name == 'cosine_similarity':
            return self.get_cosine_similarity_metric(config_dict=metric_dict)
        
        elif metric_name == 'cosine_similarity_nodes':
            return self.get_cosine_similarity_nodes_metric(config_dict=metric_dict)
        
        elif metric_name == 'cosine_similarity_full':
            return self.get_cosine_similarity_full_metric(config_dict=metric_dict)
        
        elif metric_name == 'dumper':
            return self.get_dumper_metric(config_dict=metric_dict)
        
        elif metric_name == 'runtime':
            return self.get_runtime_metric(config_dict=metric_dict)
        
        elif metric_name == 'runtime_nodes':
            return self.get_runtime_nodes_metric(config_dict=metric_dict)
        
        elif metric_name == 'embed_dataset':
            return self.get_embed_dataset_metric(config_dict=metric_dict)
        
        elif metric_name == 'suppl':
            return self.get_suppl_metric(config_dict=metric_dict)

        elif metric_name == 'labels_correctness_nodes':
            return self.get_labels_correctness_nodes_metric(config_dict=metric_dict)

        elif metric_name == 'iterations':
            return self.get_iterations_metric(config_dict=metric_dict)

        elif metric_name == 'chemical_validity':
            return self.get_chemical_validity_metric(config_dict=metric_dict)

        elif metric_name == 'original_chemical_validity':
            return self.get_original_chemical_validity_metric(config_dict=metric_dict)

        else:
            raise ValueError('''The provided evaluation metric name does not match any evaluation
             metric provided by the factory''')
        
    def get_runtime_metric(self, config_dict=None) -> EvaluationMetric:
        result = RuntimeMetric(config_dict)
        return result
    
    def get_runtime_nodes_metric(self, config_dict=None) -> EvaluationMetric:
        result = RuntimeMetricNodes(config_dict)
        return result

    def get_dumper_metric(self, config_dict=None) -> EvaluationMetric:
        result = InstancesDumper(config_dict)
        return result

    def get_correctness_metric(self, config_dict=None) -> EvaluationMetric:
        result = CorrectnessMetric(config_dict)
        return result
    
    def get_correctness_nodes_metric(self, config_dict=None) -> EvaluationMetric:
        result = CorrectnessNodesMetric(config_dict)
        return result
    
    def get_correctness_full_metric(self, config_dict=None) -> EvaluationMetric:
        result = CorrectnessFullMetric(config_dict)
        return result

    def get_node_correctness_metric(self, config_dict=None) -> EvaluationMetric:
        result = NodeCorrectnessMetric(config_dict)
        return result

    def get_oracle_calls_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleCallsMetric(config_dict)
        return result
    
    def get_oracle_calls_nodes_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleCallsNodesMetric(config_dict)
        return result

    def get_graph_edit_distance_metric(self, node_insertion_cost=1.0, node_deletion_cost=1.0, 
                                        edge_insertion_cost=1.0, edge_deletion_cost=1.0, undirected=True, config_dict=None) -> EvaluationMetric:
        
        result = GraphEditDistanceMetric(node_insertion_cost, node_deletion_cost, edge_insertion_cost, 
                                            edge_deletion_cost, undirected, config_dict)

        return result
    
    def get_graph_edit_distance_nodes_metric(self, node_insertion_cost=1.0, node_deletion_cost=1.0, 
                                        edge_insertion_cost=1.0, edge_deletion_cost=1.0, undirected=True, config_dict=None) -> EvaluationMetric:
        
        result = GraphEditDistanceNodesMetric(node_insertion_cost, node_deletion_cost, edge_insertion_cost, 
                                            edge_deletion_cost, undirected, config_dict)

        return result
    
    def get_graph_edit_distance_full_metric(self, node_insertion_cost=1.0, node_deletion_cost=1.0, 
                                        edge_insertion_cost=1.0, edge_deletion_cost=1.0, undirected=True, config_dict=None) -> EvaluationMetric:
        
        result = GraphEditDistanceFullMetric(node_insertion_cost, node_deletion_cost, edge_insertion_cost, 
                                            edge_deletion_cost, undirected, config_dict)

        return result


    def get_sparsity_metric(self, config_dict=None) -> EvaluationMetric:
        result = SparsityMetric(config_dict)
        return result
    
    def get_sparsity_nodes_metric(self, config_dict=None) -> EvaluationMetric:
        result = SparsityNodesMetric(config_dict)
        return result
    
    def get_sparsity_full_metric(self, config_dict=None) -> EvaluationMetric:
        result = SparsityFullMetric(config_dict)
        return result


    def get_fidelity_metric(self, config_dict=None) -> EvaluationMetric:
        result = FidelityMetric(config_dict)
        return result

    def get_fidelity_nodes_metric(self, config_dict=None) -> EvaluationMetric:
        result = FidelityNodesMetric(config_dict)
        return result
    
    def get_fidelity_full_metric(self, config_dict=None) -> EvaluationMetric:
        result = FidelityFullMetric(config_dict)
        return result
    
    
    def get_fidelity_node_metric(self, config_dict=None) -> EvaluationMetric:
        result = FidelityNodeMetric(config_dict)
        return result

    def get_oracle_accuracy_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleAccuracyMetric(config_dict)
        return result
    
    def get_oracle_accuracy_nodes_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleAccuracyNodesMetric(config_dict)
        return result
    
    def get_oracle_accuracy_full_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleAccuracyFullMetric(config_dict)
        return result
    

    def get_oracle_accuracy_node_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleAccuracyNodeMetric(config_dict)
        return result
    
    def get_smiles_levenshtein_metric(self, config_dict=None) -> EvaluationMetric:
        result = SmilesLevenshteinMetric(config_dict)
        return result
    
    def get_cosine_similarity_metric(self, config_dict=None) -> EvaluationMetric:
        result = CosineSimilarityMetric(config_dict)
        return result
    
    def get_cosine_similarity_nodes_metric(self, config_dict=None) -> EvaluationMetric:
        result = CosineSimilarityNodesMetric(config_dict)
        return result
    
    def get_cosine_similarity_full_metric(self, config_dict=None) -> EvaluationMetric:
        result = CosineSimilarityFullMetric(config_dict)
        return result
    
    def get_embed_dataset_metric(self, config_dict=None) -> EvaluationMetric:
        result = EmbedDatasetMetric(config_dict)
        return result
    
    def get_suppl_metric(self, config_dict=None) -> EvaluationMetric:
        result = SupplMetric(config_dict)
        return result
    
    def get_labels_correctness_nodes_metric(self, config_dict=None) -> EvaluationMetric:
        result = LabelsCorrectnessNodesMetric(config_dict)
        return result

    def get_iterations_metric(self, config_dict=None) -> EvaluationMetric:
            result = IterationsCountMetric(config_dict)
            return result

    def get_chemical_validity_metric(self, config_dict=None) -> EvaluationMetric:
            result = ChemicalValidityMetric(config_dict)
            return result

    def get_original_chemical_validity_metric(self, config_dict=None) -> EvaluationMetric:
        result = OriginalChemicalValidityMetric(config_dict)
        return result