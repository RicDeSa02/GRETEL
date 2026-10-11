import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.utils.cfg_utils import retake_oracle
from src.dataset.manipulators.base import BaseManipulator
from src.dataset.manipulators.centralities import NodeCentrality

import src.oracle.nn.torch_diffusion

class XPlore(Explainer):
    '''
    Algorithmic implementation of XPlore.
    Extended version of the Explainer based on
    Lucic et al. CF-GNNExplainer Counterfactual Explanations for Graph Neural Networks
    https://arxiv.org/abs/2102.03322
    Algorithm: given a graph G = (AG, X) where f (G) = y, generate the minimal perturbation,  ̄G = (ĀG, X), such that f (̄G) ≠ y. [5.4]
    '''
    
    def init(self):
        """
        α: Learning Rate, K: Number of iterations, β: Loss controll (Ldist wrt Lpred) (Eq.1 [4]), γ_edges: Missing edges addition fraction (per edge)
        α: 0.1, K: 500, β: 0.5 // Paper best parameters (Hyperparameter Search [6.4])
        """
        # input("Training Complete")
        self.oracle = retake_oracle(self.local_config)
        
        local_params = self.local_config['parameters']
        self.α = local_params['alpha'] # α: Learning Rate
        self.K = local_params['K'] # K: Number of Iterations (to update pertubation matrices (P_hat, etc.))
        self.β = local_params['beta'] # β: Trade-off between Lpred and Ldist Eq.1 [4]
        self.extended = local_params['extended'] # Use extended version of the algorithm (allowing to add edges and not only to drop them)
        self.γ_edges = local_params['gamma_edge'] # γ: Add missing edges to the edge perturbation matrix (γ ∈ [0, 1])
        self.update_node_feat = local_params['update_node_feat'] # Allow to update node features (gate or change them freely)
        self.change_node_feat = local_params['change_node_feat'] # Allow node features to change freely instead of just keep or discard a given feature (gating)
        self.change_all_feat = local_params['change_all_feat'] # Allow all features to change freely (node, edge, graph features)
        self.γ_node_feat = local_params['gamma_node_feat'] # γ: Add missing node features to the node features perturbation matrix (γ ∈ [0, 1]) (NOT USED FOR CURRENT INITIALIZATION Ⅱ)
        self.debugging = local_params['debugging'] # Print debugging code one iteration at a time
        self.visualize = local_params['visualize'] # Visualize inital graph and CF found (if no valid CF is found then it draws the CF at last iteration)
        self.multi_label_classification = local_params['multi_label_classification'] # Whether target classification is multi-class
        self.dataset_classes = local_params['dataset_classes'] # Dataset classes/labels amount
        self.node_classification = local_params['node_classification'] # Whether to apply node classification
        self.decay_α = local_params['decay_alpha'] # Wheter to decay learning rate (α) during explainer iterations
        self.directed = local_params['directed'] # Wheter the graph is directed or undirected

        if not self.multi_label_classification:
            # self.loss_fn = torch.nn.BCELoss() # useless as model outputs more than one logits
            # self.loss_fn = torch.nn.NLLLoss() # redundant, just use CE
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else: # Multi-Label Classification
            self.loss_fn = torch.nn.BCEWithLogitsLoss() # use for Multi-Label Classification   

        # print(f"self.oracle.__class__: {self.oracle.__class__}")
        self.diffusion_flag = True if isinstance(self.oracle, src.oracle.nn.torch_diffusion.OracleTorch) else False
        
        assert ((isinstance(self.K, float) or isinstance(self.K, int)) and self.K >= 1)
        assert ((isinstance(self.α, float) or isinstance(self.α, int)) and self.α >= 0 or True)
        assert ((isinstance(self.β, float) or isinstance(self.β, int)) and self.β >= 0 or True)
        assert (isinstance(self.extended, bool))
        assert ((isinstance(self.γ_edges, float) or isinstance(self.γ_edges, int)) and 0 <= self.γ_edges <= 1)
        assert (isinstance(self.update_node_feat, bool))
        assert (isinstance(self.change_node_feat, bool))
        assert ((isinstance(self.γ_node_feat, float) or isinstance(self.γ_node_feat, int)) and 0 <= self.γ_node_feat <= 1)
        assert (isinstance(self.debugging, bool))
        assert (isinstance(self.visualize, bool))
        assert (isinstance(self.multi_label_classification, bool))
        assert (isinstance(self.node_classification, bool))
        assert (isinstance(self.dataset_classes, int))

        if not self.extended:
            self.update_node_feat = False
            self.change_node_feat = False
            self.change_all_feat = False
        elif not self.update_node_feat:
            self.change_node_feat = False
            self.change_all_feat = False
        elif not self.change_node_feat:
            self.change_all_feat = False

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        
        if 'alpha' not in local_config['parameters']:
            local_config['parameters']['alpha'] = 0.1
        
        if 'K' not in local_config['parameters']:
            local_config['parameters']['K'] = 500
            
        if 'beta' not in local_config['parameters']:
            local_config['parameters']['beta'] = 0.5

        if 'extended' not in local_config['parameters']:
            local_config['parameters']['extended'] = True
            
        if 'gamma_edge' not in local_config['parameters']:
            local_config['parameters']['gamma_edge'] = 0

        if 'update_node_feat' not in local_config['parameters']:
            local_config['parameters']['update_node_feat'] = True    
        
        if 'change_node_feat' not in local_config['parameters']:
            local_config['parameters']['change_node_feat'] = True

        if 'change_all_feat' not in local_config['parameters']:
            local_config['parameters']['change_all_feat'] = True
            
        if 'gamma_node_feat' not in local_config['parameters']:
            local_config['parameters']['gamma_node_feat'] = 0.01

        if 'debugging' not in local_config['parameters']:
            local_config['parameters']['debugging'] = False

        if 'visualize' not in local_config['parameters']:
            local_config['parameters']['visualize'] = False

        if 'dataset_classes' not in local_config['parameters']:
            try:
                local_config['parameters']['dataset_classes'] = self.dataset.num_classes
            except:
                local_config['parameters']['dataset_classes'] = 2
        
        if 'multi_label_classification' not in local_config['parameters']:     
            if local_config['parameters']['dataset_classes'] > 2:
                local_config['parameters']['multi_label_classification'] = True
            else:
                local_config['parameters']['multi_label_classification'] = False

        if 'node_classification' not in local_config['parameters']:
            local_config['parameters']['node_classification'] = False        

        if 'decay_alppha' not in local_config['parameters']:
            local_config['parameters']['decay_alpha'] = False

        if 'directed' not in local_config['parameters']:
            local_config['parameters']['directed'] = False
               
        self.fold_id = self.local_config['parameters'].get('fold_id',-1)

    def explain(self, instance):
        """
        Find a Counterfactual for ```instance```. The closest among the ones found will be returned.
        """
        # instance.node_features = np.zeros_like(instance.node_features)
        # version = "XPlore++" if self.change_node_feat and self.update_node_feat and self.extended else "XPlore+" if self.update_node_feat and self.extended else "XPlore" if self.extended else "CF-GNNExplainer"
        # try:
        #     print(f"dataset: {self.dataset.dataset_name} - {version}")
        # except:
        #     print(f"dataset: Unknown - {version}")
        # print("num_classes =", self.dataset.num_classes)
        
        # print(instance.data.shape)
        # if self.diffusion_flag:
        #     self.f_v = self._real_predict_diffusion(instance).clone().detach()
        # else:
        #     self.f_v = self.oracle.predict(instance).clone().detach() # Get GCN prediction

        # first_nf = instance.node_features.copy()

        adj = np.ones_like(instance.data)
        edge_indices = np.where(instance.data != 0)
        edge_weights = instance.data[edge_indices]
        edge_weights_full = np.zeros_like(instance.data) 
        edge_weights_full[edge_indices] = edge_weights # weights having also 0s for missing edges
        edge_weights_full = edge_weights_full.flatten()

        print(f"adj shape: {adj.shape}")
        print(f"edge_weights_full shape: {edge_weights_full.shape}")

        instance_GI = GraphInstance( # Creating GraphInstance for recomputing manipulators
            id = instance.id,
            label = instance.label,
            data = adj,
            node_features = instance.node_features,
            # edge_features = E_v_bar_np,
            edge_weights = edge_weights_full,
            graph_features = instance.graph_features # G_v_bar_np
            )

        for manipulator in self.dataset.manipulators:
            if isinstance(manipulator, NodeCentrality):
                node_features_map = manipulator.node_info(instance_GI)
                node_feat = BaseManipulator.process_features(BaseManipulator, instance.node_features, node_features_map, self.dataset.node_features_map)

        self.instance_GI = GraphInstance( # Creating GraphInstance for Oracle's prediction
            id = instance.id,
            label = instance.label,
            data = adj,
            node_features = node_feat,
            # edge_features = E_v_bar_np,
            edge_weights = edge_weights_full,
            graph_features = instance.graph_features # G_v_bar_np
            )        

        # self.f_v = self.oracle.predict(instance).clone().detach() # Get GCN prediction
        print(f"initial prediction (instance): {self.oracle.predict(instance).clone().detach()}")
        self.f_v = self.oracle.predict(instance_GI).clone().detach() # Get GCN prediction
        print(f"initial prediction (instance_GI): {self.oracle.predict(instance_GI).clone().detach()}")
        input()

        # self.f_v = torch.tensor(self.oracle.predict(instance), dtype=torch.long) # Get GCN prediction
        self.g_v = self.f_v # CF predicted class
        noise_std = 0 # 1e-10 # Initialization of P_hat with noise to break symmetry
        N = instance.data.shape[0]

        if self.debugging: print(f"{Color.YELLOW}Initial prediction (f_v): {Color.RESET}{self.f_v}") # Debugging

        if not self.extended: # Use Base version of the explainer
            self.A_v = torch.tensor(instance.data, dtype=torch.float64) # instance.data is the adjacency matrix
            self.P_hat = torch.ones_like(self.A_v, requires_grad=False) # + noise_std * torch.randn_like(self.A_v) # Initialization of P_hat
            
        elif self.extended: # Use extended versione of the explainer (i.e. inverting the roles of A_v and P)
            # self.A_v = torch.ones(instance.data.shape, dtype=torch.float64) # Assume adjacency matrix full of ones
            
            self.A_v = torch.ones(int(N * (N+1) / 2))

            # missing_edges = np.where(instance.data == 0) # Missing A_v edges
            missing_edges_triu = torch.tensor(instance.data == 0).triu().nonzero().t()
            missing_edges_vec = (missing_edges_triu[0] * N + missing_edges_triu[1] - missing_edges_triu[0] * (missing_edges_triu[0]+1) / 2).int()
            # missing_nodefeatures = np.where(instance.node_features == 0) # Missing node_features edges        
            
            # self.P_hat = torch.tensor(instance.data, requires_grad=False) # Initialization of P_hat: Perturbation Matrix for Edges
            self.P_init = torch.ones_like(self.A_v, requires_grad=False) # Initialization of P_hat: Perturbation Matrix for Edges

            # self.P_init[missing_edges] = self.γ_edges # (Adjacency matrix is full of ones) P stores the zero edges (their value is γ_edges)
            self.P_init[missing_edges_vec] = self.γ_edges # (Adjacency matrix is full of ones) P stores the zero edges (their value is γ_edges)

            self.P_init += noise_std * torch.randn_like(self.A_v)
            self.P_init.requires_grad_(False) # Disable backpropagation
            # self.P_hat += noise_std * torch.rand_like(self.A_v)

            self.A_v = torch.ones(instance.data.shape, dtype=torch.float64) # Assume adjacency matrix full of ones

            if self.debugging: print(f"P_init == instance.data: {torch.equal(self.P_init, torch.tensor(instance.data))}") # Debugging: γ_edges != 0 ←→ it prints False

            if self.update_node_feat: # (Two branches initialize identically)
                if not self.change_node_feat: # Enable only discarding or adding of node features (node feature gating)
                    self.P_node_hat = torch.ones(instance.node_features.shape, requires_grad=True) # Initialization of P_node_hat: Perturbation Matrix for Node Features
                elif self.change_node_feat: # Allow node features to change features
                    # self.P_node_hat = torch.tensor(instance.node_features, requires_grad=False) # Initialization Ⅰ of P_node_hat: Perturbation Matrix for Node Features. Init as node_features.
                    # self.P_node_hat[missing_nodefeatures] = self.γ_node_feat # Add node features where they are missing (0)
                    # self.P_node_hat.requires_grad_(True) # Enable backpropagation
                    self.P_node_hat = torch.ones(instance.node_features.shape, requires_grad=True) # Initialization Ⅱ of P_node_hat: Perturbation Matrix for Node Features. Init as ones.

            # if self.debugging: print(f"P_init: {self.P_init}")
            self.P_sym = torch.zeros(instance.data.shape, dtype=torch.float32, requires_grad=False)
            i, j = np.triu_indices(N)
            self.P_sym[i,j] = self.P_init
            if self.directed:
                self.P_hat = self.P_sym
            else:
                self.P_hat = self.P_sym + self.P_sym.t() - torch.diag(torch.diag(self.P_sym)) # Symmetrizing P_hat

        # P_triu = torch.triu(self.P_init, diagonal=0)
        # self.P_hat = 0.5 * (self.P_init + self.P_init.t()) # Symmetrizing P_hat
        self.P_hat.requires_grad_(True) # Enable backpropagation
        # if self.debugging: print(f"P_hat symmetrized: {self.P_hat}")

        # Node features
        self.x = torch.tensor(instance.node_features, dtype=torch.float64) # Feature vector for v [3.1]
        self.v = (self.A_v, self.x) # [3.1]

        # Edge features
        self.e = torch.tensor(instance.edge_features, dtype=torch.float64) # Edge features vector
        # Creating self.edge_features to store features for all possible edges. If edge is not present, its features are set to 1 for all dimensions.
        self.feat_dim = instance.edge_features.shape[1] # Extracting edge features dimansionality
        self.edge_features = torch.zeros(instance.data.shape + (self.feat_dim,), dtype=torch.float64, requires_grad=False) # Edge features matrix for all possible edges (full of ones in each dimension)
        self.edge_indices = torch.where(torch.tensor(instance.data) != 0) # Indices of existing edges in Adjacency Matrix (integer tensor) 
        self.edge_features[self.edge_indices] = self.e # Assigning existing edge features
        self.edge_features = self.edge_features.squeeze(-1)

        # Graph features 
        # self.g = torch.tensor(instance.graph_features, dtype=torch.float64) # Graph features

        if self.change_all_feat: # Allow all features to change freely (node, edge, graph features)
            # self.P_edge_hat = torch.ones(instance.edge_features.shape, requires_grad=True) # Initialization of P_edge_hat: Perturbation Matrix for Edge Features
            self.P_edge_hat = torch.ones(instance.data.shape + (self.feat_dim, ), dtype=torch.float64, requires_grad=True) # Initialization of P_edge_hat: Perturbation Matrix for Edge Features
            # self.P_graph_hat = torch.ones(instance.graph_features.shape, requires_grad=True) # Initialization of P_graph_hat: Perturbation Matrix for Graph Features

        self.v_bar_opt = (torch.tensor(instance.data), self.x) # Initializing optimal CF with the instance itself

        self.opt_flag = False # CF not found yet
        edge_indices = np.where(instance.data != 0) # Indices of edges (int array)
        self.edge_weights_opt = instance.data[edge_indices] # Optimal CF edge weights (real array)

        if self.visualize: self.pos = nx.spring_layout(nx.from_numpy_array(instance.data)) # Fix graph orientation 

        self.lr_reduction_epoch = self.K // 5
        for k in range(int(self.K)):
            if self.debugging: print(f"Iteration: {self.k}")
            self.k = k

            self.__get_CF_example(instance) # Compute CF
            if self.opt_flag: break # Breaking at first CF found (Different from paper algorithm → If Efficiency is preferred: avoiding extra loop iterations to find better (closer) CF))

            loss = self.__calculate_loss(instance) # Compute Loss
            # self.oracle.model.zero_grad()  # clear previous grads
            # loss.sum().backward(retain_graph=True)
            # print(f"self.P_hat.grad.clone(): {self.P_hat.grad.clone()}")
            # input("new grads computed")

            # Backpropagate the loss and compute gradient of P_hat and P_node_hat
            loss.backward(retain_graph=True) # Retaining graph to sum future backward gradients
            
            # Manually retrieve and store gradients
            self.A_v_bar.grad = torch.autograd.grad(loss, self.A_v_bar, retain_graph=True)[0]
            if self.update_node_feat: self.N_v_bar.grad = torch.autograd.grad(loss, self.N_v_bar, retain_graph=True)[0]

            if not self.valid_CF: # L_pred contributing to Total Loss
                self.A_v_bar.grad += torch.zeros_like(self.A_v_bar, dtype=torch.double) # Initialize A_v_bar gradients with zeroes
                # Adding gradients from self.w.grad to the positions specified in self.edge_indices, i.e. they are the non-zero and contributing edges to the classification
                # print(f"self.w.grad: {self.w.grad}")
                # print(f"self.w.grad.shape: {self.w.grad.shape}")
                # print(f"self.P_hat.grad: {self.P_hat.grad}")
                # print(f"self.P_hat.grad.shape: {self.P_hat.grad.shape}")
                if self.w.grad is not None: self.A_v_bar.grad[self.edge_indices_ones] += self.w.grad
                
                # Adding L_pred contribution to the L_dist
                if self.update_node_feat:             
                    self.N_v_bar.grad += torch.zeros_like(self.N_v_bar, dtype=torch.double)            
                    self.N_v_bar.grad += self.x_.grad.clone() # Adding L_pred contribution to the L_dist
                
            self.A_v_bar.backward(torch.ones_like(self.A_v_bar)) # Perform backward pass
            if self.update_node_feat: self.N_v_bar.backward(torch.ones_like(self.N_v_bar)) # Perform backward pass            

            with torch.no_grad():  # Update without tracking the gradients further
                if self.debugging: print(f"P_hat before update: {self.P_hat}")
                # # grad_max = self.P_hat.grad == max(self.P_hat.grad) # Allow only max gradient to update P_hat (sparse updates)
                # idx_max = torch.argmax(torch.abs(self.P_hat.grad))
                # i_max = idx_max // self.P_hat.grad.size(1)
                # j_max = idx_max % self.P_hat.grad.size(1)
                # grad_max = torch.zeros_like(self.P_hat.grad, dtype=self.P_hat.grad.dtype)
                # grad_max[i_max, j_max] = self.P_hat.grad[i_max, j_max]
                # # print(f"P_hat.grad: {self.P_hat.grad}")
                # # print(f"grad_max: {grad_max}")
                # # self.P_hat -= self.α * grad_max # Gradient update step with learning rate
                # # input()                

                # print(f"self.A_v_bar: {self.A_v_bar}")
                present_mask = (self.A_v_bar == 1).to(self.P_hat.grad.dtype)
                missing_mask = 1.0 - present_mask

                print(f"self.P_hat.grad: {self.P_hat.grad}")
                self.P_hat.grad[self.edge_indices_ones] = self.w.grad.clone().detach().float()
                print(f"self.P_hat.grad: {self.P_hat.grad}")
                # input()
                # print(f"- self.α * self.P_hat.grad * present_mask: {- self.α * self.P_hat.grad * present_mask}")
                # print(f"+ self.α * self.P_hat.grad * missing_mask: {+ self.α * self.P_hat.grad * missing_mask}")

                self.P_hat += self.α * self.P_hat.grad # Gradient update step with learning rate 
                # self.P_hat -= self.α * (self.P_hat.grad + 1e-8 * torch.randn_like(self.P_hat)) * present_mask  # Gradient update step with learning rate with some jitter to break symmetry
                # self.P_hat += self.α * (self.P_hat.grad + 1e-8 * torch.randn_like(self.P_hat)) * 1e-7 * missing_mask  # Gradient update step with learning rate with some jitter to break symmetry
                # self.P_hat -= 1e0 * self.α * (self.P_hat.grad + 1e-1 * torch.randn_like(self.P_hat)) * present_mask * self.edge_importance # Gradient update step with edge importance
                # self.P_hat += 1e-6 * self.α * (self.P_hat.grad + 1e-1 * torch.randn_like(self.P_hat)) * missing_mask * self.edge_importance  # Gradient update step with edge importance
                # self.P_hat -= 1e-1 * self.α * present_mask * self.edge_importance # + (1e-5 * torch.randn_like(self.P_hat)) # Gradient update step with edge importance
                # self.P_hat += 1e-6 * self.α * missing_mask * self.edge_importance # + (1e-5 * torch.randn_like(self.P_hat)) # Importance update step with edge importance
                # self.P_hat -= 1e-1 * self.α * present_mask * self.P_hat.grad # + (1e-5 * torch.randn_like(self.P_hat)) # Gradient update step with P_hat.grad
                # self.P_hat += 1e-6 * self.α * missing_mask * self.P_hat.grad # + (1e-5 * torch.randn_like(self.P_hat)) # Importance update step with P_hat.grad


                
                # print(f"self.P_hat: {self.P_hat}")
                # print(torch.sigmoid(self.P_hat) > 0.5)
                # input()

                self.P_hat.copy_(0.5 * (self.P_hat + self.P_hat.t())) # Symmetrizing P_hat after update

                if self.debugging: print(f"P_hat.grad: {self.P_hat.grad}"); print(f"P_hat updated: {self.P_hat}"); print(f"self.A_v_bar: {self.A_v_bar}")

                if self.update_node_feat: # self.P_node_hat (nodes features perturbation matrix) exists only in the extended algorithm where node features perturbations are allowed
                    self.P_node_hat -= self.α * self.P_node_hat.grad # Gradient update step with learning rate
                    if self.debugging: print(f"P_node_hat.grad: {self.P_node_hat.grad}")
            
            self.P_hat.grad.zero_() # zero gradients for next iteration
            if self.update_node_feat: # self.P_node_hat exists only in the extended algorithm where node features perturbations are allowed
                self.P_node_hat.grad.zero_() # zero gradients for next iteration

            if self.visualize and False: # and self.opt_flag:
                # print(instance.data, "\nCF Adj: ", self.A_v_bar.data)
                instance_graph = nx.from_numpy_array(instance.data)
                CF_graph = nx.from_numpy_array(self.A_v_bar.clone().detach().numpy())

                # Draw graphs. Node colours are the mean of the node features
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                nx.draw(instance_graph, pos=self.pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
                axes[0].set_title(f"Initial Graph | Predicted Class: {self.f_v}")
                nx.draw(CF_graph, pos=self.pos, ax=axes[1], with_labels=True, cmap='cool', node_color=self.N_v_bar.clone().detach().numpy().mean(axis=1), edge_color='gray')
                axes[1].set_title(f"Counterfactual Graph | Predicted Class: {self.g_v}")
                plt.show()    

            if self.debugging and self.visualize: print(f"Iteration {self.k} finished | Press Enter to continue")
            elif self.debugging: input(f"Iteration {self.k} finished | Press Enter to continue")

            if (self.k+1) % self.lr_reduction_epoch == 0 and self.decay_α:
                self.α *= 0.1
                print(f"Learning rate reduced to {Color.YELLOW}{self.α:<.6f}f{Color.RESET} at iteration {Color.YELLOW}{self.k}{Color.RESET}/{self.K}")
        
        edge_indices = torch.where(self.v_bar_opt[0] != 0) # (int tensor)
        edge_weights = self.v_bar_opt[0][edge_indices] # (real tensor)
        edge_weights = edge_weights.clone().detach().numpy()
        edge_features = self.edge_features[edge_indices].clone().detach().numpy() # Edge features for the existing edges (np.array)
        
        v_bar_opt_GI = GraphInstance(
            id = instance.id,
            label = self.g_v_bar_pred, # Predicted class of the CF
            data = self.v_bar_opt[0].clone().detach().numpy(),
            node_features = self.v_bar_opt[1].clone().detach().numpy(),
            edge_features = edge_features,
            edge_weights = edge_weights,
            graph_features = instance.graph_features
            )
        
        if self.debugging: print(f"{Color.MAGENTA}Opt predicted class: {Color.RESET}{self.oracle.predict(v_bar_opt_GI)}")
        if self.visualize: # and self.opt_flag:
            # print(instance.data, "\n", self.A_v_bar)
            # print("NF")
            # print(instance.node_features)
            # print(self.N_v_bar.clone().detach().numpy())
            instance_graph = nx.from_numpy_array(instance.data)
            CF_graph = nx.from_numpy_array(self.A_v_bar.clone().detach().numpy())

            # Draw graphs. Node colours are the mean of the node features
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # print(f"fist_nf: {first_nf}")
            # print(f"instance.N_v_bar: {self.N_v_bar}")
            # print(f"self.x: {self.x}")
            # print(f"node_features_map: {node_features_map}")

            # print(f"instance.data: {instance.data}")
            # print(f"self.A_v_bar: {self.A_v_bar}")
            nx.draw(instance_graph, pos=self.pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
            axes[0].set_title(f"Initial Graph | Predicted Class: {self.f_v}")
            nx.draw(CF_graph, pos=self.pos, ax=axes[1], with_labels=True, cmap='cool', node_color=self.N_v_bar.clone().detach().numpy().mean(axis=1), edge_color='gray')
            axes[1].set_title(f"Counterfactual Graph | Predicted Class: {self.g_v} - K: {self.k}")
            fig.suptitle(f"True label: {instance.label}")
            plt.show()           

        if self.debugging: print(f"{Color.GREEN}{self.f_v}{Color.RESET} | {Color.MAGENTA}{self.oracle.predict(instance)}{Color.RESET} | {Color.YELLOW}{self.oracle.predict(v_bar_opt_GI)}{Color.RESET} | {self.g_v_bar_pred}")

        if not self.opt_flag:
            if not self.node_classification: print(f"{Color.RED}CF not found{Color.RESET}, original: {Color.MAGENTA}{self.f_v}{Color.RESET}")
            else: print(f"{Color.RED}CF not found{Color.RESET}, original: {Color.MAGENTA}{self.f_v[self.node_id]} [^node id: {self.node_id}]{Color.RESET}")
            return instance

        return v_bar_opt_GI


    def __get_CF_example(self, instance):
        """
        Compute the CF for ```instance```.
        Function GET_CF_EXAMPLE() of Algorithm 1 [5.4].
        lines in brackes '[line 0]' refer to lines of pseudo code in the paper.
        """
        # [line 1]: P ← threshold(σ(P_hat))
        P_sigmoid = torch.sigmoid(self.P_hat) # Threshold on sigmoid of P_hat
        mask = (P_sigmoid > .5).float().clone() # Hard mask (> instead of >=, with >= also 0 values in P_hat evaluate to 1 and a fully connected matrix is obtained)
        P = P_sigmoid + (mask - P_sigmoid).detach() # Gradients can flow through P

        self.A_v_bar = P * self.A_v # [line 2]: Ā_v = P ⊙ A_v
        # self.A_v_bar.fill_diagonal_(1) # Add self-loops Eq(4) [5.2]
        A_v_bar_np = self.A_v_bar.clone().detach().numpy() # Conversion to numpy array (GraphInstance requires np arrays)
        # print(f"A_v_bar_np.shape: {A_v_bar_np.shape}")
        A_v_bar_np = torch.ones_like(self.A_v_bar).clone().detach().numpy()
        # print(f"A_v_bar_np.shape: {A_v_bar_np.shape}")
        
        self.edge_indices = torch.where(self.A_v_bar != 0) # (integer tensor)
        edge_weights = self.A_v_bar[self.edge_indices] # Values of A_v_bar edges, i.e. weights for the presence of edges (real tensor)
        edge_weights_np = edge_weights.clone().detach().numpy() # Conversion to numpy array (GraphInstance requires np arrays)
        # print(f"edge_weights_np.shape: {edge_weights_np.shape}")

        self.edge_indices_ones = torch.where(torch.ones_like(self.A_v_bar) != 0) # (integer tensor)
        edge_weights_full = torch.zeros_like(self.A_v_bar) 
        edge_weights_full[self.edge_indices] = edge_weights # weights having also 0s for missing edges
        edge_weights_full = edge_weights_full[self.edge_indices_ones]
        edge_weights_np = edge_weights_full.clone().detach().numpy() # Conversion to numpy array (GraphInstance requires np arrays)
        # print(f"edge_weights_np.shape: {edge_weights_np.shape}")
        # print(f"edge_weights_full: {edge_weights_full}")

        if self.update_node_feat: # Perturb node features
            if not self.change_node_feat: # Either discard or maintain the node feature (gating)
                # Repeat previous steps for N_v_bar
                N_sigmoid = torch.sigmoid(self.P_node_hat) # Threshold on sigmoid of P_node_hat
                mask = (N_sigmoid > .5).float().clone() # Hard mask (> instead of >=)
                N = N_sigmoid + (mask - N_sigmoid).detach() # Gradients can flow through N

                self.N_v_bar = N * self.x # N ⊙ x
            else: # Allow node features to change freely
                # self.N_v_bar = self.P_node_hat # For initialization Ⅰ
                self.N_v_bar = self.P_node_hat * self.x # N_hat ⊙ x (For initialization)
        else: # Keep same node features
            # self.N_v_bar = self.x.clone()
            self.N_v_bar = self.x.clone().detach() # Feature vector for v [3.1]
            self.N_v_bar.requires_grad_(False) # Disable backpropagation

        if self.change_all_feat: # Perturb edge and graph features            
            self.E_v_bar = self.P_edge_hat * self.edge_features # E_v_bar = P_edge_hat ⊙ edge_feats
            self.E_v_bar = self.E_v_bar[self.edge_indices] # Take only the edge features for the existing edges
            # self.G_v_bar = self.P_graph_hat * self.g # G_v_bar = P_graph_hat ⊙ graph_feats
        else: # Keep same edge, graph features
            edge_features = self.edge_features[self.edge_indices] # Edge features for the existing edges
            self.E_v_bar = edge_features
            self.E_v_bar.requires_grad_(False) # Disable backpropagation
            # print(f"self.E_v_bar shape: {self.E_v_bar.shape}")

            self.E_v_bar_full = self.edge_features
            self.E_v_bar_full = self.E_v_bar_full[self.edge_indices_ones]
            self.E_v_bar_full.requires_grad_(False) # Disable backpropagation
            # print(f"self.E_v_bar_full shape: {self.E_v_bar_full.shape}")


            # self.G_v_bar = torch.tensor(instance.graph_features, dtype=torch.float64)
            # self.G_v_bar.requires_grad_(False) # Disable backpropagation
        
        v_bar_cand = (self.A_v_bar, self.N_v_bar) # [line 3]: v_bar_cand ← (Ā_v, x)   
        
        N_v_bar_np = self.N_v_bar.clone().detach().numpy() # Conversion to numpy array (GraphInstance requires np arrays)
        # E_v_bar_np = self.E_v_bar.clone().detach().numpy() # Conversion to numpy array (GraphInstance requires np arrays)
        E_v_bar_np = self.E_v_bar_full.clone().detach().numpy() # Conversion to numpy array (GraphInstance requires np arrays)
        # G_v_bar_np = self.G_v_bar.clone().detach().numpy() # Conversion to numpy array (GraphInstance requires np arrays)

        print(f"A_v_bar_np shape: {A_v_bar_np.shape}")
        print(f"N_v_bar_np shape: {N_v_bar_np.shape}")
        # print(f"N_v_bar_np: {N_v_bar_np}")
        print(f"edge_weights_np shape: {edge_weights_np.shape}")
        print(f"E_v_bar_np shape: {E_v_bar_np.shape}")
        input()

        temp_A_v_bar_GI = GraphInstance( # Creating GraphInstance for recomputing manipulators
            id = instance.id,
            label = instance.label,
            data = A_v_bar_np,
            node_features = N_v_bar_np,
            edge_features = E_v_bar_np,
            edge_weights = edge_weights_np,
            graph_features = instance.graph_features # G_v_bar_np
            )

        # print(" - " * 30)
        # print("\n\nXPlore manipulator - before")
        # print("instance.node_features:\n", self.x)
        for manipulator in self.dataset.manipulators:
            if isinstance(manipulator, NodeCentrality):
                # print(f"manipulator: {manipulator}")
                # print(f"instance.data: {instance.data}")
                # print(f"self.A_v_bar: {self.A_v_bar}")
                node_features_map = manipulator.node_info(temp_A_v_bar_GI)
                N_v_bar_np = BaseManipulator.process_features(BaseManipulator, N_v_bar_np, node_features_map, self.dataset.node_features_map)
                self.N_v_bar = torch.tensor(N_v_bar_np, dtype=self.N_v_bar.dtype, requires_grad=self.N_v_bar.requires_grad)
                # print(f"node_features_map: {node_features_map} |\norignal {manipulator.node_info(instance)}")
                # print(f"Adj: {A_v_bar_np} |\norignal {instance.data}")
                # print(f"N_v_bar_np: {N_v_bar_np}")
                # print("Manipulation")
                # print(manipulator.process_instance(instance))
        # print("XPlore manipulator - after")
        # print("instance.node_features:\n", self.x)
        # input()

        self.A_v_bar_GI = GraphInstance( # Creating GraphInstance for Oracle's prediction
            id = instance.id,
            label = instance.label,
            data = A_v_bar_np,
            node_features = N_v_bar_np,
            edge_features = E_v_bar_np,
            edge_weights = edge_weights_np,
            graph_features = instance.graph_features # G_v_bar_np
            )

        # Computing f(v_bar_cand)
        if self.diffusion_flag:
            self.g_v_logits = self._real_predict_gradients_diffusion(self.A_v_bar_GI)
        else:
            self.g_v_logits = self._real_predict_gradients(self.A_v_bar_GI) # Oracle CF prediction → returns probabilities
            # self.g_v_logits = self._real_predict_gradients(temp_A_v_bar_GI) # Oracle CF prediction → returns probabilities
        g_v_bar_pred = torch.argmax(self.g_v_logits, dim=-1) # Getting the predicted class
        print(f"new pred: {g_v_bar_pred}"); input()
        # if torch.any(g_v_bar_pred != self.oracle.predict(A_v_bar_GI)): input(f"{Color.RED}Warning{Color.RESET} | Oracle's prediction is ambiguous")
        self.g_v_bar_pred = g_v_bar_pred

        self.valid_CF = False # Flag for valid CF
        with torch.no_grad():
            # [line 4]: if f(v) ≠ f(v_bar_cand) (valid CF: initial prediction is different from current one)
            if not self.node_classification and torch.all(self.f_v != g_v_bar_pred) or\
            self.node_classification and self.f_v[self.node_id]!= g_v_bar_pred[self.node_id]:
                self.valid_CF = True # Valid CF is found
                v_bar = v_bar_cand # [line 5]: v_bar ← v_bar_cand

                if not self.opt_flag: # [line 6]: if not v_bar_opt then
                    self.v_bar_opt = v_bar # [line 7]: v_bar_opt ← v_bar # First CF
                    if not self.node_classification: print(f"{Color.GREEN}Found valid counterfactual - {Color.CYAN}Counterfactual predicted class: {Color.GREEN}{g_v_bar_pred}{Color.CYAN} instead of {Color.MAGENTA}{self.f_v}{Color.RESET} | K: {self.k}")  # Debugging
                    else: print(f"{Color.GREEN}Found valid counterfactual [^node id:{self.node_id}] - {Color.CYAN}Counterfactual predicted class: {Color.GREEN}{g_v_bar_pred[self.node_id]}{Color.CYAN} instead of {Color.MAGENTA}{self.f_v[self.node_id]}{Color.RESET} | K: {self.k}")  # Debugging
                    self.edge_weights_opt = edge_weights_np
                    self.opt_flag = True # CF found
                    self.g_v = g_v_bar_pred

                # Check for 'closer' CF, i.e. CF that requires less perturbations
                elif self.__distance(self.v[0], v_bar[0]) < self.__distance(self.v[0], self.v_bar_opt[0]): # [line 8]: else if d(v, v_bar) ≤ d(v, v_bar*) then
                    self.v_bar_opt = v_bar # [line 9]: v_bar* ← v_bar # Keep track of best CF
                    self.edge_weights_opt = edge_weights_np
                    if not self.node_classification: print(f"{Color.BLUE}Found new best counterfactual - {Color.CYAN}Counterfactual predicted class: {Color.RESET}{g_v_bar_pred} | K: {self.k}")  # Debugging
                    else: print(f"{Color.BLUE}Found new best counterfactual [^node id:{self.node_id}] - {Color.CYAN}Counterfactual predicted class: {Color.RESET}{g_v_bar_pred[self.node_id]} | K: {self.k}")  # Debugging
                    self.g_v = g_v_bar_pred

        return

    def __calculate_loss(self, instance):
        """
        Loss function based on:
        L = Lpred(v,  v_bar | f, g) + β Ldist(v, v_bar | d) Eq(1) [4].
        L_pred(v, v_bar | f, g) = -1 [f(v) = f(v_bar)] * L_NLL(f(v), g(v_bar)) Eq(5) [5.3].
        L_dist(v, v_bar | d): the element-wise difference between A_v and A_v_bar, i.e., the number of edges removed.
        """
        # 1. Prediction loss: L_pred
        # Get the original prediction and counterfactual prediction
        if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss): # Single-label classification (single-class/multi-class)
            inputs = self.g_v_logits
            targets = self.f_v
        elif isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss): # Multi-label classification
            # inputs = self.g_v_logits - (self.g_v_logits - self.g_v_logits.argmax(dim=-1).unsqueeze(-1)).detach()
            inputs = self.g_v_logits # F.gumbel_softmax(self.g_v_logits, dim=-1, hard=True)
            # targets = (torch.nn.functional.one_hot(self.f_v, num_classes=self.dataset_classes).sum(dim=1)>0).float() # Create a multi-one-hot from a vector with many class labes eg. [0,2,3] -> [1,0,1,1] (mask to prevent eventual class label repetitions eg. [2,4,4,5])
            targets = torch.nn.functional.one_hot(self.f_v, num_classes=self.dataset_classes).float() # having target label for each node -> one-hot for each label
            # print(f"targets: {targets.shape} - inputs: {inputs.shape}")
        elif isinstance(self.loss_fn, torch.nn.NLLLoss): # CE is just NNL with log+softmax included
            # print(f"self.g_v_logits: {self.g_v_logits}")
            inputs = F.log_softmax(self.g_v_logits, dim=0) # Prediction log probabilities required for NLL loss
            targets = self.f_v

        if self.node_classification:
            inputs = inputs[self.node_id]
            targets = targets[self.node_id]

        # If f(v) == f(v_bar), the loss is 0; otherwise, compute NLL loss
        if not self.valid_CF: 
            # Use negative log-likelihood loss (NLL) between for predicted logits and ground truth (Loss Function Optimization) [5.3]
            L_pred = -1 * self.loss_fn(inputs, targets)
        else:
            L_pred = 0  # No loss if predictions are different

        # # --- Edge sensitivity weights ---
        # edge_indices = torch.where(torch.ones_like(self.A_v_bar))
        # self.edge_importance = torch.ones_like(self.A_v_bar)
        # with torch.no_grad():
        #     for i, j in zip(*edge_indices):
        #         A_tmp = self.A_v_bar.clone()
        #         A_tmp[i,j] = 0 if A_tmp[i,j] else 1 # invert edge presence

        #         edge_indx = torch.where(A_tmp != 0) # (integer tensor)
        #         edge_weights = A_tmp[edge_indx] # Values of A_tmp edges, i.e. weights for the presence of edges (real tensor)
        #         edge_weights_np = edge_weights.clone().detach().numpy() # Conversion to numpy array (GraphInstance requires np arrays)

        #         tmp_GI = GraphInstance(
        #             id=instance.id,
        #             label=instance.label,
        #             data=A_tmp.clone().detach().numpy(),
        #             node_features=instance.node_features,
        #             # edge_features=instance.edge_features,
        #             edge_weights = edge_weights_np,
        #             graph_features=instance.graph_features
        #         )

        #         for manipulator in self.dataset.manipulators:
        #             if isinstance(manipulator, NodeCentrality):
        #                 node_features_map = manipulator.node_info(tmp_GI)
        #                 N_v_bar_np = BaseManipulator.process_features(BaseManipulator, instance.node_features, node_features_map, self.dataset.node_features_map)

        #         tmp_GI = GraphInstance(
        #             id=instance.id,
        #             label=instance.label,
        #             data=A_tmp.clone().detach().numpy(),
        #             node_features=N_v_bar_np,
        #             # edge_features=instance.edge_features,
        #             edge_weights = edge_weights_np,
        #             graph_features=instance.graph_features
        #         )

        #         pred_tmp = self.oracle.predict(tmp_GI)
        #         # magnitude of change in prediction as edge importance
        #         self.edge_importance[i,j] = torch.norm(pred_tmp - self.g_v_logits)
        
        # self.edge_importance = 0.5 * (self.edge_importance + self.edge_importance.t())
        # print(f"edge_importance: {self.edge_importance}")


        # Select the target class for gradient signal
        # if self.dataset_classes == 2:
        #     target_class = 1 - self.f_v
        # else:
        #     target_class = (self.f_v + 1) % self.dataset_classes

        # if self.node_classification:
        #     logits_target = self.g_v_logits[self.node_id, target_class]
        # else:
        #     logits_target = self.g_v_logits[target_class]
        # return logits_target

        # # COUNTERFACTUAL MARGIN LOSS
        # targets = self.f_v

        # # target class (flip)
        # if self.dataset_classes == 2:
        #     target_class = 1 - targets
        # else:
        #     # fallback: pick next class
        #     target_class = (targets + 1) % self.dataset_classes

        # # Extract the logits from model output
        # inputs = self.g_v_logits

        # if self.node_classification:
        #     logit_orig = inputs[self.node_id, targets]
        #     logit_target = inputs[self.node_id, target_class]
        # else:
        #     logit_orig = inputs[targets]
        #     logit_target = inputs[target_class]

        # # Margin loss: encourages target class to surpass original
        # L_pred_margin = logit_orig - logit_target
        
        # 2. Distance loss: L_dist
        # Compute the number of edges removed (element-wise difference between A_v and A_v_bar) and the change in node features
        edge_diff = torch.abs(torch.tensor(instance.data) - self.A_v_bar) # Element-wise absolute difference for edges
        node_diff = torch.abs(torch.tensor(instance.node_features) - self.N_v_bar) # Element-wise absolute difference for nodes features
        D_edges = edge_diff.sum() # L1-norm for edges: count the number of edges changed
        D_nodes = node_diff.sum() # L1-norm for nodes: distance between the node features changed
        L_dist = D_edges + D_nodes # Total L1-norm
        
        # 3. Total loss
        # print("Loss:", L_pred, L_dist)
        total_loss = - (L_pred - self.β * L_dist)
        # total_loss = - (L_pred_margin - self.β * L_dist)
        
        # return L_pred
        return total_loss
    
    # Define costume function for gradients computation. Torch Geometric ones do not handle gradients properly:
    # they do not require grads for the weights matrix, hence we cannot compute how the perturbation on P influences the loss.
    def _real_predict_gradients(self, data_instance):
        return self._real_predict_proba_gradients(data_instance)

    def _real_predict_proba_gradients(self, data_inst):
        data_inst = self.to_geometric_gradients(data_inst)
        self.node_features = data_inst.x.to(self.oracle.device)
        self.edge_index = data_inst.edge_index.to(self.oracle.device)
        self.edge_weights = data_inst.edge_attr.to(self.oracle.device)            
        return self.oracle.model(self.node_features,self.edge_index,self.edge_weights, None).cpu().squeeze()

    def to_geometric_gradients(self, instance: GraphInstance, label=0) -> Data:
        self.adj = torch.from_numpy(instance.data).double().requires_grad_(True) # Adjacency matrix (Added requires_grad(True))
        self.x_ = torch.from_numpy(instance.node_features).double().requires_grad_(True) # Edge features (Added requires_grad(True))
        self.a = torch.nonzero(self.adj).int() # Edge indices
        self.w = torch.from_numpy(instance.edge_weights).double().requires_grad_(True) # Edge weights (Added requires_grad(True))
        label = torch.tensor(label).long()
        return Data(x=self.x_, y=label, edge_index=self.a.T, edge_attr=self.w)
    
    def _real_predict_diffusion(self, data_instance):
        return torch.argmax(self._real_predict_gradients_diffusion(data_instance), dim=-1)
    
    def _real_predict_gradients_diffusion(self, data_inst):
        """Compute prediction using the diffusion oracle for a single instance, keeping gradients."""
        data_inst = self.to_geometric_gradients(data_inst)
        
        node_features = data_inst.x.to(self.oracle.device).double()
        edge_index = data_inst.edge_index.to(self.oracle.device)
        edge_weights = data_inst.edge_attr.to(self.oracle.device).double()
        
        # Single graph -> batch tensor of zeros
        batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=self.oracle.device)

        # Initialize noisy labels
        B = 1
        y_t = torch.randn(B, self.oracle.dataset.num_classes, device=self.oracle.device, dtype=torch.double)

        betas, alphas, alphas_bar = self.oracle.get_noise_schedule(self.oracle.T)

        # Reverse diffusion loop
        for t in reversed(range(self.oracle.T)):
            t_vec = torch.full((B,), t, device=self.oracle.device, dtype=torch.long)
            out = self.oracle.model(
                node_features=node_features,
                edge_index=edge_index,
                edge_weight=edge_weights,
                batch=batch,
                y_t=y_t,
                t=t_vec
            )
            if isinstance(out, tuple) and len(out) == 4:
                eps_X_pred, eps_y_pred, eps_W_pred, label_logits = out
            else:
                eps_X_pred, eps_y_pred, eps_W_pred = out
                label_logits = None
            alpha_t = alphas[t]
            alpha_bar_t = alphas_bar[t]
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            y_prev = coef1 * (y_t - coef2 * eps_y_pred)
            if t > 0:
                sigma_t = torch.sqrt(betas[t])
                y_prev = y_prev + sigma_t * torch.randn_like(y_prev, device=self.oracle.device, dtype=y_prev.dtype)
            y_t = y_prev

        # Return logits (keep gradients for explainer)
        return y_t.squeeze()
    
    def set_node_id(self, node_id:int):
        """Call it when computing metrics and iterating over the nodes of the graph. Node id needed for the loss."""
        self.node_id = node_id

    @torch.no_grad()
    def __distance(self, v, CF):
        """
        Computes the L2-norm of the difference between ```v``` (A_v_bar tensor of initial instance) and ```CF``` (A_v_bar tensor of the CF).
        """        
        v = v.clone().detach()
        CF = CF.clone().detach()
        l2norm = torch.norm(v-CF, p=2) # L2-norm of the difference
        return l2norm

class Color:
    """Print nice console colors for readibility' sake"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'  