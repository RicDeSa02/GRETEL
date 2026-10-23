import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.utils.cfg_utils import retake_oracle
from src.dataset.manipulators.base import BaseManipulator
from src.dataset.manipulators.centralities import NodeCentrality
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os

import src.oracle.nn.torch_diffusion

class XPlore(Explainer):
    '''
    Algorithmic implementation of XPlore.
    Extended version of the Explainer based on
    Lucic et al. CF-GNNExplainer Counterfactual Explanations for Graph Neural Networks
    https://arxiv.org/abs/2102.03322
    Algorithm: given a graph G = (AG, X) where f (G) = y, generate the minimal perturbation,  ̄G = (ĀG, X), such that f (̄G) ≠ y. [5.4]
    '''

    #region init
    def init(self):
        """
        α: Learning Rate, K: Number of iterations, β: Loss controll (Ldist wrt Lpred) (Eq.1 [4]), γ_edges: Missing edges addition fraction (per edge)
        α: 0.1, K: 500, β: 0.5 // Paper best parameters (Hyperparameter Search [6.4])
        """
        # input("Training Complete")
        self.oracle = retake_oracle(self.local_config)
        self.manipulators = [type(m).__name__ for m in self.dataset.manipulators]
        self.CHEMICAL_DATASETS = {"MUTAG", "Mutagenicity", "AIDS", "BZR", "COX2", "DHFR", "PTC_MR", "PTC_FM", "PTC_FR", "PTC_MM", "NCI1", "NCI109"}
        
        local_params = self.local_config['parameters']
        self.α = local_params['alpha'] # α: Learning Rate
        self.K = local_params['K'] # K: Number of Iterations (to update pertubation matrices (P_hat, etc.))
        self.β = local_params['beta'] # β: Trade-off between Lpred and Ldist Eq.1 [4]
        self.extended = local_params['extended'] # Use extended version of the algorithm (allowing to add edges and not only to drop them)
        self.γ_edges = local_params['gamma_edge'] # γ: Add missing edges to the edge perturbation matrix (γ ∈ [0, 1])
        self.update_node_feat = local_params['update_node_feat'] # Allow to update node features (gate or change them freely)
        self.change_node_feat = local_params['change_node_feat'] # Allow node features to change freely instead of just keep or discard a given feature (gating)
        self.change_all_feat = local_params['change_all_feat'] # Allow all features to change freely (node, edge, graph features)
        # self.γ_node_feat = local_params['gamma_node_feat'] # γ: Add missing node features to the node features perturbation matrix (γ ∈ [0, 1]) (NOT USED FOR CURRENT INITIALIZATION Ⅱ)
        self.debugging = local_params['debugging'] # Print debugging code one iteration at a time
        self.visualize = local_params['visualize'] # Visualize initial graph and CF found (if no valid CF is found then it draws the CF at last iteration)
        self.vis_id = local_params['vis_id'] # Graph to visualize if self.visualize is true
        self.multi_label_classification = local_params['multi_label_classification'] # Whether target classification is multi-class
        self.dataset_classes = local_params['dataset_classes'] # Dataset classes/labels amount
        self.node_classification = local_params['node_classification'] # Whether to apply node classification
        self.boost_α = local_params['boost_alpha'] # Wheter to decay learning rate (α) during explainer iterations
        self.directed = local_params['directed'] # Wheter the graph is directed or undirected
        self.device = local_params['device']
        self.chem_flag = local_params.get('chem_flag', False)   

        # Node Features stats for normalization
        X = torch.cat([torch.as_tensor(g.node_features) for g in self.dataset.instances])     # [N*total_nodes, F]
        std = X.std(dim=0) # .clamp_min(1e-8).to(self.device)   
        self.feat_std = torch.where(std < 1e-6, torch.ones_like(std), std).to(self.device) # [F], 1 for constant features, avoid blowup in distance computation

        if not self.multi_label_classification:
            # self.loss_fn = torch.nn.BCELoss() # useless as model outputs more than one logits
            # self.loss_fn = torch.nn.NLLLoss() # redundant, just use CE
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else: # Multi-Label Classification
            self.loss_fn = torch.nn.BCEWithLogitsLoss() # use for Multi-Label Classification   

        self.diffusion_flag = True if isinstance(self.oracle, src.oracle.nn.torch_diffusion.OracleTorch) else False

        assert ((isinstance(self.K, float) or isinstance(self.K, int)) and self.K >= 1)
        assert ((isinstance(self.α, float) or isinstance(self.α, int)) and self.α >= 0 or True)
        assert ((isinstance(self.β, float) or isinstance(self.β, int)) and self.β >= 0 or True)
        assert (isinstance(self.extended, bool))
        assert ((isinstance(self.γ_edges, float) or isinstance(self.γ_edges, int)) and 0 <= self.γ_edges <= 1)
        assert (isinstance(self.update_node_feat, bool))
        assert (isinstance(self.change_node_feat, bool))
        # assert ((isinstance(self.γ_node_feat, float) or isinstance(self.γ_node_feat, int)) and 0 <= self.γ_node_feat <= 1)
        assert (isinstance(self.debugging, bool))
        assert (isinstance(self.visualize, bool))
        assert (isinstance(self.vis_id, int))
        assert (isinstance(self.multi_label_classification, bool))
        assert (isinstance(self.node_classification, bool))
        assert (isinstance(self.dataset_classes, int))

        # if not self.extended:
        #     self.update_node_feat = False
        #     self.change_node_feat = False
        #     self.change_all_feat = False
        # elif not self.update_node_feat:
        #     self.change_node_feat = False
        #     self.change_all_feat = False
        # elif not self.change_node_feat:
        #     self.change_all_feat = False

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
            
        # if 'gamma_node_feat' not in local_config['parameters']:
        #     local_config['parameters']['gamma_node_feat'] = 0

        if 'debugging' not in local_config['parameters']:
            local_config['parameters']['debugging'] = False

        if 'visualize' not in local_config['parameters']:
            local_config['parameters']['visualize'] = False
        
        if 'vis_id' not in local_config['parameters']:
            local_config['parameters']['vis_id'] = -1

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

        if 'boost_alpha' not in local_config['parameters']:
            local_config['parameters']['boost_alpha'] = False

        if 'directed' not in local_config['parameters']:
            local_config['parameters']['directed'] = False

        if 'device' not in local_config['parameters']:
            local_config['parameters']['device'] = self.oracle.device
                # ("cuda"
                # if torch.cuda.is_available() or torch.cuda.device_count()
                # else "mps"
                # if torch.backends.mps.is_available()
                # else "cpu")
               
        self.fold_id = self.local_config['parameters'].get('fold_id',-1)

    #region Explain
    def explain(self, instance):
        """
        Find a Counterfactual for ```instance```. The closest among the ones found will be returned.
        """
        # if instance.id > 10: input() #XXX
        # if instance.id < 500: return instance #XXX      
        if self.visualize and instance.id != self.vis_id and self.vis_id != -1:
            return instance

        self.oracle.model.float()
        self.oracle.model.eval()
        self.oracle.model.to(self.device)

        # instance.node_features = np.zeros_like(instance.node_features)
        # version = "XPlore++" if self.change_node_feat and self.update_node_feat and self.extended else "XPlore+" if self.update_node_feat and self.extended else "XPlore" if self.extended else "CF-GNNExplainer"
        # try:
        #     ff"dataset: {self.dataset.dataset_name} - {version}")
        # except:
        #     print(f"dataset: Unknown - {version}")
        
        # if self.diffusion_flag:
        #     self.f_v = self._real_predict_diffusion(instance).clone().detach()
        # else:
        #     self.f_v = self.oracle.predict(instance).clone().detach() # Get GCN prediction

        # first_nf = instance.node_features.copy()

        self.data = torch.tensor(instance.data, dtype=torch.float32, device=self.device)
        self.batch = torch.zeros(instance.node_features.shape[0], dtype=torch.long, device=self.device)

        if not self.diffusion_flag:
            self.f_v = self.oracle.predict(instance).clone().detach().to(self.device) # Get GCN prediction

        #region Diffusion flag
        elif self.diffusion_flag:
            # edge_index = torch.nonzero(self.data).int().T
            # edge_indices = torch.where(self.data != 0) # (int tensor)
            # edge_weights = torch.tensor(self.data.clone().detach()[edge_indices], dtype=torch.float32, device=self.device)
            # input()
            edge_indices = torch.where(self.data != 0) # (int tensor)
            edge_weights = self.data.detach().clone()[edge_indices[0], edge_indices[1]]
            # input()

            self.adj_full = torch.ones_like(self.data)
            # edge_indices = np.where(instance.data != 0)
            # edge_weights = instance.data[edge_indices]
            edge_weights_full = torch.zeros_like(self.data) 
            edge_weights_full[edge_indices] = edge_weights # weights having also 0s for missing edges
            edge_weights_full = edge_weights_full.flatten()
            edge_index_full = self.adj_full.nonzero(as_tuple=False).T

            # input("enter to continue")

            # self.f_v = self.oracle.predict(instance).clone().detach() # Get GCN prediction
            # self.f_v = torch.argmax(self.oracle.model(torch.tensor(instance.node_features, dtype=torch.float32, device=self.device),edge_index,edge_weights,None).clone().detach(), dim=-1).squeeze(-1) # Get GCN prediction
            if self.diffusion_flag:
                self.f_v = torch.argmax(self.predict_diffusion(torch.tensor(instance.node_features, dtype=torch.float32, device=self.device),edge_index_full,edge_weights_full), dim=-1).squeeze(-1) # Get diffusion prediction
            else:
                self.f_v = torch.argmax(self.oracle.model(torch.tensor(instance.node_features, dtype=torch.float32, device=self.device),edge_index_full,edge_weights_full,self.batch).clone().detach(), dim=-1).squeeze(-1) # Get GCN prediction
                self.oracle._call_counter += 1

        # self.f_v = torch.tensor(self.oracle.predict(instance), dtype=torch.long) # Get GCN prediction
        self.g_v = self.f_v # CF predicted class
        noise_std = 0 # 1e-10 # Initialization of P_hat with noise to break symmetry
        N = instance.data.shape[0]

        if self.debugging: print(f"{Color.YELLOW}Initial prediction (f_v): {Color.RESET}{self.f_v}") # Debugging

        # if not self.extended: # Use Base version of the explainer
        #     self.A_v = torch.tensor(instance.data, dtype=torch.float32, device=self.device) # instance.data is the adjacency matrix
        #     self.P_hat = torch.ones_like(self.A_v, requires_grad=False, device=self.device) # + noise_std * torch.randn_like(self.A_v) # Initialization of P_hat
            
        elif self.extended or True: # Use extended versione of the explainer (i.e. inverting the roles of A_v and P)
            # self.A_v = torch.ones(instance.data.shape, dtype=torch.float32) # Assume adjacency matrix full of ones
            
            # self.A_v = torch.ones(int(N * (N+1) / 2), device=self.device)

            # missing_edges = np.where(instance.data == 0) # Missing A_v edges
            # missing_edges_triu = torch.tensor(instance.data == 0).triu().nonzero().t()
            # missing_edges_vec = (missing_edges_triu[0] * N + missing_edges_triu[1] - missing_edges_triu[0] * (missing_edges_triu[0]+1) / 2).int()
            # missing_nodefeatures = np.where(instance.node_features == 0) # Missing node_features edges        
            
            # self.P_hat = torch.tensor(instance.data, requires_grad=False) # Initialization of P_hat: Perturbation Matrix for Edges
            # self.P_init = torch.ones_like(self.A_v, requires_grad=False, dtype=torch.float32, device=self.device) # Initialization of P_hat: Perturbation Matrix for Edges

            # self.P_init[missing_edges] = self.γ_edges # (Adjacency matrix is full of ones) P stores the zero edges (their value is γ_edges)
            # self.P_init[missing_edges_vec] = self.γ_edges # (Adjacency matrix is full of ones) P stores the zero edges (their value is γ_edges)

            # self.P_init += noise_std * torch.randn_like(self.A_v)
            # self.P_init.requires_grad_(False) # Disable backpropagation
            # self.P_hat += noise_std * torch.rand_like(self.A_v)

            # self.A_v = torch.ones(instance.data.shape, dtype=torch.float32, device=self.device) # Assume adjacency matrix full of ones
            self.A_v = torch.tensor(instance.data, dtype=torch.float32, device=self.device) # instance.data is the adjacency matrix

            if self.debugging: print(f"P_init == instance.data: {torch.equal(self.P_init, torch.tensor(instance.data))}") # Debugging: γ_edges != 0 ←→ it prints False

            #region Node Features perturbation init
            if self.update_node_feat: # (Two branches initialize identically)
                if not self.change_node_feat: # Enable only discarding or adding of node features (node feature gating)
                    self.P_node_hat = (- torch.ones(instance.node_features.shape, requires_grad=False, device=self.device)).requires_grad_(True) # Initialization of P_node_hat: Perturbation Matrix for Node Features
                elif self.change_node_feat: # Allow node features to change features
                    # self.P_node_hat = torch.tensor(instance.node_features, requires_grad=False) # Initialization Ⅰ of P_node_hat: Perturbation Matrix for Node Features. Init as node_features.
                    # self.P_node_hat[missing_nodefeatures] = self.γ_node_feat # Add node features where they are missing (0)
                    # self.P_node_hat.requires_grad_(True) # Enable backpropagation
                    self.P_node_hat = torch.zeros(instance.node_features.shape, requires_grad=True, device=self.device) # Initialization Ⅱ of P_node_hat: Perturbation Matrix for Node Features. Init as ones.

            # # if self.debugging: print(f"P_init: {self.P_init}")
            # self.P_sym = torch.zeros(instance.data.shape, dtype=torch.float32, requires_grad=False, device=self.device)
            # i, j = np.triu_indices(N)
            # self.P_sym[i,j] = self.P_init
            # if self.directed:
            #     self.P_hat = self.P_sym
            # else:
            #     self.P_hat = self.P_sym + self.P_sym.t() - torch.diag(torch.diag(self.P_sym)) # Symmetrizing P_hat

        # # P_triu = torch.triu(self.P_init, diagonal=0)
        # # self.P_hat = 0.5 * (self.P_init + self.P_init.t()) # Symmetrizing P_hat
        # self.P_hat = torch.nn.Parameter(self.P_hat.clone())
        # self.P_hat.requires_grad_(True) # Enable backpropagation
        # # if self.debugging: print(f"P_hat symmetrized: {self.P_hat}")

        #region Adjacency Perturbation init
        self.P_hat = (- torch.ones_like(self.A_v, requires_grad=False, dtype=torch.float32, device=self.device)).requires_grad_(True) # make it a leaf tensor

        # Node features
        self.x = torch.tensor(instance.node_features, dtype=torch.float32, device=self.device) # Feature vector for v [3.1]
        self.adj = torch.tensor(instance.data, dtype=torch.float32, device=self.device)
        self.v = (self.adj, self.x) # [3.1]

        # Edge features
        self.e = torch.tensor(instance.edge_features, dtype=torch.float32) # Edge features vector
        # Creating self.edge_features to store features for all possible edges. If edge is not present, its features are set to 1 for all dimensions.
        self.feat_dim = instance.edge_features.shape[1] # Extracting edge features dimansionality
        self.edge_features = torch.zeros(instance.data.shape + (self.feat_dim,), dtype=torch.float32, requires_grad=False) # Edge features matrix for all possible edges (full of ones in each dimension)
        self.edge_indices = torch.where(torch.tensor(instance.data) != 0) # Indices of existing edges in Adjacency Matrix (integer tensor) 
        self.edge_features[self.edge_indices] = self.e # Assigning existing edge features
        self.edge_features = self.edge_features.squeeze(-1)

        # Graph features 
        # self.g = torch.tensor(instance.graph_features, dtype=torch.float32) # Graph features
        
        #region Edge Features Perturbation init
        if self.change_all_feat: # Allow all features to change freely (node, edge, graph features)
            # self.P_edge_hat = torch.ones(instance.edge_features.shape, requires_grad=True) # Initialization of P_edge_hat: Perturbation Matrix for Edge Features
            self.P_edge_hat = torch.ones(instance.data.shape + (self.feat_dim, ), dtype=torch.float32, requires_grad=True) # Initialization of P_edge_hat: Perturbation Matrix for Edge Features
            # self.P_graph_hat = torch.ones(instance.graph_features.shape, requires_grad=True) # Initialization of P_graph_hat: Perturbation Matrix for Graph Features

        self.v_bar_opt = (self.data, self.x) # Initializing optimal CF with the instance itself

        self.opt_flag = False # CF not found yet
        edge_indices = np.where(instance.data != 0) # Indices of edges (int array)
        self.edge_weights_opt = instance.data[edge_indices] # Optimal CF edge weights (real array)

        if self.visualize: self.pos = nx.spring_layout(nx.from_numpy_array(instance.data)) # Fix graph orientation 

        if self.chem_flag and self.atom_label is not None:
            if "MUTAG" in self.dataset.name:
                MUTAG_VALENCE = {0: 4, 1: 3, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}  # C,N,O,F,I,Cl,Br
                self.max_valence = torch.tensor(
                    [MUTAG_VALENCE.get(int(t), 4) for t in self.atom_label],
                    dtype=torch.float32, device=self.device
                )
            else:
                self.max_valence = None  # skip L_val

        self.n = instance.data.shape[0]
        self.vel = torch.zeros((self.n, self.n), dtype=torch.float32, device=self.P_hat.device)
        self.ema = self.P_hat.data.clone()
        self.α = self.local_config['parameters']['alpha']
        self.lr_reduction_epoch = self.K // self.K #XXX 

        self.num_structural_feat = 0
        if 'Causality' in self.manipulators: self.num_structural_feat += 1
        if 'NodeCentrality' in self.manipulators: self.num_structural_feat += 7

        if self.dataset.name.split('-', 1)[0] in self.CHEMICAL_DATASETS:
            self.atom_label = np.argmax(instance.node_features[:, self.dataset.data.num_node_attributes:instance.node_features.shape[1]-self.num_structural_feat], axis=1).astype(float)  # shape (N,) # NOTE we exclude node attributes/features that are preprended to node labels and structural features that are appended.
            # self.n_atom_types = self.dataset.data.num_node_labels

        for k in (pbar:=tqdm(range(int(self.K)), disable=not (self.visualize or self.debugging))): #XXX
            # pbar.set_description(f"self.alpha: {self.α}")
            if self.debugging: print(f"Iteration: {self.k}")
            self.k = k
            self.temperature = 1
            self.new_CF = False

            self.__get_CF_example(instance) # Compute CF #XXX

            loss = self.__calculate_loss(instance) # Compute Loss #XXX
            # loss = self.g_v_logits[self.f_v]
            loss.backward()

            ...
            # input()
            # from src.utils.torch.gcn import GCNConvWithEdgeTracking
            # edge_messages = [conv.last_messages.abs().sum(dim=1) for conv in self.oracle.model.graph_convs if isinstance(conv, GCNConvWithEdgeTracking)]

            # num_nodes = self.P_hat.shape[0]
            # edge_influence = torch.zeros_like(self.P_hat)
            # for layer_msgs in edge_messages:  # shape: (num_edges,)
            #     src = self.edge_index[0]
            #     dst = self.edge_index[1]
            #     edge_influence[src, dst] += layer_msgs

            μ = 0.9
            β = 0.9
            max_step = 0.1

            # reinitialize momentum + EMA if shape changed
            if not hasattr(self, "vel") or self.vel.shape != (self.n, self.n):
                self.vel = torch.zeros((self.n, self.n), dtype=torch.float32, device=self.P_hat.device)

            if self.update_node_feat and (not hasattr(self, "vel_node") or self.vel_node.shape != self.x.shape):
                self.vel_node = torch.zeros(self.x.shape, dtype=torch.float32, device=self.P_node_hat.device)

            if not hasattr(self, "ema") or self.ema.shape != (self.n, self.n):
                self.ema = self.P_hat.data.clone()

            if self.update_node_feat and (not hasattr(self, "ema_node") or self.ema_node.shape !=self.x.shape):
                self.ema_node = self.P_node_hat.data.clone()

            # input()

            if self.extended:
                g = self.P_hat.grad # NOTE 
                g = g / (g.abs().max() + 1e-8) # NOTE 
                g = g / (g.norm() + 1e-8) # NOTE 
                self.P_hat.grad = g # NOTE 

                # --- MOMENTUM ---
                self.vel = μ * self.vel + (1 - μ) * self.P_hat.grad

                # --- GRADIENT CLIP ---
                # self.vel.data.clamp_(-max_step, max_step)

            if self.update_node_feat and self.P_node_hat.grad is not None:
                if self.num_structural_feat > 0: self.P_node_hat.grad[:, -self.num_structural_feat:] = 0 # Prevents optimization effort for structural features
                g_node = self.P_node_hat.grad
                g_node = g_node / (g_node.abs().max() + 1e-8)
                g_node = g_node / (g_node.norm() + 1e-8)
                self.P_node_hat.grad = g_node
                self.vel_node = μ * self.vel_node + (1 - μ) * self.P_node_hat.grad

            with torch.no_grad():  # Update without tracking the gradients further
                if self.debugging: print(f"P_hat before update: {self.P_hat}")                           

                # present_mask = (self.A_v_bar == 1).to(self.P_hat.grad.dtype)
                # missing_mask = 1.0 - present_mask

                # self.P_hat.grad[self.edge_indices_ones] = self.w.grad.clone().detach().float()

                # parameter update
                if self.extended:
                    self.P_hat.data += self.α * self.vel # NOTE We do gradient ascent since we want to invert prediction

                    # --- EMA SMOOTH ---
                    self.ema = β * self.ema + (1 - β) * self.P_hat.data
                    self.P_hat.data = self.ema.clone()

                    self.P_hat.data = (self.P_hat.data + self.P_hat.data.T) / 2 # Symmetrizing P_hat after update
                    # if instance.id == 2 and (self.k == 0 or self.k == 179):
                    #     print(f"self.P_hat: {self.P_hat}")                        

                elif not self.extended: # CF-GNNExplainer only drops edges
                    missing_mask = torch.tensor(instance.data == 0)
                    self.P_hat.data[missing_mask] = 0

                    # self.P_hat.data.clamp_(-5, 5)

                    # self.P_hat.grad = self.P_hat.grad / (self.P_hat.grad.abs().max() + 1e-8) # normalize # normalize to [-1,1]
                    
                    # self.P_hat.data = β * self.P_hat + (1 - β) * (self.P_hat + self.α * self.P_hat.grad)

                    self.P_hat += self.α * self.P_hat.grad # Gradient update step with learning rate 
                
                if self.debugging: print(f"P_hat.grad: {self.P_hat.grad}"); print(f"P_hat updated: {self.P_hat}"); print(f"self.A_v_bar: {self.A_v_bar}")

                if self.update_node_feat: # self.P_node_hat (nodes features perturbation matrix) exists only in the extended algorithm where node features perturbations are allowed
                    # self.P_node_hat -= self.α * self.P_node_hat.grad # Gradient update step with learning rate
                    self.P_node_hat.data += self.α * self.vel_node
                    self.ema_node = β * self.ema_node + (1 - β) * self.P_node_hat.data
                    self.P_node_hat.data = self.ema_node.clone()
                    if self.debugging: print(f"P_node_hat.grad: {self.P_node_hat.grad}")
            
            if self.P_hat.grad is not None: self.P_hat.grad.zero_() # zero gradients for next iteration
            if self.update_node_feat: # self.P_node_hat exists only in the extended algorithm where node features perturbations are allowed
                self.P_node_hat.grad.zero_() # zero gradients for next iteration

            if self.visualize and self.new_CF: # or (self.A_v_bar.data.numpy() != instance.data).any(): # and self.opt_flag:
                instance_graph = nx.from_numpy_array(instance.data)
                CF_graph = nx.from_numpy_array(self.A_v_bar.clone().detach().cpu().numpy())

                # Draw graphs. Node colours are the mean of the node features
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                nx.draw(instance_graph, pos=self.pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
                axes[0].set_title(f"Initial Graph | Predicted Class: {self.f_v}")
                nx.draw(CF_graph, pos=self.pos, ax=axes[1], with_labels=True, cmap='cool', node_color=self.N_v_bar.clone().detach().cpu().numpy().mean(axis=1), edge_color='gray')
                axes[1].set_title(f"Counterfactual Graph | Predicted Class: {self.g_v} - K: {self.k}")
                fig.suptitle(f"True label: {instance.label}")
                plt.show()
                ...

            if self.debugging and self.visualize: print(f"Iteration {self.k} finished | Press Enter to continue")
            elif self.debugging: input(f"Iteration {self.k} finished | Press Enter to continue")

            if (self.k+1) % self.lr_reduction_epoch == 0 and self.boost_α and self.extended:
                if (self.A_v_bar.data.clone().detach().cpu().numpy() == instance.data).all():
                    self.α *= 1.15
                else:
                    self.α *= 1
                # if self.α > 1e1*8: self.α = 1e1*8
                pbar.set_description(f"Learning rate changed to {Color.YELLOW}{self.α:<.6f}f{Color.RESET} at iteration {Color.YELLOW}{self.k+1}{Color.RESET}/{self.K}")
            
            if self.opt_flag: break # Breaking at first CF found (Different from paper algorithm → If Efficiency is preferred: avoiding extra loop iterations to find better (closer) CF))

        
        # edge_indices = torch.where(self.v_bar_opt[0] != 0) # (int tensor)
        # edge_weights = self.v_bar_opt[0][edge_indices] # (real tensor)
        # edge_weights = edge_weights.clone().detach().cpu().numpy()
        # edge_features = self.edge_features[edge_indices].clone().detach().cpu().numpy() # Edge features for the existing edges (np.array)

        edge_indices = torch.as_tensor(torch.stack(torch.where(self.v_bar_opt[0] != 0)), device=self.device) # (int tensor)
        edge_weights = self.v_bar_opt[0].clone().detach()[edge_indices].to(dtype=torch.float32, device=self.device)

        if not self.opt_flag:
            self.v_bar_opt = (self.A_v_bar, self.v_bar_opt[1])
        edge_indices = torch.where(self.v_bar_opt[0] != 0) # (int tensor)
        edge_weights = self.v_bar_opt[0].detach().clone()[edge_indices[0], edge_indices[1]]
        edge_weights_full = torch.zeros_like(self.v_bar_opt[0])
        edge_weights_full[edge_indices] = edge_weights # weights having also 0s for missing edges
        edge_weights_full = edge_weights_full.flatten().clone().detach().cpu().numpy()

        # edge_indices = (self.v_bar_opt[0] != 0).nonzero(as_tuple=True)[0]
        # edge_features = torch.tensor(instance.edge_features[edge_indices])
        # edge_features_full = torch.zeros(self.v_bar_opt[0].shape + (self.feat_dim,), dtype=torch.float32,)
        # edge_features_full[edge_indices] = edge_features # features having also 0s for missing edges
        # edge_features_full = edge_features_full.flatten().clone().detach().cpu().numpy()

        
        # v_bar_opt_GI = GraphInstance(
        #     id = instance.id,
        #     label = self.g_v_bar_pred, # Predicted class of the CF
        #     data = self.v_bar_opt[0].clone().detach().cpu().numpy(),
        #     node_features = self.v_bar_opt[1].clone().detach().cpu().numpy(),
        #     edge_features = edge_features,
        #     edge_weights = edge_weights,
        #     graph_features = instance.graph_features
        #     )

        #region GraphInstance return
        v_bar_opt_GI = GraphInstance(
            id = instance.id,
            label = self.g_v_bar_pred, # Predicted class of the CF
            data = self.v_bar_opt[0].clone().detach().cpu().numpy(), # self.adj_full.clone().detach().cpu().numpy(),
            node_features = self.v_bar_opt[1].clone().detach().cpu().numpy(),
            # edge_features = edge_features,
            edge_weights = edge_weights.clone().detach().cpu().numpy(), # edge_weights_full,
            graph_features = instance.graph_features,
            iteration = self.k
            )
        
        # from src.evaluation.evaluation_metric_correctness_full import CorrectnessFullMetric
        # CorrectnessFullMetric.evaluate(v_bar_opt_GI)
        # input()

        if self.debugging: print(f"{Color.MAGENTA}Opt predicted class: {Color.RESET}{self.oracle.predict(v_bar_opt_GI)}")
        if self.visualize and not self.opt_flag: # and self.opt_flag:
            instance_graph = nx.from_numpy_array(instance.data)
            CF_graph = nx.from_numpy_array(self.A_v_bar.clone().detach().cpu().numpy())

            # Draw graphs. Node colours are the mean of the node features
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            nx.draw(instance_graph, pos=self.pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
            axes[0].set_title(f"Initial Graph | Predicted Class: {self.f_v}")
            nx.draw(CF_graph, pos=self.pos, ax=axes[1], with_labels=True, cmap='cool', node_color=self.N_v_bar.clone().detach().cpu().numpy().mean(axis=1), edge_color='gray')
            axes[1].set_title(f"Counterfactual Graph | Predicted Class: {self.g_v} - K: {self.k}")
            fig.suptitle(f"True label: {instance.label}")
            plt.show()           

        if self.debugging: print(f"{Color.GREEN}{self.f_v}{Color.RESET} | {Color.MAGENTA}{self.oracle.predict(instance)}{Color.RESET} | {Color.YELLOW}{self.oracle.predict(v_bar_opt_GI)}{Color.RESET} | {self.g_v_bar_pred}")

        # if self.visualize:
        #     save = input("save graph? (y/n): ").lower()
        
        # if self.visualize and save.lower() == "y":
        if self.visualize and input("save graph? (y/n): ").lower() == "y":

            oracle_name = str(self.oracle.model.__class__).split('.')[-2]
            explainer_name = "CFGNNE" if not self.extended else "XPlore"
            G = nx.from_numpy_array(instance.data)

            for i, feat in enumerate(instance.node_features):
                G.nodes[i]["Feature"] = feat.mean()

            nx.write_gexf(G, f"CFs_figs\\{instance.id}-original.gexf")

            G = nx.from_numpy_array(edge_weights_full.reshape(instance.data.shape))
            print(edge_weights_full.reshape(instance.data.shape).shape)

            for i, feat in enumerate(self.N_v_bar.clone().detach().cpu().numpy()):
                G.nodes[i]["Feature"] = feat.mean()

            nx.write_gexf(G, f"CFs_figs\\{instance.id}-{explainer_name}-{oracle_name}.gexf")
            input(f"Graph saved to CFs_figs\\{instance.id}-{explainer_name}-{oracle_name}.gexf")

        if not self.opt_flag:
            if not self.node_classification: print(f"{Color.RED}CF not found{Color.RESET}, original: {Color.MAGENTA}{self.f_v}{Color.RESET}")
            else: print(f"{Color.RED}CF not found{Color.RESET}, original: {Color.MAGENTA}{self.f_v[self.node_id]} [^node id: {self.node_id}]{Color.RESET}")
            return instance

        #region Chemical Validity check
        if self.chem_flag:
            # Chemical validity check
            # Dataset-specific atom type maps
            ATOM_MAPS = {
                "MUTAG": {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"},
                "Mutagenicity": {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br", 7: "S"},
                "AIDS": {0:"C",1:"O",2:"N",3:"Cl",4:"F",5:"S",6:"Se",7:"P",8:"Na",9:"I", 10:"Co",11:"Br",12:"Li",13:"Si",14:"Mg",15:"Cu",16:"As",17:"B",18:"Pt",19:"Ru", 20:"K",21:"Pd",22:"Au",23:"Te",24:"W",25:"Rh",26:"Zn",27:"Bi",28:"Pb",29:"Ge", 30:"Sb",31:"Sn",32:"Ga",33:"Hg",34:"Ho",35:"Tl",36:"Ni",37:"Tb"}
            }
            BOND_MAPS = {}
            BOND_MAPS["MUTAG"] = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
                4: Chem.BondType.AROMATIC
            }
            BOND_MAPS["AIDS"] = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}

            def nx_to_mol(G: nx.Graph, dataset_name: str, node_label_attr="label", edge_label_attr="label"):
                atom_map = ATOM_MAPS.get(dataset_name)
                if atom_map is None:
                    return None  # atom types unknown for this dataset
                mol = Chem.RWMol()
                for node, data in G.nodes(data=True):
                    symbol = atom_map.get(data.get(node_label_attr, 0), "C")
                    mol.AddAtom(Chem.Atom(symbol))
                for u, v, data in G.edges(data=True):
                    if u == v: continue  # skip self-loops
                    bond_type = BOND_MAPS.get(data.get(edge_label_attr, 1), Chem.BondType.SINGLE)
                    mol.AddBond(int(u), int(v), bond_type)
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                    return mol  # valid molecule
                except Exception:
                    return None  # invalid molecule

            def is_valid_molecule(G: nx.Graph, dataset_name: str, **kwargs) -> bool:
                return nx_to_mol(G, dataset_name, **kwargs) is not None

            def get_cf_validity(instance, cf_adj: np.ndarray, dataset_name: str) -> bool:
                """
                Check if a CF adjacency matrix corresponds to a valid molecule.
                Requires instance.atom_label to be set (from TUDataset populate()). # XXX Update: handled per instance with self.atom_label
                Returns False if atom types are unavailable for this dataset.
                """
                if self.atom_label is None:
                    return False
                cf_adj_binary = (cf_adj > 0.5).astype(int)
                G = nx.from_numpy_array(cf_adj_binary)
                # if not nx.is_connected(G):
                    # return False
                for i, atom_idx in enumerate(self.atom_label):
                    G.nodes[i]['label'] = int(atom_idx)
                return is_valid_molecule(G, dataset_name)
            
            def draw_molecule(cf_adj, dataset_name, filepath="molecule.png"):
                mol = nx_to_mol(
                    nx.from_numpy_array(cf_adj), 
                    dataset_name
                )
                if mol is None:
                    print("Invalid molecule, cannot draw.")
                    return
                # Set atom labels from atom_label
                G = nx.from_numpy_array(cf_adj)
                for i, atom_idx in enumerate(self.atom_label):
                    G.nodes[i]['label'] = int(atom_idx)
                mol = nx_to_mol(G, dataset_name)
                Draw.MolToFile(mol, filepath)
                return mol
            
            def plot_molecule(mol, title="", filepath=None, ref_mol=None):
                # Generate 2D coords
                import random
                if ref_mol is not None:
                    try:
                        AllChem.GenerateDepictionMatching2DStructure(mol, ref_mol)
                    except ValueError:
                        AllChem.Compute2DCoords(mol)  # fallback to independent layout
                else:
                    AllChem.Compute2DCoords(mol)
                img = Draw.MolToImage(mol, size=(300, 300))

                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(title)
                if filepath:
                    plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0, transparent=True)
                    print(f"Saved to {filepath}")
                plt.show()
                plt.close()

            def project_to_valid_molecule(mol):
                try:
                    Chem.SanitizeMol(mol)
                    print("Already valid, no changes needed")
                    return mol
                except Exception as e:
                    print(f"Sanitization failed: {e}")
                    try:
                        mol = Chem.RWMol(mol)
                        # Print SMILES before fix
                        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                        return mol
                    except Exception as e2:
                        print(f"Projection failed: {e2}")
                        return None

            if any(dataset in self.dataset.name for dataset in self.CHEMICAL_DATASETS):
                chem_valid = get_cf_validity(instance, self.v_bar_opt[0].detach().cpu().numpy(), self.dataset.name)
                print(f"Chemically valid: {chem_valid}")
                cf_adj = self.v_bar_opt[0].detach().cpu().numpy()
                G = nx.from_numpy_array(cf_adj)
                G_original = nx.from_numpy_array(instance.data)
                if self.atom_label is not None:
                    for i, atom_idx in enumerate(self.atom_label):
                        G.nodes[i]['label'] = int(atom_idx)
                        G_original.nodes[i]['label'] = int(atom_idx)
                # mol = nx_to_mol(G, self.dataset.name)
                mol = nx_to_mol(G, self.dataset.name)
                mol_original = nx_to_mol(G_original, self.dataset.name)
                if mol:
                    print(f"SMILES raw: {Chem.MolToSmiles(mol)}")
                    # plot_molecule(mol, title=f"CF Molecule - Valid: {chem_valid}")
                else:
                    print("Pre-projection: invalid molecule")
                
                # Project and visualize
                # mol_projected = project_to_valid_molecule(Chem.RWMol(mol)) if mol else None
                # if mol_projected:
                #     print(f"SMILES projected: {Chem.MolToSmiles(mol_projected)}")
                #     plot_molecule(mol_projected, title="", filepath=f"")NH
                # else:
                #     print("Post-projection: still invalid")
                
                if mol:
                    oracle_name = str(self.oracle.model.__class__).split('.')[-2]
                    explainer_name = "CFGNNE" if not self.extended else "XPlore"
                    # draw_molecule(mol, "MUTAG", filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-{explainer_name}-{oracle_name}.png")
                    os.makedirs(f"CFs_figs\\{self.dataset.name}", exist_ok=True)
                    if input("save graph? (y/n): ").lower() == "y":
                        plot_molecule(mol, filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-{explainer_name}-{oracle_name}.pdf", ref_mol=mol_original)
                        nx.write_gexf(G, f"CFs_figs\\{self.dataset.name}\\{instance.id}-{explainer_name}-{oracle_name}.gexf")
                        
                        plot_molecule(mol_original, filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-original.pdf", ref_mol=mol_original)
                        nx.write_gexf(G_original, f"CFs_figs\\{self.dataset.name}\\{instance.id}-original.gexf")
                        input(f"Graph saved to CFs_figs\\{self.dataset.name}\\{instance.id}-{explainer_name}-{oracle_name}.gexf")

        return v_bar_opt_GI

    #region get CF example
    def __get_CF_example(self, instance):
        """
        Compute the CF for ```instance```.
        Algorithm (1) Xplore [3.3].
        lines in brackes '[line 0]' refer to lines of pseudo code in the paper.
        """
        # [line 1]: P ← threshold(σ(P_hat))
        # self.temperature *= self.temperature
        P_sigmoid = torch.sigmoid(self.P_hat / self.temperature) # Threshold on sigmoid of P_hat
        threshold = 0.5 # if self.extended else .5 # NOTE This equals CF-GNNExplainer using a threshold of 0.5 on original P_hat
        # STE
        mask = (P_sigmoid >= threshold).float() # Hard mask
        # P = P_sigmoid + (mask - P_sigmoid).clone().detach() # Gradients can flow through P
        P = mask + (P_sigmoid - P_sigmoid.clone().detach()) # Gradients can flow through P

        # EDGE FLIP GATE - Eq. (4)
        # self.A_v_bar = P * self.A_v # [line 2]: Ā_v = P ⊙ A_v
        self.A_v_bar = self.A_v + P * (1 - 2 * self.A_v)

        if not self.extended and not (self.update_node_feat or self.change_node_feat):
            self.A_v_bar.fill_diagonal_(1) # Add self-loops Eq(4) [5.2]

        # self.edge_index = self.A_v_bar.nonzero(as_tuple=False).T
        # self.edge_weights = self.A_v_bar[self.edge_index[0], self.edge_index[1]]
        # or equivalently
        # self.edge_indices = torch.where(self.A_v_bar != 0) # (integer tensor)
        # self.edge_weights = self.A_v_bar[self.edge_indices] # Values of A_v_bar edges, i.e. weights for the presence of edges (real tensor)
        # self.edge_index = torch.nonzero(self.A_v_bar).int().T

        # i,j = torch.triu_indices(self.n, self.n, offset=0, device=self.device)
        # self.full_edge_index = torch.stack([i, j], dim=0)  # 2 x M
        # self.full_edge_index_rev = torch.stack([j, i], dim=0)
        # self.edge_index = torch.cat([self.full_edge_index, self.full_edge_index_rev], dim=1) # make it symmetric
        # # edge_weights = P_sigmoid[i,j]
        # ei = self.edge_index[0]
        # ej = self.edge_index[1]
        # self.edge_weights = self.A_v_bar[ei, ej]

        self.edge_index = torch.ones_like(self.A_v).nonzero(as_tuple=False).T
        self.edge_weights = self.A_v_bar[self.edge_index[0], self.edge_index[1]]

        # self.edge_index = torch.ones_like(self.A_v_bar).nonzero(as_tuple=False).T
        # self.edge_weights = self.A_v_bar[self.edge_index[0], self.edge_index[1]]

        # edge_indices = torch.where(self.A_v != 0) # (int tensor)
        # edge_weights = self.A_v_bar[edge_indices[0], edge_indices[1]]
        # self.adj_full = torch.ones_like(self.A_v_bar)
        # edge_weights_full = torch.zeros_like(self.data) 
        # edge_weights_full[edge_indices] = edge_weights # weights having also 0s for missing edges
        # self.edge_weights_full = edge_weights_full.flatten()
        # self.edge_index_full = self.adj_full.nonzero(as_tuple=False).T

        # # self.edge_index_full = torch.where(torch.ones_like(self.A_v_bar) != 0) # (integer tensor)
        # # self.edge_weights_full = torch.zeros_like(self.A_v_bar) 
        # # self.edge_weights_full[self.edge_indices] = self.edge_weights # weights having also 0s for missing edges
        # # self.edge_weights_full = self.edge_weights_full[self.edge_index_full]

        if d := self.__distance(self.data, self.A_v_bar):
            graph = nx.from_numpy_array(P.clone().detach().cpu().numpy())
            num_nodes = graph.number_of_nodes()
            features_to_stack = []

            # copy original dataset features

            # num_node_labels = self.dataset.data.num_node_labels if getattr(self.dataset, "data", None) is not None else 0

            base_features = (instance.node_features[:, :-(self.num_structural_feat)] if self.num_structural_feat > 0 else instance.node_features)
            # base_features = (instance.node_features[:, :-(num_node_labels+self.num_structural_feat)] if self.num_structural_feat > 0 else instance.node_features)
            features_to_stack.append(base_features)

            # if any(dataset in self.dataset.name for dataset in self.CHEMICAL_DATASETS):
            #     atom_onehot = instance.node_features[:, :instance.n_atom_label]
            #     features_to_stack.insert(0, atom_onehot)  # prepend, before causality

            # Compute Centrality features as lists in node order
            if self.manipulators != [] and 'NodeCentrality' in self.manipulators:
                degree = [d for n, d in sorted(graph.degree())]
                betweenness = [v for n, v in sorted(nx.betweenness_centrality(graph).items())]
                closeness = [v for n, v in sorted(nx.closeness_centrality(graph).items())]
                harmonic = [v for n, v in sorted(nx.harmonic_centrality(graph).items())]
                clustering = [v for n, v in sorted(nx.clustering(graph).items())]
                katz = [v for n, v in sorted(nx.katz_centrality_numpy(graph).items())]
                try: laplacian = list(nx.laplacian_spectrum(graph))[:num_nodes]  # take first n eigenvalues if needed
                # except: laplacian = list(nx.laplacian_centrality(graph).values())
                except: laplacian = [0.0] * num_nodes
                centralities = np.stack([degree, betweenness, closeness, harmonic, clustering, katz, laplacian], axis=1)
                # node_features = np.stack([degree, betweenness, closeness, harmonic, clustering, katz, laplacian], axis=1) # Stack into a (num_nodes, num_features) array
                
             # compute Causality features
            if self.manipulators != [] and 'Causality' in self.manipulators:
                causality_manip = next((m for m in self.dataset.manipulators if type(m).__name__ == 'Causality'), None)
                # input(causality_manip)
                if causality_manip:
                    # dim = causality_manip.causality_dim_choice
                    u = int(causality_manip.causalities[instance.id])
                    noise_1 = (causality_manip.max_1[u] - causality_manip.min_1[u]) * np.random.random_sample() + causality_manip.min_1[u]
                    causality_val = noise_1 + 0.5 * np.mean(degree)
                    causality = np.full((num_nodes, 1), causality_val)  # (N, dim) not (N,)
                    features_to_stack.append(causality)

            if self.manipulators != [] and 'NodeCentrality' in self.manipulators:
                features_to_stack.append(centralities)
            
            node_features = np.hstack(features_to_stack)
            self.N_v_bar = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        else: # Keep same node features
            self.N_v_bar = self.x.clone().detach() # Feature vector for v [3.1]
            self.N_v_bar.requires_grad_(False) # Disable backpropagation
 
        if self.update_node_feat: # Perturb node features
            if not self.change_node_feat: # Either discard or maintain the node feature (gating)
                # Repeat previous steps for N_v_bar
                N_sigmoid = torch.sigmoid(self.P_node_hat) # Threshold on sigmoid of P_node_hat
                # N_sigmoid[:, -self.num_structural_feat:] = 1.0 # Structural features unchanged #XXX inplace operation cause issues with autograd
                N_sigmoid = torch.cat([N_sigmoid[:, :-self.num_structural_feat], torch.zeros_like(N_sigmoid[:, -self.num_structural_feat:])], dim=1) if self.num_structural_feat > 0 else N_sigmoid
                # STE
                mask = (N_sigmoid > threshold).float().clone() # Hard mask (> instead of >=)
                N = N_sigmoid + (mask - N_sigmoid).detach() # Gradients can flow through N

                # NODE FEATURE FLIP GATE
                self.N_v_bar = N * (torch.ones_like(self.x) - 2 * self.x) + self.x # N ⊙ x
            else: # Allow node features to change freely
                N_hat = self.P_node_hat.clone()
                # N_sigmoid[:, -self.num_structural_feat:] = 1.0 # Structural features unchanged #XXX inplace operation cause issues with autograd
                N_hat = torch.cat([N_hat[:, :-self.num_structural_feat], torch.zeros_like(N_hat[:, -self.num_structural_feat:])], dim=1) if self.num_structural_feat > 0 else N_hat
                # self.N_v_bar = self.P_node_hat # For initialization Ⅰ
                self.N_v_bar = N_hat + self.x # N_hat + x (For initialization)
        # else: # Keep same node features
            # self.N_v_bar = self.x.clone()
            # self.N_v_bar = self.x.clone().detach() # Feature vector for v [3.1]
            # self.N_v_bar.requires_grad_(False) # Disable backpropagation

        if self.change_all_feat: # Perturb edge and graph features            
            self.E_v_bar = self.P_edge_hat * self.edge_features # E_v_bar = P_edge_hat ⊙ edge_feats
            self.E_v_bar = self.E_v_bar[self.edge_indices] # Take only the edge features for the existing edges
            # self.G_v_bar = self.P_graph_hat * self.g # G_v_bar = P_graph_hat ⊙ graph_feats
        else: # Keep same edge, graph features
            edge_features = self.edge_features[self.edge_indices] # Edge features for the existing edges
            self.E_v_bar = edge_features
            self.E_v_bar.requires_grad_(False) # Disable backpropagation

            # self.E_v_bar_full = self.edge_features
            # self.E_v_bar_full = self.E_v_bar_full[self.edge_index_full]
            # self.E_v_bar_full.requires_grad_(False) # Disable backpropagation


            # self.G_v_bar = torch.tensor(instance.graph_features, dtype=torch.float32)
            # self.G_v_bar.requires_grad_(False) # Disable backpropagation
        
        v_bar_cand = (self.A_v_bar, self.N_v_bar)   
        
        # Computing f(v_bar_cand)
        if self.diffusion_flag:
            self.g_v_logits = self.predict_diffusion(self.N_v_bar, self.edge_index, self.edge_weights)
        else:
            # self.edge_indices_full = torch.nonzero(torch.ones_like(self.A_v_bar)).int().T.requires_grad_(True)

            # self.g_v_logits = self.oracle.model(self.N_v_bar, self.edge_index, self.edge_weights, None).cpu().squeeze()

            # self.g_v_logits = self.oracle.model(self.N_v_bar, self.edge_index_full, self.edge_weights_full, self.batch).squeeze()
            self.g_v_logits = self.oracle.model(self.N_v_bar, self.edge_index, self.edge_weights, self.batch).squeeze()
            # input("logits computed")


            # self.g_v_logits = self._real_predict_gradients(self.A_v_bar_GI) # Oracle CF prediction → returns probabilities
            # self.g_v_logits = self._real_predict_gradients(temp_A_v_bar_GI) # Oracle CF prediction → returns probabilities
        self.oracle._call_counter += 1

        g_v_bar_pred = torch.argmax(self.g_v_logits, dim=-1) # Getting the predicted class
        # if torch.any(g_v_bar_pred != self.oracle.predict(A_v_bar_GI)): input(f"{Color.RED}Warning{Color.RESET} | Oracle's prediction is ambiguous")
        self.g_v_bar_pred = g_v_bar_pred

        # input("Predictions computed")

        self.valid_CF = False # Flag for valid CF
        with torch.no_grad():
            # [line 4]: if f(v) ≠ f(v_bar_cand) (valid CF: initial prediction is different from current one)
            if not self.node_classification and torch.all(self.f_v != g_v_bar_pred) or\
            self.node_classification and self.f_v[self.node_id]!= g_v_bar_pred[self.node_id]:
                self.valid_CF = True # Valid CF is found
                v_bar = v_bar_cand # [line 5]: v_bar ← v_bar_cand

                if not self.opt_flag: # [line 6]: if not v_bar_opt then
                    self.v_bar_opt = v_bar # [line 7]: v_bar_opt ← v_bar # First CF
                    if not self.node_classification: self.context.logger.info(f"{Color.GREEN}Found valid counterfactual - {Color.CYAN}Counterfactual predicted class: {Color.GREEN}{g_v_bar_pred}{Color.CYAN} instead of {Color.MAGENTA}{self.f_v}{Color.RESET} | K: {self.k}")  # Debugging
                    else: self.context.logger.info(f"{Color.GREEN}Found valid counterfactual [^node id:{self.node_id}] - {Color.CYAN}Counterfactual predicted class: {Color.GREEN}{g_v_bar_pred[self.node_id]}{Color.CYAN} instead of {Color.MAGENTA}{self.f_v[self.node_id]}{Color.RESET} | K: {self.k}")  # Debugging
                    self.edge_weights_opt = self.edge_weights
                    self.opt_flag = True # CF found
                    self.g_v = g_v_bar_pred
                    self.new_CF = True

                # Check for 'closer' CF, i.e. CF that requires less perturbations
                elif self.__distance(self.v[0], v_bar[0]) < self.__distance(self.v[0], self.v_bar_opt[0]): # [line 8]: else if d(v, v_bar) ≤ d(v, v_bar*) then
                    self.v_bar_opt = v_bar # [line 9]: v_bar* ← v_bar # Keep track of best CF
                    self.edge_weights_opt = self.edge_weights
                    if not self.node_classification: self.context.logger.info(f"{Color.BLUE}Found new best counterfactual - {Color.CYAN}Counterfactual predicted class: {Color.RESET}{g_v_bar_pred} | K: {self.k}")  # Debugging
                    else: self.context.logger.info(f"{Color.BLUE}Found new best counterfactual [^node id:{self.node_id}] - {Color.CYAN}Counterfactual predicted class: {Color.RESET}{g_v_bar_pred[self.node_id]} | K: {self.k}")  # Debugging
                    self.g_v = g_v_bar_pred
                    self.new_CF = True

        return

    #region Loss
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
        elif isinstance(self.loss_fn, torch.nn.NLLLoss): # CE is just NNL with log+softmax included
            inputs = F.log_softmax(self.g_v_logits, dim=0) # Prediction log probabilities required for NLL loss
            targets = self.f_v

        if self.node_classification:
            inputs = inputs[self.node_id]
            targets = targets[self.node_id]

        # If f(v) == f(v_bar), the loss is 0; otherwise, compute NLL loss
        if not self.valid_CF: 
            # Use negative log-likelihood loss (NLL) between for predicted logits and ground truth (Loss Function Optimization) [5.3]
            L_pred = -1 * self.loss_fn(inputs, targets) # NOTE here L_pred is the NLL itself (with BCE)
        else:
            L_pred = 0  # No loss if predictions are different

        if isinstance(self.loss_fn, torch.nn.NLLLoss):
            L_pred *= -1 # NOTE NLL is negated later in total_loss

        # 2. Distance loss: L_dist
        # Compute the number of edges removed (element-wise difference between A_v and A_v_bar) and the change in node features
        edge_diff = torch.abs(self.data - self.A_v_bar) # Element-wise absolute difference for edges
        node_diff = torch.abs(torch.tensor(instance.node_features, device=self.device) - self.N_v_bar) / self.feat_std # Element-wise absolute difference for nodes features. Normalized by dataset std
        D_edges = edge_diff.mean() # L1-norm for edges: count the number of edges changed
        D_nodes = node_diff.mean() # L1-norm for nodes: distance between the node features changed
        # D_nodes = 0
        L_dist = D_edges + D_nodes # Total L1-norm


        # Chemical loss
        if self.chem_flag:   
            valence_penalty = self.__valence_penalty() if self.atom_label is not None else 0
            connectivity_penalty = self.__connectivity_penalty()
            total_loss = - L_pred - (self.β * L_dist + 0.1 * valence_penalty + 0.5 * connectivity_penalty) # NOTE here we do - NLL (normal BCE)

        else:
            # 3. Total loss
            # NOTE we want to flip the prediction. By doing gradient ascent we want to maximize the loss on the logits to produce misclassification. Distance is negative since we want to go in the direction of negative distance i.e., closeness.
            total_loss = - L_pred - (self.β * L_dist) # NOTE here we do -NLL (normal BCE)
            # total_loss = - (L_pred_margin - self.β * L_dist)
        
        # return L_pred
        return total_loss
    
    def set_node_id(self, node_id:int):
        """Call it when computing metrics and iterating over the nodes of the graph. Node id needed for the loss."""
        self.node_id = node_id

    @torch.no_grad()
    def __distance(self, v, CF):
        return (v != CF).sum()
    
    def __valence_penalty(self):
        soft_degree = self.A_v_bar.sum(dim=1) - torch.diag(self.A_v_bar)  # P ⊙ A_v, gradients flow - subtract self-loops if present
        violation = torch.relu(soft_degree - self.max_valence)
        return violation.sum()
    
    def __connectivity_penalty(self):
        A = self.A_v_bar
        A = (A + A.T) / 2  # symmetrize
        D = torch.diag(A.sum(dim=1))
        L = D - A
        eigvals = torch.linalg.eigvalsh(L)
        fiedler = eigvals[1]
        return torch.relu(0.5 - fiedler)

    def predict_diffusion(self, node_features, edge_index, edge_weights):
        """Compute prediction using the diffusion oracle for a single instance, keeping gradients."""
        
        # Single graph -> batch tensor of zeros
        batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=self.oracle.device)

        # Initialize noisy labels
        B = 1
        # y_t = torch.randn(B, self.oracle.dataset.num_classes, device=self.oracle.device, dtype=torch.float32)
        y_t = torch.zeros(B, self.oracle.dataset.num_classes, device=self.oracle.device, dtype=torch.float32) # deterministic

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
                # sigma_t = torch.sqrt(betas[t]) # stochastic
                # y_prev = y_prev + sigma_t * torch.randn_like(y_prev, device=self.oracle.device, dtype=y_prev.dtype) # stochastic
                pass
            y_t = y_prev

        # Return logits (keep gradients for explainer)
        return y_t.squeeze()

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