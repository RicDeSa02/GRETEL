from numbers import Number
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseGraphConv
from torch.utils.data import DataLoader

from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.dataset.utils.dataset_torch import TorchDataset
from src.utils.logger import GLogger
from src.utils.utils import pad_adj_matrix

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os
import networkx as nx, matplotlib.pyplot as plt


class CLEARExplainer(Trainable, Explainer):

    def init(self):
        super().init()        
        self.batch_size = self.local_config['parameters']['batch_size']
        self.h_dim = self.local_config['parameters']['h_dim']
        self.z_dim = self.local_config['parameters']['z_dim']
        self.dropout = self.local_config['parameters']['dropout']
        self.encoder_type = self.local_config['parameters']['encoder_type']
        self.graph_pool_type = self.local_config['parameters']['graph_pool_type']
        self.disable_u = self.local_config['parameters']['disable_u']
        self.epochs = self.local_config['parameters']['epochs']
        self.alpha = self.local_config['parameters']['alpha']
        self.feature_dim = self.local_config['parameters']['feature_dim']
        self.lr = self.local_config['parameters']['lr']
        self.weight_decay = self.local_config['parameters']['weight_decay']
        self.lambda_sim = self.local_config['parameters']['lambda_sim']
        self.lambda_kl = self.local_config['parameters']['lambda_kl']
        self.lambda_cfe = self.local_config['parameters']['lambda_cfe']
        self.beta_x = self.local_config['parameters']['beta_x']
        self.beta_adj = self.local_config['parameters']['beta_adj']
        self.n_nodes = self.local_config['parameters']['n_nodes']
        self.multiclass = self.local_config['parameters']['multiclass']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = CLEAR(feature_dim=self.feature_dim,
                           graph_pool_type=self.graph_pool_type,
                           encoder_type=self.encoder_type,
                           n_nodes=self.n_nodes,
                           h_dim=self.h_dim,
                           z_dim=self.z_dim,
                           dropout=self.dropout,
                           disable_u=self.disable_u,
                           device=self.device).to(self.device)
                        
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        self._logger = GLogger.getLogger()
        self.visualize = self.local_config['parameters'].get('visualize', False)
        self.vis_id = self.local_config['parameters'].get('vis_id', -1)
        self.chem_flag = self.local_config['parameters'].get('chem_flag', False)    

        
    def explain(self, instance):
        if self.visualize and instance.id != self.vis_id and self.vis_id != -1:
            return instance

        original_instance = instance#.__deepcopy__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure the model is on the same device
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            # pad the adjacency matrix of the current instance
            padded_adj = pad_adj_matrix(instance.data, self.n_nodes)
            # create a new instance
            new_instance = GraphInstance(id=instance.id,
                                        label=instance.label,
                                        data=padded_adj,
                                        dataset=instance._dataset)
            # redo the manipulators
            instance._dataset.manipulate(new_instance)
            # get the features, adj matrix, graph causality and label
            features = torch.from_numpy(np.array(new_instance.node_features)).float().to(self.device)[None,:,:]
            adj = torch.from_numpy(new_instance.data).float().to(self.device)[None,:,:]
            # causality = torch.from_numpy(np.array(new_instance.graph_features[self.dataset.graph_features_map["graph_causality"]])).float().to(self.device)[None,:]
            
            graph_features_map = instance._dataset.graph_features_map
            # print(graph_features_map)
            # print("- -  -")
            # print(graph_features_map["graph_causality"])
            # print("- -  -")
            # print(torch.from_numpy(np.array(new_instance.graph_features[0, self.dataset.graph_features_map["graph_causality"]])))
            if "graph_causality" in graph_features_map:
                if graph_features_map["graph_causality"] == 1:
                    causality = torch.from_numpy(np.array([new_instance.graph_features[0, self.dataset.graph_features_map["graph_causality"]]])).float().to(self.device)[None,:]
                else:
                    causality = torch.from_numpy(np.array(new_instance.graph_features[self.dataset.graph_features_map["graph_causality"]])).float().to(self.device)[None,:]
            else:
                # print("Warning: 'graph_causality' not found in graph_features_map. Using default value.")
                causality = torch.zeros((1, 1)).to(self.device)

            labels = torch.from_numpy(np.array([new_instance.label])).to(self.device)[None,:]
            # Move tensors to the same device
            features = features.to(device)
            causality = causality.to(device)
            adj = adj.to(device)
            labels = labels.to(device).float()
            
            model_return = self.model(features, causality, adj, labels)
            adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']
            
            print("adj stats:",
                adj_reconst.min().item(),
                adj_reconst.max().item(),
                torch.isnan(adj_reconst).any().item())
            adj_reconst_binary = torch.bernoulli(adj_reconst.squeeze())
            
            cf_instance = GraphInstance(id=instance.id,
                                        label=instance.label,
                                        data=adj_reconst_binary.to("cpu").detach().numpy(),
                                        node_features=features_reconst.squeeze().to("cpu").detach().numpy())
            
            orig_pred = self.oracle.predict(original_instance)
            cf_pred = self.oracle.predict(cf_instance)
            
            
            if cf_instance and self.visualize:
                pos = nx.spring_layout(nx.from_numpy_array(cf_instance.data)) # Fix graph orientation
                instance_graph = nx.from_numpy_array(original_instance.data)
                CF_graph = nx.from_numpy_array(cf_instance.data)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                nx.draw(instance_graph, pos=pos, ax=axes[0], with_labels=True, cmap='cool', node_color=original_instance.node_features.mean(axis=1), edge_color='gray')
                axes[0].set_title(f"Initial Graph | Predicted Class: {orig_pred}")
                nx.draw(CF_graph, pos=pos, ax=axes[1], with_labels=True, cmap='cool', node_color=cf_instance.node_features.mean(axis=1), edge_color='gray')
                axes[1].set_title(f"Counterfactual Graph | Predicted Class: {cf_pred}")
                fig.suptitle(f"True label: {original_instance.label}")
                plt.show()

                save = input("save graph? (y/n): ")
            
                if save.lower() == "y":

                    oracle_name = str(self.oracle.model.__class__).split('.')[-2]

                    G = nx.from_numpy_array(cf_instance.data)

                    for i, feat in enumerate(cf_instance.node_features):
                        # print(feat.mean())
                        G.nodes[i]["Feature"] = feat.mean()

                    nx.write_gexf(G, f"CFs_figs\\{instance.id}-CLEAR-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.gexf")
                    input(f"Graph saved to CFs_figs\\{instance.id}-CLEAR-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.gexf")

            if cf_instance and self.chem_flag:
                ATOM_MAPS = {
                    "MUTAG": {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"},
                    "Mutagenicity": {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br", 7: "S"},
                }
                BOND_MAPS = {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                    4: Chem.BondType.AROMATIC
                }

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
                    Requires instance.atom_types to be set (from TUDataset populate()).
                    Returns False if atom types are unavailable for this dataset.
                    """
                    if instance.atom_types is None:
                        return False
                    cf_adj_binary = (cf_adj > 0.5).astype(int)
                    G = nx.from_numpy_array(cf_adj_binary)
                    if not nx.is_connected(G):
                        return False
                    for i, atom_idx in enumerate(instance.atom_types):
                        G.nodes[i]['label'] = int(atom_idx)
                    return is_valid_molecule(G, dataset_name)
                
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
                
                print(f"orig_pred: {orig_pred} | cf_pred: {cf_pred}")
                chem_valid = get_cf_validity(instance, cf_instance.data, "MUTAG")
                print(f"Chemically valid: {chem_valid}")
                cf_adj = cf_instance.data
                G = nx.from_numpy_array(cf_adj)
                G_original = nx.from_numpy_array(instance.data)
                if instance.atom_types is not None:
                    for i, atom_idx in enumerate(instance.atom_types):
                        G.nodes[i]['label'] = int(atom_idx)
                        G_original.nodes[i]['label'] = int(atom_idx)
                        
                mol = nx_to_mol(G, "MUTAG")
                mol_original = nx_to_mol(G_original, "MUTAG")
                
                if mol:
                    oracle_name = str(self.oracle.model.__class__).split('.')[-2]
                    # draw_molecule(mol, "MUTAG", filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-{explainer_name}-{oracle_name}.png")
                    os.makedirs(f"CFs_figs\\{self.dataset.name}", exist_ok=True)
                    if input("save graph? (y/n): ").lower() == "y":
                        plot_molecule(mol, filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-CLEAR-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.pdf", ref_mol=mol_original)
                        nx.write_gexf(G, f"CFs_figs\\{self.dataset.name}\\{instance.id}-CLEAR-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.gexf")
                        input(f"Graph saved to CFs_figs\\{self.dataset.name}\\{instance.id}-CLEAR-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.gexf")

            return cf_instance

    def real_fit(self):
        train_loader = DataLoader(
            self.dataset.get_torch_instances(fold_id=self.fold_id,
                                             dataset_kls='src.explainer.generative.clear.CLEARDataset',
                                             max_nodes=self.n_nodes), 
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            
            batch_num = 0
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0, 0, 0, 0, 0
            for adj, features, labels, causality in train_loader:
                batch_num += 1
                # send the tensors to the device chosen
                features = features.float().to(self.device)
                causality = causality.float().to(self.device)
                adj = adj.float().to(self.device)
                if self.multiclass:
                    labels = labels.float().to(self.device)[:,None] # Multi-class classification
                else:
                    labels = (1 - labels.float()).to(self.device)[:,None]
                ########################################################
                self.optimizer.zero_grad()
                # forward pass
                retr = self.model(features, causality, adj, labels)
                # z_cf
                z_mu_cf, z_logvar_cf = self.model.encoder(
                    retr['features_reconst'], 
                    causality, 
                    retr['adj_reconst'], 
                    labels)
                # compute loss
                loss_params = {
                    'model': self.model,
                    'oracle': self.oracle,
                    'adj_input': adj,
                    'features_input': features,
                    'y_cf': labels,
                    'z_mu_cf': z_mu_cf,
                    'z_logvar_cf': z_logvar_cf
                }
                loss_params.update(retr)
                
                loss_results = self.__compute_loss(loss_params)
                loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'],\
                    loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
                    
                loss += loss_batch
                loss_kl += loss_kl_batch
                loss_sim += loss_sim_batch
                loss_cfe += loss_cfe_batch
                loss_kl_cf += loss_kl_batch_cf
                
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl / batch_num, loss_sim / batch_num, loss_cfe / batch_num, loss_kl_cf / batch_num
            
            self.context.logger.info(f'Epoch {epoch+1} ---> loss {loss}')
            # backward
            alpha = self.alpha if epoch >= 5 else 0
            # ((loss_sim + loss_kl + alpha * loss_cfe) / batch_num).backward() 
            loss = self.lambda_sim * loss_sim + self.lambda_kl * loss_kl + self.lambda_cfe * loss_cfe
            loss.backward()       
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if not torch.isfinite(loss):
                print("Loss exploded")
            self.optimizer.step()
        
        self.model._fitted = True
        
    def __compute_loss(self, params):
        _, oracle, z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
            _, _, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf = params['model'], params['oracle'], params['z_mu'], \
                params['z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params['features_reconst'], \
                    params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], params['z_mu_cf'], params['z_logvar_cf']

        adj_reconst = torch.nan_to_num(adj_reconst, nan=0.0, posinf=0.0, neginf=0.0)         
        features_reconst = torch.nan_to_num(features_reconst, nan=0.0, posinf=0.0, neginf=0.0)
        # kl loss
        z_logvar = torch.clamp(z_logvar, -10, 10)
        z_u_logvar = torch.clamp(z_u_logvar, -10, 10)
        loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
        loss_kl = torch.mean(loss_kl)
        
        # similarity loss
        size = len(features_permuted)
        dist_x = torch.mean(self.__distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
        # adj_permuted /= torch.max(adj_permuted)
        adj_permuted /= (adj_permuted.max(dim=-1, keepdim=True)[0].clamp(min=1e-8))
        adj_permuted = adj_permuted.clamp(0, 1)
        # print("adj min/max:", adj_reconst.min().item(), adj_reconst.max().item())
        # print("features finite:", torch.isfinite(features_reconst).all().item())
        # print("adj finite:", torch.isfinite(adj_reconst).all().item())
        dist_a = self.__distance_graph_prob(adj_permuted, adj_reconst)
                
        loss_sim = self.beta_x * dist_x + self.beta_adj * dist_a
        
        # CFE loss
        y_pred = []
        for i in range(len(adj_reconst)):
            temp_instance = GraphInstance(id=-1,
                                          label=None,
                                          data=adj_reconst[i].to("cpu").detach().numpy().squeeze(),
                                          node_features=features_reconst[i].to("cpu").detach().numpy().squeeze())
            y_pred.append(np.array(oracle.predict_proba(temp_instance)))
            # assert np.isfinite(y_pred).all(), "y_pred contains non-finite values"

        y_pred = torch.from_numpy(np.array(y_pred)).float().squeeze()
        # print(f"y_pred.shape: {y_pred.shape} | y_cf.shape: {y_cf.shape}")
        # print(f"y_pred[0]: {y_pred[0]} |\ny_cf: {-y_cf.to("cpu").view(-1).long()}")        # print(F.one_hot(y_cf.to("cpu").view(-1).long(), 10))        
        loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.to("cpu").view(-1).long())
        
        # rep loss
        if z_mu_cf is None:
            loss_kl_cf = 0
        else:
            loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
            loss_kl_cf = torch.mean(loss_kl_cf)
            
        loss = self.lambda_sim * loss_sim + self.lambda_kl * loss_kl + self.lambda_cfe * loss_cfe

        loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe, 'loss_kl_cf':loss_kl_cf}
        return loss_results  
    
    
    def __distance_feature(self, feat_1, feat_2):
        pdist = nn.PairwiseDistance(p=2)
        return pdist(feat_1, feat_2) / 4
    
    def __distance_graph_prob(self, adj_1, adj_2_prob):
        # if not ((adj_1 >= -1).all() and (adj_1 <= 1 ).all() and (adj_2_prob > -1).all() and (adj_2_prob < 1 ).all()):
            # ...

        # return F.binary_cross_entropy(adj_2_prob, adj_1)
        return F.binary_cross_entropy_with_logits(adj_2_prob, adj_1)
    
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['batch_size'] =  self.local_config['parameters'].get('batch_size', 8)
        self.local_config['parameters']['h_dim'] =  self.local_config['parameters'].get('h_dim', 10)
        self.local_config['parameters']['z_dim'] =  self.local_config['parameters'].get('z_dim', 10)
        self.local_config['parameters']['dropout'] =  self.local_config['parameters'].get('dropout', .1)
        self.local_config['parameters']['encoder_type'] =  self.local_config['parameters'].get('encoder_type', 'gcn')
        self.local_config['parameters']['graph_pool_type'] =  self.local_config['parameters'].get('graph_pool_type', 'mean')
        self.local_config['parameters']['disable_u'] =  self.local_config['parameters'].get('disable_u', False)
        self.local_config['parameters']['epochs'] =  self.local_config['parameters'].get('epochs', 200)
        self.local_config['parameters']['alpha'] =  self.local_config['parameters'].get('alpha', 5)
        self.local_config['parameters']['lr'] =  self.local_config['parameters'].get('lr', 1e-3)
        self.local_config['parameters']['weight_decay'] =  self.local_config['parameters'].get('weight_decay', 1e-5)
        self.local_config['parameters']['lambda_sim'] =  self.local_config['parameters'].get('lambda_sim', 1)
        self.local_config['parameters']['lambda_kl'] =  self.local_config['parameters'].get('lambda_kl', 1)
        self.local_config['parameters']['lambda_cfe'] =  self.local_config['parameters'].get('lambda_cfe', 1)
        self.local_config['parameters']['beta_x'] =  self.local_config['parameters'].get('beta_x', 10)
        self.local_config['parameters']['beta_adj'] =  self.local_config['parameters'].get('beta_adj', 10)
        self.local_config['parameters']['multiclass'] = self.local_config['parameters'].get('multiclass', False)

        self.local_config['parameters']['n_nodes'] = np.max(self.dataset.num_nodes_values)

        self.local_config['parameters']['feature_dim'] = len(self.dataset.node_features_map)
    
class CLEAR(nn.Module):

    def __init__(self,
                 feature_dim,
                 h_dim=16,
                 z_dim=16,
                 dropout=False,
                 n_nodes=10,
                 encoder_type='gcn',
                 graph_pool_type='mean',
                 disable_u=False,
                 device='cuda'
                ):
        super(CLEAR, self).__init__()
        
        self.x_dim = feature_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.encoder_type = encoder_type
        self.graph_pool_type = graph_pool_type
        self.disable_u = disable_u
        self.device = device
        self.u_dim = 1 # init_params['u_dim']
        
        if self.disable_u:
            self.u_dim = 0
        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        else:
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)
        
        
        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim,
                              n_layers=1, activation='none', slope=.1,
                              device=self.device)
        
        self.prior_var = nn.Sequential(
            MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1,
                activation='none', slope=.1, device=self.device),
            nn.Sigmoid()
        )
        
        # encoder
        self.encoder_mean = nn.Sequential(
            nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU()
        )
        
        
        self.encoder_var = nn.Sequential(
            nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU(),
            nn.Sigmoid()
        )
                
        # decoder
        self.decoder_x = nn.Sequential(
            nn.Linear(self.z_dim + 1, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.n_nodes * self.x_dim)
        )
        
        in_channels_a = self.z_dim + 1 if self.disable_u else self.z_dim + 2
        self.decoder_a = nn.Sequential(
            nn.Linear(in_channels_a, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.n_nodes * self.n_nodes),
            nn.Sigmoid()
        )
        self.graph_norm = nn.BatchNorm1d(self.h_dim)
        
        
    def encoder(self, features, causality, adj, y_cf):
        # Q(Z | X, causality, A, Y^CF)
        # input: x, causality, A, y^cf
        # output: z
        graph_rep = self.graph_model(features, adj) # n x num_node x h_dim
        graph_rep  = self.graph_pooling(graph_rep, self.graph_pool_type) # n x h_dim
        graph_rep = self.graph_norm(graph_rep)
        
        if self.disable_u:
            z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
        else:
            z_mu = self.encoder_mean(torch.cat((graph_rep, causality, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, causality, y_cf), dim=1))
            
        return z_mu, z_logvar
    
    def decoder(self, z, y_cf, causality):
        if self.disable_u:
            adj_reconst = self.decoder_a(
                torch.cat((z, y_cf), dim=1)
                ).view(-1, self.n_nodes, self.n_nodes)
        else:
            adj_reconst = self.decoder_a(
                torch.cat((z, causality, y_cf), dim=1)
                ).view(-1, self.n_nodes, self.n_nodes)
                            
        features_reconst = self.decoder_x(
            torch.cat((z, y_cf), dim=1)
            ).view(-1, self.n_nodes, self.x_dim)
        
        return features_reconst, adj_reconst
    
    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.mean(x, dim=1, keepdim=False)
        return out
    
    def prior_params(self, causality): # P(Z | causality)
        if self.disable_u:
            z_u_mu = torch.zeros((len(causality), self.h_dim)).to(self.device)
            z_u_logvar = torch.ones((len(causality), self.h_dim)).to(self.device)
        else:
            z_u_logvar = self.prior_var(causality)
            z_u_mu = self.prior_mean(causality)
            
        return z_u_mu, z_u_logvar
    
    
    def reparametrize(self, mu, logvar):
        # compute z = mu + std * epsilon
        if self.training:
            # compute std from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with
            # mean 0 and variance 1
            eps = torch.rand_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
        
    def forward(self, features, causality, adj, y_cf):
        u_onehot = causality
        
        z_u_mu, z_u_logvar = self.prior_params(u_onehot)
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)
        # reparametrize
        z_sample = self.reparametrize(z_mu, z_logvar)
        # decoder
        
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf, u_onehot)
        
        return {
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'adj_permuted': adj,
            'features_permuted': features,
            'adj_reconst': adj_reconst,
            'features_reconst': features_reconst,
            'z_u_mu': z_u_mu,
            'z_u_logvar': z_u_logvar
        }
        
        

class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers,
                 activation='none', slope=.1, device='cuda'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.slope = slope
        self.device = device
        
        
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError(f'Wrong argument type for hidden_dim {hidden_dim}')
        
        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError(f'Wrong argument type for activation {activation}')
        
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError(f'Incorrect activation: {act}')

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
                
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            
        self.fc = nn.ModuleList(_fc_list)
        
        
    @staticmethod
    def xtanh(x, alpha=.1):
        return x.tanh() + alpha * x
    
    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h
    
    
class CLEARDataset(TorchDataset):
    
    def __init__(self, instances: List[GraphInstance], max_nodes=10):
        super(CLEARDataset, self).__init__(instances=instances, max_nodes=max_nodes)
        
    @classmethod
    def to_geometric(self, instance: GraphInstance, label=0):   
        adj, x, label = super().to_geometric(instance, label)



        #### [[0,1]]
        #### indice: 1 -> err
        #### label_0, causality

        #### [[1]]
        #### indice: 0 -> [1]
        #### causality
        ## causality = torch.from_numpy(np.array(instance.graph_features[0,instance._dataset.graph_features_map["graph_causality"]]))
        ## causality = [causality]

        ##### THIS IS A PATCH. TODO: if there is more than a graph feature, it crashes, because the shape is wrong.
        ##### https://github.com/danielegrattarola/spektral/blob/master/spektral/datasets/tudataset.py#L16

        graph_features_map = instance._dataset.graph_features_map

        if "graph_causality" in graph_features_map:
            # print(instance.graph_features)
            # print("- -  -")
            # print(instance._dataset.graph_features_map["graph_causality"])
            # print("- -  -")
            # print(torch.from_numpy(np.array(instance.graph_features[0,instance._dataset.graph_features_map["graph_causality"]])))
            if instance._dataset.graph_features_map["graph_causality"] == 1:
                causality = torch.from_numpy(np.array([instance.graph_features[0,instance._dataset.graph_features_map["graph_causality"]]]))
            else:
                causality = torch.from_numpy(np.array(instance.graph_features[instance._dataset.graph_features_map["graph_causality"]]))
        else:
            # print("Warning: 'graph_causality' not found in graph_features_map. Using default value.")
            causality = torch.zeros(1)

        # causality = torch.from_numpy(np.array(instance.graph_features[instance._dataset.graph_features_map["graph_causality"]]))

        return adj, x, label, causality
    
    # @classmethod
    # def to_geometric(self, instance: GraphInstance, label=0):
    #     adj, x, label = super().to_geometric(instance, label)

    #     graph_features_map = instance._dataset.graph_features_map

    #     if "graph_causality" in graph_features_map:
    #         # 1) grab the raw feature array
    #         gf = instance.graph_features  
    #         # 2) flatten to 1D (so shape (F,))
    #         flat_gf = gf.reshape(-1)
    #         # 3) pick out the causality index
    #         idx = graph_features_map["graph_causality"]
    #         # 4) convert that single value to a torch tensor
    #         causality = torch.tensor(flat_gf[idx], dtype=torch.float32)
    #     else:
    #         causality = torch.zeros(1, dtype=torch.float32)

    #     return adj, x, label, causality
