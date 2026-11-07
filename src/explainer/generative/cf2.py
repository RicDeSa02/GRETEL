import math
import numpy as np
import torch
from copy import deepcopy
from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.core.oracle_base import Oracle
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os
import networkx as nx, matplotlib.pyplot as plt


class CF2Explainer(Trainable, Explainer):

    def init(self):
        self.n_nodes = self.local_config['parameters']['n_nodes']
        self.batch_size_ratio = self.local_config['parameters']['batch_size_ratio']
        self.lr = self.local_config['parameters']['lr']
        self.weight_decay = self.local_config['parameters']['weight_decay']
        self.gamma = self.local_config['parameters']['gamma']
        self.lam = self.local_config['parameters']['lam']
        self.alpha = self.local_config['parameters']['alpha']
        self.epochs = self.local_config['parameters']['epochs']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        self.model = ExplainModelGraph(self.n_nodes).to(self.device)
        self.model._fitted = False

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.visualize = self.local_config['parameters'].get('visualize', False)
        self.vis_id = self.local_config['parameters'].get('vis_id', -1)
        self.chem_flag = self.local_config['parameters'].get('chem_flag', False)

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['batch_size_ratio'] =  self.local_config['parameters'].get('batch_size_ratio', 0.1)
        self.local_config['parameters']['lr'] =  self.local_config['parameters'].get('lr', 1e-3)
        self.local_config['parameters']['weight_decay'] =  self.local_config['parameters'].get('weight_decay', 0)
        self.local_config['parameters']['gamma'] =  self.local_config['parameters'].get('gamma', 1e-4)
        self.local_config['parameters']['lam'] =  self.local_config['parameters'].get('lam', 1e-4)
        self.local_config['parameters']['alpha'] =  self.local_config['parameters'].get('alpha', 1e-4)
        self.local_config['parameters']['epochs'] =  self.local_config['parameters'].get('epochs', 200)

        # fix the number of nodes
        n_nodes = self.local_config['parameters'].get('n_nodes', None)
        if not n_nodes:
            n_nodes = max([x.num_nodes for x in self.dataset.instances])
        self.local_config['parameters']['n_nodes'] = n_nodes

    def real_fit(self):
        self.model.train()

        for epoch in range(self.epochs):         
            losses = list()

            for graph in self.dataset.instances:
                pred1, pred2 = self.model(graph, self.oracle)
                # print(f"pred1: {pred1}, pred2: {pred2}")
                # print(f"type(pred1): {type(pred1)}", pred1)
                # print(f"type(pred2): {type(pred2)}", pred2)
                
                # input()
                loss = self.model.loss(graph,
                                           pred1, pred2,
                                           self.gamma, self.lam,
                                           self.alpha)
                
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                self.optimizer.step()
            self.context.logger.info(f"Epoch {epoch+1} --- loss {np.mean(losses)}")
        
        self.model._fitted = True

    def explain(self, instance : GraphInstance):
        if self.visualize and instance.id != self.vis_id and self.vis_id != -1:
            return instance

        if(not self.model._fitted):
            self.fit()

        # SANITY CHECK: oracle sensitivity
        # base_pred = self.oracle.predict(instance)
        # test_graph = deepcopy(instance)
        # test_graph.data[:] = 0  # remove all edges
        # ablated_pred = self.oracle.predict(test_graph)
        # print(f"[CF2 sanity] base: {base_pred}, no-edge: {ablated_pred}")

        self.model.eval()
        
        with torch.no_grad():
            cf_instance = deepcopy(instance)

            weighted_adj = self.model._rebuild_weighted_adj(instance)
            dvc = weighted_adj.get_device()
            # print(dvc, self.device)
            if dvc == -1 and self.device == 'cpu':
                masked_adj = self.model.get_masked_adj(weighted_adj).numpy()
            else: 
                masked_adj = self.model.get_masked_adj(weighted_adj).cpu().numpy()
            full_masked = self.model.get_masked_adj(weighted_adj).cpu().numpy() # (shape: self.n_nodes × self.n_nodes)
            
            orig_n = instance.num_nodes
            sliced = full_masked[:orig_n, :orig_n]

            # update instance copy from masked_ajd
            # cf_instance.data = masked_adj        

            # new_adj = np.where(masked_adj != 0, 1, 0)
            # # the weights need to be an array of real numbers with
            # # length equal to the number of edges
            # row_indices, col_indices = np.where(masked_adj != 0)
            # weights = masked_adj[row_indices, col_indices]
            new_adj = (sliced != 0).astype(int)
            rows, cols = np.nonzero(sliced)
            weights = sliced[rows, cols]

            cf_instance.data = new_adj
            cf_instance.edge_weights = weights
            # avoid the old nx representation
            cf_instance._nx_repr = None

            orig_pred = self.oracle.predict(instance)
            cf_pred = self.oracle.predict(cf_instance)

            if cf_instance and self.visualize:
                pos = nx.spring_layout(nx.from_numpy_array(instance.data)) # Fix graph orientation
                instance_graph = nx.from_numpy_array(instance.data)
                CF_graph = nx.from_numpy_array(cf_instance.data)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                nx.draw(instance_graph, pos=pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
                axes[0].set_title(f"Initial Graph | Predicted Class: {orig_pred}")
                nx.draw(CF_graph, pos=pos, ax=axes[1], with_labels=True, cmap='cool', node_color=cf_instance.node_features.mean(axis=1), edge_color='gray')
                axes[1].set_title(f"Counterfactual Graph | Predicted Class: {cf_pred}")
                fig.suptitle(f"True label: {instance.label}")
                plt.show()

                save = input("save graph? (y/n): ")
            
                if save.lower() == "y":

                    oracle_name = str(self.oracle.model.__class__).split('.')[-2]

                    G = nx.from_numpy_array(cf_instance.data)

                    for i, feat in enumerate(cf_instance.node_features):
                        # print(feat.mean())
                        G.nodes[i]["Feature"] = feat.mean()

                    nx.write_gexf(G, f"CFs_figs\\{instance.id}-CF2-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.gexf")
                    input(f"Graph saved to CFs_figs\\{instance.id}-CF2-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.gexf")

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
                    G = nx.from_numpy_array(cf_adj)
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
                
                print(f"result.label: {cf_instance.label}")
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
                
                if chem_valid and mol:
                    oracle_name = str(self.oracle.model.__class__).split('.')[-2]
                    # draw_molecule(mol, "MUTAG", filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-{explainer_name}-{oracle_name}.png")
                    os.makedirs(f"CFs_figs\\{self.dataset.name}", exist_ok=True)
                    if input("save graph? (y/n): ").lower() == "y":
                        plot_molecule(mol, filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-CF2-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.pdf", ref_mol=mol_original)
                        nx.write_gexf(G, f"CFs_figs\\{self.dataset.name}\\{instance.id}-CF2-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.gexf")
                        input(f"Graph saved to CFs_figs\\{self.dataset.name}\\{instance.id}-CF2-{oracle_name}-{"cf" if orig_pred!=cf_pred else "not_cf"}.gexf")

			
            return cf_instance


class ExplainModelGraph(torch.nn.Module):
    
    def __init__(self, n_nodes: int):
        super(ExplainModelGraph, self).__init__()

        self.n_nodes = n_nodes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mask = self.build_adj_mask()

    def forward(self, graph : GraphInstance, oracle : Oracle):        
        pred1 = oracle.predict(graph)

        # re-build weighted adjacency matrix
        weighted_adj = self._rebuild_weighted_adj(graph).to(self.device)
        # get the masked_adj
        masked_adj = self.get_masked_adj(weighted_adj).to(self.device)
        # get the new weights as the difference between
        # the weighted adjacency matrix and the masked learned
        new_weights = weighted_adj - masked_adj
        # get only the edges that exist
        row_indices, col_indices = torch.where(new_weights != 0)

        cf_instance = deepcopy(graph)
        if self.device == "cuda":
            cf_instance.edge_weights = new_weights[row_indices, col_indices].detach().cpu().numpy()
        else:
            cf_instance.edge_weights = new_weights[row_indices, col_indices].detach().numpy()
        # avoid old nx representation
        cf_instance._nx_repr = None
        pred2 = oracle.predict(cf_instance)

        pred1 = torch.Tensor([pred1]).float()  # factual
        pred2 = torch.Tensor([pred2]).float()  # counterfactual

        return pred1, pred2

    def build_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.n_nodes, self.n_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.n_nodes + self.n_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_adj(self, weights):
        sym_mask = torch.sigmoid(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        weights = weights.to(self.device)
        sym_mask = sym_mask.to(self.device)
        masked_adj = weights * sym_mask
        return masked_adj

    def loss(self, graph : GraphInstance, pred1, pred2, gam, lam, alp):
        weights = self._rebuild_weighted_adj(graph)
        bpr1 = torch.nn.functional.relu(gam + 0.5 - pred1).to(self.device)  # factual
        bpr2 = torch.nn.functional.relu(gam + pred2 - 0.5).to(self.device)  # counterfactual
        masked_adj = torch.flatten(self.get_masked_adj(weights))
        L1 = torch.linalg.norm(masked_adj, ord=1).to(self.device)
        return L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
    
    
    # todo reimplement this part
    def _rebuild_weighted_adj_old(self, graph):
        weights = np.zeros((self.n_nodes, self.n_nodes))

        u = []
        v = []
        for i, j in zip(*np.nonzero(graph.data)):
            if i < j:
                u.append(i)
                v.append(j)
        #print(graph.edge_weights.shape)
        #print(graph.edge_weights)
        #print(u)
        #print(v)
        weights[u+v,v+u] = graph.edge_weights
        return torch.from_numpy(weights).float()
    
    def _rebuild_weighted_adj(self, graph):
        weights = np.zeros((self.n_nodes, self.n_nodes))

        # Get only upper-triangle edges (i < j) to avoid double-counting
        rows, cols = np.nonzero(graph.data)
        upper_mask = rows < cols
        u = rows[upper_mask]
        v = cols[upper_mask]

        edge_weights = graph.edge_weights

        # If the graph stores weights for all directed edges (both i→j and j→i),
        # we need to sub-select just the upper-triangle ones
        if len(edge_weights) == len(rows):  # weights for every nonzero cell
            edge_weights = edge_weights[upper_mask]

        # Now u, v, edge_weights should all be the same length
        assert len(u) == len(edge_weights), \
            f"Upper-tri edges: {len(u)}, weights: {len(edge_weights)}"

        # Fill both directions (symmetric adjacency)
        weights[u, v] = edge_weights
        weights[v, u] = edge_weights

        return torch.from_numpy(weights).float()