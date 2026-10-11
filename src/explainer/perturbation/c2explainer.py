import copy

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.utils import is_undirected, sort_edge_index, coalesce

from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable


class C2Explainer(Explainer):
    r"""
    Per-graph (graph classification) port of C2Explainer.

    Logic kept identical to the original graph-classification branch:
      - all existing edges are candidates for deletion (edge_index_remain is empty)
      - `n_virtual_nodes` isolated nodes (copies of the first nodes' features) are
        appended and every original node can be connected to them (candidate additions)
      - sigmoid masks + straight-through binarization, Adam on the masks,
        loss = cf_loss + beta * perturb_loss (+ gamma * ent_loss)
      - the best CF is the one with the fewest perturbations seen during optimization

    Removed: node-level branch, FPM (node-level only), PyG ExplainerAlgorithm dependency.
    """
    DEFAULT_COEFFS = {
        'beta': 1,      # perturb loss
        'gamma': 0.1,   # entropy loss
        'EPS': 1e-15,
    }

    def __init__(self, context, local_config):
        Explainer.__init__(self, context, local_config)

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        p['epochs'] = p.get('epochs', 1000)
        p['lr'] = p.get('lr', 0.1)
        p['print_loss'] = p.get('print_loss', False)
        p['subgraph_mode'] = p.get('subgraph_mode', False)
        p['silent_mode'] = p.get('silent_mode', True)
        p['wo_ST'] = p.get('wo_ST', False)
        p['AT_loss'] = p.get('AT_loss', False)
        p['ENT_loss'] = p.get('ENT_loss', False)
        p['desired_y'] = p.get('desired_y', None)
        p['undirected'] = p.get('undirected', True)
        p['n_virtual_nodes'] = p.get('n_virtual_nodes', 3)
        p['beta'] = p.get('beta', 1)
        p['gamma'] = p.get('gamma', 0.1)

    def init(self):
        p = self.local_config['parameters']
        self.epochs = p['epochs']
        self.lr = p['lr']
        self.print_loss = p['print_loss']
        self.subgraph_mode = p['subgraph_mode']
        self.silent_mode = p['silent_mode']
        self.wo_ST = p['wo_ST']
        self.AT_loss = p['AT_loss']
        self.ENT_loss = p['ENT_loss']
        self.desired_y_cfg = p['desired_y'] # user-supplied target class (optional)
        self.undirected = p['undirected']
        self.n_virtual_nodes = p['n_virtual_nodes']
        self.coeffs = dict(self.DEFAULT_COEFFS)
        self.coeffs['beta'] = p['beta']
        self.coeffs['gamma'] = p['gamma']

        self._desired = None
        self.pred_same = None
        self.edge_mask_add = None
        self.edge_mask_delete = None
        self.orig_mask_remain = None
        self.orig_mask_delete = None
        self.orig_mask_add = None

    def real_fit(self):
        # No global training: the optimization is done per instance in explain().
        pass

    def supports(self) -> bool:
        return True

    def explain(self, instance):
        model = self.oracle.model # adapt if your oracle wraps the torch module differently
        device = next(model.parameters()).device
        x, edge_index = self._instance_to_tensors(instance, device)

        was_training = model.training
        model.eval()
        requires = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            best_cf, x_aug = self._explain_graph(model, x, edge_index)
        finally:
            for p, r in zip(model.parameters(), requires):
                p.requires_grad_(r)
            model.train(was_training)

        if best_cf is None:
            return copy.deepcopy(instance)   # GRETEL convention: no CF found -> original instance
        return self._build_cf_instance(instance, best_cf, x_aug)

    # instance <-> tensors
    @staticmethod
    def _instance_to_tensors(instance, device):
        # ADAPT to your instance class (here: dense adjacency in .data, features in .node_features)
        adj = torch.as_tensor(instance.data)
        edge_index = torch.nonzero(adj).T.long().to(device)
        x = torch.as_tensor(instance.node_features).double().to(device)
        return x, edge_index

    @staticmethod
    def _build_cf_instance(instance, cf_edge_index, x_aug):
        # ADAPT to your instance class. The CF lives on the augmented graph
        # (original nodes + virtual nodes), as in the original method.
        n = x_aug.size(0)
        adj = torch.zeros((n, n))
        adj[cf_edge_index[0].cpu(), cf_edge_index[1].cpu()] = 1
        cf = copy.deepcopy(instance)
        cf.data = adj.numpy()
        cf.node_features = x_aug.cpu().numpy()
        if hasattr(cf, 'edge_weights'):
            cf.edge_weights = adj[cf_edge_index[0].cpu(), cf_edge_index[1].cpu()].numpy()
        return cf

    # core
    @staticmethod
    def _predict(model, x, edge_index, edge_weight=None):
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return model(x, edge_index, edge_weight, batch)[0]   # logits of the single graph: [n_classes]

    def _explain_graph(self, model, x, edge_index):
        N = x.size(0)
        total_edges = edge_index.size(1)

        # candidate sets (original graph-classification branch)
        edge_index_remain = edge_index.new_zeros((2, 0))
        edge_index_delete = edge_index

        k = self.n_virtual_nodes
        a = torch.arange(N, device=x.device)
        b = torch.arange(N, N + k, device=x.device)
        x = torch.cat([x, x[torch.arange(k, device=x.device) % N]])   # virtual isolated nodes
        edge_index_add = torch.cartesian_prod(a, b).T.long()

        if self.subgraph_mode:
            edge_index_add = edge_index.new_zeros((2, 0))

        if self.undirected:
            edge_index_delete = self._sort_edge_index(edge_index_delete)
            edge_index_add = self._sort_edge_index(edge_index_add)

        all_edge_index = torch.cat([edge_index_remain, edge_index_delete, edge_index_add], dim=1)
        self._initialize_masks(x, edge_index_remain, edge_index_delete, edge_index_add)

        # explain the model output (not the ground truth), on the augmented graph as in the original
        pred = self._predict(model, x, edge_index).detach()
        target = torch.argmax(pred).item()

        if self.desired_y_cfg is not None:
            self._desired = self.desired_y_cfg
        else:
            self._desired = pred.topk(2)[1][1].item()   # second most likely class

        if self.AT_loss:
            self.pred_same = True

        cfs, cf_labels, cf_perturbs = self._train(model, x, all_edge_index, target=target)
        best_cf, _, _, _ = self._post_process_cfs(cfs, cf_labels, cf_perturbs, total_edges)
        return best_cf, x

    def _train(self, model, x, all_edge_index, *, target):
        if not self.subgraph_mode:
            parameters = [self.edge_mask_delete, self.edge_mask_add]
        else:
            parameters = [self.edge_mask_delete]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        all_edge_mask = self._to_all_edge_mask()
        orig_mask = torch.cat([self.orig_mask_remain, self.orig_mask_delete, self.orig_mask_add])

        cfs, cf_labels, cf_perturbs = [], [], []
        for _ in range(1, self.epochs + 1):
            optimizer.zero_grad()

            y_hat = self._predict(model, x, all_edge_index, all_edge_mask)

            if not self.AT_loss:
                cf_loss = self._cf_loss(y_hat)
            else:
                cf_loss = self._at_cf_loss(y_hat, target)

            perturb_loss = self._perturb_loss()

            if self.ENT_loss:
                ent_loss = self._ent_loss()
                loss = cf_loss + self.coeffs['beta'] * perturb_loss + self.coeffs['gamma'] * ent_loss
            else:
                loss = cf_loss + self.coeffs['beta'] * perturb_loss

            if self.print_loss:
                msg = f"loss:{loss:.4f}, cf_loss:{cf_loss:.4f}, perturb_loss:{perturb_loss:.4f}"
                if self.ENT_loss:
                    msg += f", ent_loss:{ent_loss:.4f}"
                print(msg)

            loss.backward()
            optimizer.step()

            all_edge_mask = self._to_all_edge_mask()

            with torch.no_grad():
                if self.wo_ST:
                    cf_edge_mask = self._binarilize_mask(all_edge_mask.detach().clone(), 0.5)
                else:
                    cf_edge_mask = all_edge_mask.detach().clone()
                cf_edge_index = all_edge_index[:, cf_edge_mask.to(torch.bool)]
                if self.undirected:
                    assert is_undirected(cf_edge_index)
                y_hat = self._predict(model, x, cf_edge_index)
                new_label = torch.argmax(y_hat).item()
                if new_label != target:
                    num_perturb = torch.sum(torch.abs(cf_edge_mask - orig_mask)).item()
                    cfs.append(cf_edge_index)
                    cf_perturbs.append(num_perturb)
                    cf_labels.append(new_label)
                if self.AT_loss:
                    self.pred_same = new_label == target

        return cfs, cf_labels, cf_perturbs

    def _initialize_masks(self, x, edge_index_remain, edge_index_delete, edge_index_add):
        E_remain = edge_index_remain.size(1)
        E_delete, E_add = edge_index_delete.size(1), edge_index_add.size(1)

        if self.print_loss:
            print(E_remain, E_delete, E_add)

        dev = x.device
        self.orig_mask_remain = torch.ones(E_remain, device=dev, dtype=torch.double)
        self.orig_mask_delete = torch.ones(E_delete, device=dev, dtype=torch.double)
        self.orig_mask_add = torch.zeros(E_add, device=dev, dtype=torch.double)

        # random init in [-1, 1)
        edge_mask_delete = torch.rand(E_delete, device=dev, dtype=torch.double) * 2 - 1
        edge_mask_add = torch.rand(E_add, device=dev, dtype=torch.double) * 2 - 1

        self.edge_mask_delete = Parameter(edge_mask_delete)
        if not self.subgraph_mode:
            self.edge_mask_add = Parameter(edge_mask_add)
        else:
            self.edge_mask_add = edge_mask_add

    # losses
    def _perturb_loss(self) -> Tensor:
        a = torch.sigmoid(self.edge_mask_delete) - self.orig_mask_delete
        b = torch.sigmoid(self.edge_mask_add) - self.orig_mask_add
        return torch.sum(torch.abs(a)) + torch.sum(torch.abs(b))

    def _ent_loss(self) -> Tensor:
        m = torch.sigmoid(torch.cat([self.edge_mask_delete, self.edge_mask_add]))
        eps = self.coeffs['EPS']
        return (-m * torch.log(m + eps) - (1 - m) * torch.log(1 - m + eps)).sum()

    def _cf_loss(self, y_hat: Tensor) -> Tensor:
        # oracle returns raw logits (original: ModelReturnType.raw)
        y_vec = y_hat.new_zeros(y_hat.size())
        y_vec[self._desired] = 1.0
        return F.cross_entropy(y_hat, y_vec)

    def _at_cf_loss(self, y_hat: Tensor, y) -> Tensor:
        y = torch.tensor(y, device=y_hat.device)
        cf_loss = -F.cross_entropy(y_hat, y)
        return float(self.pred_same) * cf_loss

    # post-processing
    def _post_process_cfs(self, cfs, cf_labels, cf_perturbs, total_edges):
        if not cfs:
            best_cf = min_perturbs = prop_perturbs = nearest_cf_label = None
        else:
            min_perturbs = min(cf_perturbs)
            idx = cf_perturbs.index(min_perturbs)
            prop_perturbs = min_perturbs / total_edges
            best_cf = cfs[idx]
            nearest_cf_label = cf_labels[idx]
            if not self.silent_mode:
                print(f"num_cfs:{len(cf_perturbs)}, min_perturbs:{min_perturbs}, "
                      f"prop_perturbs:{prop_perturbs}, cf_label {nearest_cf_label}")
                print("=========")

        self._clean_model()
        return best_cf, nearest_cf_label, min_perturbs, prop_perturbs

    def _clean_model(self):
        self.edge_mask_add = None
        self.edge_mask_delete = None
        self.orig_mask_remain = None
        self.orig_mask_add = None
        self.orig_mask_delete = None
        self.pred_same = None
        self._desired = None

    # helpers
    @staticmethod
    def _sort_edge_index(edge_index):
        edge_index = sort_edge_index(edge_index)
        row, col = edge_index[0], edge_index[1]
        edge_index_triu = edge_index[:, row < col]
        edge_index_tril = edge_index_triu.new_zeros(edge_index_triu.size())
        edge_index_tril[0], edge_index_tril[1] = edge_index_triu[1], edge_index_triu[0]
        edge_index_tril = edge_index_tril.flip(dims=[1])
        return torch.cat((edge_index_triu, edge_index_tril), dim=1)

    def _to_all_edge_mask(self):
        edge_mask_delete = torch.sigmoid(self.edge_mask_delete)
        edge_mask_add = torch.sigmoid(self.edge_mask_add)

        def _to_undirected_mask(m):  # without slicing, very important!
            return (m + m.flip(dims=[0])) / 2

        if self.undirected:
            edge_mask_delete = _to_undirected_mask(edge_mask_delete)
            edge_mask_add = _to_undirected_mask(edge_mask_add)

        if self.wo_ST:
            return torch.cat([self.orig_mask_remain, edge_mask_delete, edge_mask_add])
        return torch.cat([self.orig_mask_remain,
                          self._ST_trick(edge_mask_delete),
                          self._ST_trick(edge_mask_add)])

    @staticmethod
    def _ST_trick(edge_mask):
        binarized = (edge_mask > 0.5).detach().clone().to(edge_mask.dtype)
        return binarized - edge_mask.detach() + edge_mask

    @staticmethod
    def _binarilize_mask(mask, threshold):
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        return mask

    @staticmethod
    def _remove_edges(edge_index, edge_index_to_remove):
        r"""Remove edges in edge_index that are also in edge_index_to_remove."""
        all_edge_index = torch.cat([edge_index, edge_index_to_remove], dim=1)
        all_edge_weights = torch.cat([torch.zeros(edge_index.size(1)),
                                      torch.ones(edge_index_to_remove.size(1))]
                                     ).to(all_edge_index.device)
        all_edge_index, all_edge_weights = coalesce(all_edge_index, all_edge_weights)
        return all_edge_index[:, all_edge_weights == 0]