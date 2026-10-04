import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from .base import ExplainerD4
from .diffusion.graph_utils import (
    gen_full,
    gen_list_of_data_single,
    generate_mask,
    graph2tensor,
    tensor2graph,
)
from .diffusion.pgnn import Powerful

import torch.nn.functional as F
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import retake_oracle
import argparse
from torch_geometric.data import Batch, Data
from src.dataset.instances.graph import GraphInstance
from torch_geometric.utils import to_scipy_sparse_matrix

def model_save(args, model, mean_train_loss, best_sparsity, mean_test_acc):
    """
    Save the model to disk
    :param args: arguments
    :param model: model
    :param mean_train_loss: mean training loss
    :param best_sparsity: best sparsity
    :param mean_test_acc: mean test accuracy
    """
    to_save = {
        "model": model.state_dict(),
        "train_loss": mean_train_loss,
        "eval sparsity": best_sparsity,
        "eval acc": mean_test_acc,
    }
    exp_dir = f"{args.root}/{args.dataset}/"
    os.makedirs(exp_dir, exist_ok=True)
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    print(f"save model to {exp_dir}/best_model.pth")


def loss_func_bce(score_list, groundtruth, sigma_list, mask, device, sparsity_level):
    """
    Loss function for binary cross entropy
    param score_list: [len(sigma_list)*bsz, N, N]
    param groundtruth: [len(sigma_list)*bsz, N, N]
    param sigma_list: list of sigma values
    param mask: [len(sigma_list)*bsz, N, N]
    param device: device
    param sparsity_level: sparsity level
    return: BCE loss
    """
    bsz = int(score_list.size(0) / len(sigma_list))
    num_node = score_list.size(-1)
    score_list = score_list * mask
    groundtruth = groundtruth * mask
    pos_weight = torch.full([num_node * num_node], sparsity_level).to(device)
    BCE = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    score_list_ = torch.flatten(score_list, start_dim=1, end_dim=-1)
    groundtruth_ = torch.flatten(groundtruth, start_dim=1, end_dim=-1)
    loss_matrix = BCE(score_list_, groundtruth_)
    loss_matrix = loss_matrix.view(groundtruth.size(0), num_node, num_node)
    loss_matrix = loss_matrix * (
        1
        - 2
        * torch.tensor(sigma_list)
        .repeat(bsz)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(groundtruth.size(0), num_node, num_node)
        .to(device)
        + 1.0 / len(sigma_list)
    )
    loss_matrix = loss_matrix * mask
    loss_matrix = (loss_matrix + torch.transpose(loss_matrix, -2, -1)) / 2
    loss = torch.mean(loss_matrix)
    return loss


def sparsity(score, groundtruth, mask, threshold=0.5):
    """
    Calculate the sparsity of the predicted adjacency matrix
    :param score: [bsz, N, N, 1]
    :param groundtruth: [bsz, N, N]
    :param mask: [bsz, N, N]
    :param threshold: threshold for the predicted adjacency matrix
    :return: sparsity
    """
    score_tensor = torch.stack(score, dim=0).squeeze(-1)  # [len_sigma_list, bsz, N, N]
    score_tensor = torch.mean(score_tensor, dim=0)  # [bsz, N, N]
    pred_adj = torch.where(torch.sigmoid(score_tensor) > threshold, 1, 0).to(groundtruth.device)
    pred_adj = pred_adj * mask
    groundtruth_ = groundtruth * mask
    adj_diff = torch.abs(groundtruth_ - pred_adj)  # [bsz, N, N]
    num_edge_b = groundtruth_.sum(dim=(1, 2))
    adj_diff_ratio = adj_diff.sum(dim=(1, 2)) / num_edge_b
    ratio_average = torch.mean(adj_diff_ratio)
    return ratio_average


def gnn_pred(graph_batch, graph_batch_sub, gnn_model, ds, task, device):
    """
    Predict the labels of the graph
    :param graph_batch: graph batch
    :param graph_batch_sub: subgraph batch
    :param gnn_model: GNN model
    :param ds: dataset
    :param task: task
    :return: predicted labels (full graph and subgraph)
    """
    graph_batch = graph_batch.to(device)
    graph_batch_sub = graph_batch_sub.to(device)
    gnn_model = gnn_model.to(device)
    gnn_model.eval()
    if task == "nc":
        output_prob = F.softmax(gnn_model.get_node_pred_subgraph(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_weight=graph_batch.edge_index,
            mapping=graph_batch.mapping,
        ), dim=1)
        output_prob_sub = F.softmax(gnn_model.get_node_pred_subgraph(
            x=graph_batch_sub.x,
            edge_index=graph_batch_sub.edge_index,
            edge_weight=graph_batch.edge_index,
            mapping=graph_batch_sub.mapping,
        ), dim=1)
    else:
        # print(graph_batch.edge_index.shape); input()
        output_prob = F.softmax(gnn_model.forward(
                node_features=graph_batch.x,
                edge_index=graph_batch.edge_index,
                edge_weight=graph_batch.edge_attr,
                batch=graph_batch.batch,
            ), dim=1)
        output_prob_sub = F.softmax(gnn_model.forward(
                node_features=graph_batch_sub.x,
                edge_index=graph_batch_sub.edge_index,
                edge_weight=None,
                batch=graph_batch_sub.batch,
            ), dim=1)

    y_pred = output_prob.argmax(dim=-1)
    y_exp = output_prob_sub.argmax(dim=-1)
    return y_pred, y_exp


def loss_cf_exp(gnn_model, graph_batch, score, y_pred, y_exp, full_edge, mask, ds, task="nc"):
    """
    Loss function for counterfactual explanation
    :param gnn_model: GNN model
    :param graph_batch: graph batch
    :param score: list of scores
    :param y_pred: predicted labels
    :param y_exp: predicted labels for subgraph
    :param full_edge: full edge index
    :param mask: mask
    :param ds: dataset
    :param task: task
    :return: loss
    """
    score_tensor = torch.stack(score, dim=0).squeeze(-1)
    score_tensor = torch.mean(score_tensor, dim=0).view(-1, 1)
    mask_bool = mask.bool().view(-1, 1)
    edge_mask_full = score_tensor[mask_bool]
    assert edge_mask_full.size(0) == full_edge.size(1)
    criterion = torch.nn.NLLLoss()

    edge_mask_full = edge_mask_full.sigmoid()
    if task == "nc":
        output_repr_cont = gnn_model.forward(
            node_features=graph_batch.x,
            edge_index=full_edge,
            edge_weight=edge_mask_full,
        )
        output_prob_cont = F.softmax(output_repr_cont, dim=1)[graph_batch.mapping]
    else:
        output_repr_cont = gnn_model.forward(
            node_features=graph_batch.x,
            edge_index=full_edge,
            edge_weight=edge_mask_full,
            batch=graph_batch.batch,
        )
        output_prob_cont = F.softmax(output_repr_cont, dim=1)
    n = output_repr_cont.size(-1)
    bsz = output_repr_cont.size(0)
    y_exp = output_prob_cont.argmax(dim=-1)
    inf_diag = torch.diag(-torch.ones((n)) / 0).unsqueeze(0).repeat(bsz, 1, 1).to(y_pred.device)
    neg_prop = (output_repr_cont.unsqueeze(1).expand(bsz, n, n) + inf_diag).logsumexp(-1)
    neg_prop = neg_prop - output_repr_cont.logsumexp(-1).unsqueeze(1).repeat(1, n)
    loss_cf = criterion(neg_prop, y_pred)
    labels = torch.LongTensor([[i] for i in y_pred]).to(y_pred.device)
    fid_drop = (1 - output_prob_cont.gather(1, labels).view(-1)).detach().cpu().numpy()
    fid_drop = np.mean(fid_drop)
    acc_cf = float(y_exp.eq(y_pred).sum().item() / y_pred.size(0))  # less, better
    return loss_cf, fid_drop, acc_cf

def dict_to_namespace(d):
    ns = argparse.Namespace()
    for key, value in d.items():
        setattr(ns, key, dict_to_namespace(value) if isinstance(value, dict) else value)
    return ns

class DiffExplainer(ExplainerD4, Trainable, Explainer):
    def init_args(self):
        self.cuda = self.local_config['parameters']['cuda']
        self.root = self.local_config['parameters']['root']
        self.dataset = self.local_config['parameters']['dataset']
        self.verbose = self.local_config['parameters']['verbose']
        self.gnn_type = self.local_config['parameters']['gnn_type']
        self.task = self.local_config['parameters']['task']
        self.train_batchsize = self.local_config['parameters']['train_batchsize']
        self.test_batchsize = self.local_config['parameters']['test_batchsize']
        self.sigma_length = self.local_config['parameters']['sigma_length']
        self.epoch = self.local_config['parameters']['epoch']
        self.feature_in = self.local_config['parameters']['feature_in']
        self.data_size = self.local_config['parameters']['data_size']
        self.threshold = self.local_config['parameters']['threshold']
        self.dropout = self.local_config['parameters']['dropout']
        self.learning_rate = self.local_config['parameters']['learning_rate']
        self.lr_decay = self.local_config['parameters']['lr_decay']
        self.weight_decay = self.local_config['parameters']['weight_decay']
        self.prob_low = self.local_config['parameters']['prob_low']
        self.prob_high = self.local_config['parameters']['prob_high']
        self.sparsity_level = self.local_config['parameters']['sparsity_level']
        self.normalization = self.local_config['parameters']['normalization']
        self.num_layers = self.local_config['parameters']['num_layers']
        self.layers_per_conv = self.local_config['parameters']['layers_per_conv']
        self.n_hidden = self.local_config['parameters']['n_hidden']
        self.cat_output = self.local_config['parameters']['cat_output']
        self.residual = self.local_config['parameters']['residual']
        self.noise_mlp = self.local_config['parameters']['noise_mlp']
        self.simplified = self.local_config['parameters']['simplified']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def init(self):
        # self.init_args()        
        self.args = dict_to_namespace(self.local_config['parameters'])
        self.model = self.oracle.model
        ExplainerD4.init(self, self.args.device)

    def explain_graph_task(self, train_dataset, test_dataset):
        """
        Explain the graph for a specific dataset and task
        :param args: arguments
        :param train_dataset: training dataset
        :param test_dataset: test dataset
        """
        args = self.args
        gnn_model = self.model.to(args.device)
        model = Powerful(args).to(args.device)
        self.real_fit(args, model, gnn_model, train_dataset, test_dataset)

    def real_fit(self):
        """
        Train the model
        :param args: arguments
        :param model: Powerful (explanation) model
        :param gnn_model: GNN model
        :param train_dataset: training dataset
        :param test_dataset: test dataset
        """
        args = self.args
        gnn_model = self.model
        model = Powerful(args).to(args.device)
        train_dataset = self.dataset.get_torch_instances(fold_id=self.fold_id, dataset_kls='src.dataset.utils.dataset_torch.TorchGeometricDataset', usage='train') # max_nodes=args.n_nodes, 
        test_dataset = self.dataset.get_torch_instances(fold_id=self.fold_id, dataset_kls='src.dataset.utils.dataset_torch.TorchGeometricDataset', usage='test') # max_nodes=args.n_nodes, 

        best_sparsity = np.inf
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
        noise_list = args.noise_list
        saved = False
        for epoch in range(args.epoch):
            print(f"start epoch {epoch}")
            train_losses = []
            train_loss_dist = []
            train_loss_cf = []
            train_acc = []
            train_fid = []
            train_sparsity = []
            train_remain = []
            model.train()
            train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True)
            for i, graph in enumerate(train_loader):
                # print(i, type(graph)); input()
                if graph.is_directed():
                    edge_index_temp = graph.edge_index
                    graph.edge_index = to_undirected(edge_index=edge_index_temp)
                graph.to(args.device)
                train_adj_b, train_x_b = graph2tensor(graph, device=args.device)
                # train_adj_b: [bsz, N, N]; train_x_b: [bsz, N, C]
                sigma_list = (
                    list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length))
                    if noise_list is None
                    else noise_list
                )
                train_node_flag_b = train_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)  # [bsz, N]
                # all nodes that are not connected with others
                if isinstance(sigma_list, float):
                    sigma_list = [sigma_list]
                (train_x_b, train_ori_adj_b, train_node_flag_sigma, train_noise_adj_b, _) = gen_list_of_data_single(
                    train_x_b, train_adj_b, train_node_flag_b, sigma_list, args
                )
                optimizer.zero_grad()
                train_noise_adj_b_chunked = train_noise_adj_b.chunk(len(sigma_list), dim=0)
                train_x_b_chunked = train_x_b.chunk(len(sigma_list), dim=0)
                train_node_flag_sigma = train_node_flag_sigma.chunk(len(sigma_list), dim=0)
                score = []
                masks = []
                for i, sigma in enumerate(sigma_list):
                    mask = generate_mask(train_node_flag_sigma[i])
                    score_batch = model(
                        A=train_noise_adj_b_chunked[i].to(args.device),
                        node_features=train_x_b_chunked[i].to(args.device),
                        mask=mask.to(args.device),
                        noiselevel=sigma,
                    )  # [bsz, N, N, 1]
                    score.append(score_batch)
                    masks.append(mask)
                graph_batch_sub = tensor2graph(graph, score, mask)
                y_pred, y_exp = gnn_pred(graph, graph_batch_sub, gnn_model, ds=args.dataset, task=args.task, device=args.device)
                full_edge_index = gen_full(graph.batch, mask)
                score_b = torch.cat(score, dim=0).squeeze(-1).to(args.device)  # [len(sigma_list)*bsz, N, N]
                masktens = torch.cat(masks, dim=0).to(args.device)  # [len(sigma_list)*bsz, N, N]
                modif_r = sparsity(score, train_adj_b, mask)
                remain_r = sparsity(score, train_adj_b, train_adj_b)
                loss_cf, fid_drop, acc_cf = loss_cf_exp(
                    gnn_model, graph, score, y_pred, y_exp, full_edge_index, mask, ds=args.dataset, task=args.task
                )
                loss_dist = loss_func_bce(
                    score_b,
                    train_ori_adj_b,
                    sigma_list,
                    masktens,
                    device=args.device,
                    sparsity_level=args.sparsity_level,
                )
                loss = loss_dist + args.alpha_cf * loss_cf
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_loss_dist.append(loss_dist.item())
                train_loss_cf.append(loss_cf.item())
                train_acc.append(acc_cf)
                train_fid.append(fid_drop)
                train_sparsity.append(modif_r.item())
                train_remain.append(remain_r.item())
            scheduler.step(epoch)
            mean_train_loss = np.mean(train_losses)
            mean_train_acc = 1- np.mean(train_acc)
            mean_train_fidelity = np.mean(train_fid)
            mean_train_sparsity = np.mean(train_sparsity)
            print(
                (
                    f"Training Epoch: {epoch} | "
                    f"training loss: {mean_train_loss} | "
                    f"training fidelity drop: {mean_train_fidelity} | "
                    f"training cf acc: {mean_train_acc} | "
                    f"training average modification: {mean_train_sparsity} | "
                )
            )
            # evaluation
            if (epoch + 1) % args.verbose == 0:
                test_losses = []
                test_loss_dist = []
                test_loss_cf = []
                test_acc = []
                test_fid = []
                test_sparsity = []
                test_remain = []
                test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batchsize, shuffle=False)
                model.eval()
                for graph in test_loader:
                    if graph.is_directed():
                        edge_index_temp = graph.edge_index
                        graph.edge_index = to_undirected(edge_index=edge_index_temp)

                    graph.to(args.device)
                    test_adj_b, test_x_b = graph2tensor(graph, device=args.device)
                    test_x_b = test_x_b.to(args.device)
                    test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
                    sigma_list = (
                        list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length))
                        if noise_list is None
                        else noise_list
                    )
                    if isinstance(sigma_list, float):
                        sigma_list = [sigma_list]
                    (test_x_b, test_ori_adj_b, test_node_flag_sigma, test_noise_adj_b, _) = gen_list_of_data_single(
                        test_x_b, test_adj_b, test_node_flag_b, sigma_list, args
                    )
                    with torch.no_grad():
                        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
                        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
                        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)
                        score = []
                        masks = []
                        for i, sigma in enumerate(sigma_list):
                            mask = generate_mask(test_node_flag_sigma[i])
                            score_batch = model(
                                A=test_noise_adj_b_chunked[i].to(args.device),
                                node_features=test_x_b_chunked[i].to(args.device),
                                mask=mask.to(args.device),
                                noiselevel=sigma,
                            ).to(args.device)
                            masks.append(mask)
                            score.append(score_batch)
                        graph_batch_sub = tensor2graph(graph, score, mask)
                        y_pred, y_exp = gnn_pred(graph, graph_batch_sub, gnn_model, ds=args.dataset, task=args.task, device=args.device)
                        full_edge_index = gen_full(graph.batch, mask)
                        score_b = torch.cat(score, dim=0).squeeze(-1).to(args.device)
                        masktens = torch.cat(masks, dim=0).to(args.device)
                        modif_r = sparsity(score, test_adj_b, mask)
                        loss_cf, fid_drop, acc_cf = loss_cf_exp(
                            gnn_model,
                            graph,
                            score,
                            y_pred,
                            y_exp,
                            full_edge_index,
                            mask,
                            ds=args.dataset,
                            task=args.task,
                        )
                        loss_dist = loss_func_bce(
                            score_b,
                            test_ori_adj_b,
                            sigma_list,
                            masktens,
                            device=args.device,
                            sparsity_level=args.sparsity_level,
                        )
                        loss = loss_dist + args.alpha_cf * loss_cf
                        test_losses.append(loss.item())
                        test_loss_dist.append(loss_dist.item())
                        test_loss_cf.append(loss_cf.item())
                        test_acc.append(acc_cf)
                        test_fid.append(fid_drop)
                        test_sparsity.append(modif_r.item())
                mean_test_loss = np.mean(test_losses)
                mean_test_acc = 1- np.mean(test_acc)
                mean_test_fid = np.mean(test_fid)
                mean_test_sparsity = np.mean(test_sparsity)
                print(
                    (
                        f"Evaluation Epoch: {epoch} | "
                        f"test loss: {mean_test_loss} | "
                        f"test fidelity drop: {mean_test_fid} | "
                        f"test cf acc: {mean_test_acc} | "
                        f"test average modification: {mean_test_sparsity} | "
                    )
                )
                if mean_test_sparsity < best_sparsity:
                    best_sparsity = mean_test_sparsity
                    model_save(args, model, mean_train_loss, best_sparsity, mean_test_acc)
                    saved = True
                if not saved:
                    best_sparsity = mean_test_sparsity
                    model_save(args, model, mean_train_loss, best_sparsity, mean_test_acc)
                    saved = True
        if not saved:
            model_save(args, model, mean_train_loss, best_sparsity, mean_test_acc:=None)

    def explain(self, graph):
        """
        Explain the graph with the trained model
        :param args: arguments
        :param graph: graph to be explained
        :return: the explanation (edge_mask, original prediction, explanation prediction, modification rate)
        """
        initial_graph = graph
        args = self.args
        model = Powerful(args).to(args.device)
        exp_dir = f"{args.root}/{args.dataset}/"
        model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth"))["model"])
        model.eval()
        # graph.to(args.device)
        # print(graph.data); input()
        # print(np.where(graph.data != 0)[0].shape)
        edge_index = np.where(graph.data != 0)
        # print(f"edge_index type: {type(edge_index)}")
        # print(graph.node_features.shape, graph.data.shape, graph.edge_features.shape); input()
        data_graph = Data(x=graph.node_features, edge_index=edge_index).to(args.device)
        # data_graph.batch = torch.zeros(data_graph.x.size(0), dtype=torch.long)
        graph = Batch.from_data_list([data_graph])
        # graph = Batch.from_data_list([graph])
        test_adj_b, test_x_b = graph2tensor(graph, device=args.device)  # [bsz, N, N]
        # test_adj_b, test_x_b = torch.from_numpy(graph.data).to(args.device), torch.from_numpy(graph.node_features).to(args.device)
        test_adj_b = test_adj_b.to(args.device)
        test_x_b = test_x_b.to(args.device)
        test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
        sigma_list = (
            list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length))
            if args.noise_list is None
            else args.noise_list
        )
        if isinstance(sigma_list, float):
            sigma_list = [sigma_list]
        (test_x_b, _, test_node_flag_sigma, test_noise_adj_b, _) = gen_list_of_data_single(
            test_x_b, test_adj_b, test_node_flag_b, sigma_list, args
        )
        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)
        score = []
        masks = []
        for i, sigma in enumerate(sigma_list):
            mask = generate_mask(test_node_flag_sigma[i])
            score_batch = model(
                A=test_noise_adj_b_chunked[i].to(args.device),
                node_features=test_x_b_chunked[i].to(args.device),
                mask=mask.to(args.device),
                noiselevel=sigma,
            ).to(args.device)
            masks.append(mask)
            score.append(score_batch)
        graph_batch_sub = tensor2graph(graph, score, mask)
        full_edge_index = gen_full(graph.batch, mask)
        modif_r = sparsity(score, test_adj_b, mask)
        score_tensor = torch.stack(score, dim=0).squeeze(-1)  # len_sigma_list, bsz, N, N]
        score_tensor = torch.mean(score_tensor, dim=0).view(-1, 1)  # [bsz*N*N,1]
        mask_bool = mask.bool().view(-1, 1)
        edge_mask_full = score_tensor[mask_bool].double()
        # print(type(graph.x[0]), type(full_edge_index))
        # print(graph.x)
        graph.x = torch.from_numpy(np.array(graph.x)).double()
        if graph.x.dim() == 3 and graph.x.size(0) == 1:
            graph.x = graph.x.squeeze(0)
        if args.task == "nc":
            output_prob_cont = self.model.forward(
                node_features=graph.x.to(args.device), edge_index=full_edge_index.to(args.device), edge_weight=edge_mask_full.to(args.device)
            )
            output_prob_cont = F.softmax(output_prob_cont, dim=1)[graph.mapping]
        else:
            output_prob_cont = self.model.forward(
                node_features=graph.x.to(args.device), edge_index=full_edge_index.to(args.device), edge_weight=edge_mask_full.to(args.device), batch=graph.batch.to(args.device)
            )
            output_prob_cont = F.softmax(output_prob_cont, dim=1)
        y_ori = graph.y if args.task == "gc" else graph.self_y
        y_exp = output_prob_cont.argmax(dim=-1)

        # final_graph = Batch.to_data_list(graph_batch_sub)[0]

        if graph_batch_sub.edge_index.numel() == 0:
            # Create an empty Data object with the same node features
            final_graph = Data(x=graph_batch_sub.x[0], edge_index=torch.empty((2, 0), dtype=torch.long))
        else:
            final_graph = Batch.to_data_list(graph_batch_sub)[0]

        adjacency = np.zeros((final_graph.x.shape[0], final_graph.x.shape[0]))
        indices = final_graph.edge_index.cpu().numpy()
        # print(f"indices: {indices}")
        adjacency[indices[0], indices[1]] = 1
        # print(f"adjacency: {adjacency}")
        weights = adjacency[indices[0], indices[1]]

        final_graph_GI = GraphInstance(
            id = initial_graph.id,
            label = y_exp,
            data = adjacency,
            node_features = final_graph.x,
            # edge_features = initial_graph.edge_features,
            edge_weights = weights,
            graph_features = initial_graph.graph_features
        )
        return final_graph_GI # graph_batch_sub, y_ori, y_exp, modif_r

    def one_step_model_level(self, args, random_adj, node_feature, sigma):
        """
        One-step Model level explanation using the trained model
        Run multiple steps to get model-level explanation.
        :param args: arguments
        :param random_adj: a random adjacency matrix seed
        :param node_feature: node features of the dataset
        :param sigma: noise level
        :return: A predicted adjacency matrix
        """
        random_adj = random_adj.unsqueeze(0)  # batchsize=1
        node_feature = node_feature.unsqueeze(0)  # batchsize=1
        mask = torch.ones_like(random_adj).to(args.device)
        model = Powerful(args).to(args.device)
        exp_dir = f"{args.root}/{args.dataset}/"
        model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth"))["model"])
        model.eval()
        score = model(A=random_adj, node_features=node_feature, mask=mask, noiselevel=sigma).to(args.device)
        score = score.squeeze(0).squeeze(-1)
        pred_adj = torch.where(torch.sigmoid(score) > 0.5, 1, 0).to(score.device)
        return pred_adj  # [N, N]

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['cuda'] = self.local_config['parameters'].get('cuda', 0)
        self.local_config['parameters']['root'] = self.local_config['parameters'].get('root', "src/explainer/D4Explainer/explainers/results")
        self.local_config['parameters']['dataset'] = self.local_config['parameters'].get('dataset', "Tree_Cycle")
        self.local_config['parameters']['verbose'] = self.local_config['parameters'].get('verbose', 10)
        self.local_config['parameters']['gnn_type'] = self.local_config['parameters'].get('gnn_type', 'gcn')
        self.local_config['parameters']['task'] = self.local_config['parameters'].get('task', 'nc')
        self.local_config['parameters']['train_batchsize'] = self.local_config['parameters'].get('train_batchsize', 32)
        self.local_config['parameters']['test_batchsize'] = self.local_config['parameters'].get('test_batchsize', 32)
        self.local_config['parameters']['sigma_length'] = self.local_config['parameters'].get('sigma_length', 10)
        self.local_config['parameters']['epoch'] = self.local_config['parameters'].get('epoch', 5)
        self.local_config['parameters']['feature_in'] = self.local_config['parameters'].get('feature_in', len(self.dataset.node_features_map))
        self.local_config['parameters']['data_size'] = self.local_config['parameters'].get('data_size', -1)
        self.local_config['parameters']['threshold'] = self.local_config['parameters'].get('threshold', 0.5)
        self.local_config['parameters']['alpha_cf'] = self.local_config['parameters'].get('alpha_cf', 0.5)
        self.local_config['parameters']['dropout'] = self.local_config['parameters'].get('dropout', 0.001)
        self.local_config['parameters']['learning_rate'] = self.local_config['parameters'].get('learning_rate', 1e-3)
        self.local_config['parameters']['lr_decay'] = self.local_config['parameters'].get('lr_decay', 0.999)
        self.local_config['parameters']['weight_decay'] = self.local_config['parameters'].get('weight_decay', 0.0)
        self.local_config['parameters']['prob_low'] = self.local_config['parameters'].get('prob_low', 0.0)
        self.local_config['parameters']['prob_high'] = self.local_config['parameters'].get('prob_high', 0.4)
        self.local_config['parameters']['sparsity_level'] = self.local_config['parameters'].get('sparsity_level', 2.5)
        self.local_config['parameters']['normalization'] = self.local_config['parameters'].get('normalization', 'instance')
        self.local_config['parameters']['num_layers'] = self.local_config['parameters'].get('num_layers', 6)
        self.local_config['parameters']['layers_per_conv'] = self.local_config['parameters'].get('layers_per_conv', 1)
        self.local_config['parameters']['n_hidden'] =  self.local_config['parameters'].get('n_hidden', 64)
        self.local_config['parameters']['cat_output'] = self.local_config['parameters'].get('cat_output', True)
        self.local_config['parameters']['residual'] = self.local_config['parameters'].get('residual', False)
        self.local_config['parameters']['noise_mlp'] = self.local_config['parameters'].get('noise_mlp', True)
        self.local_config['parameters']['simplified'] = self.local_config['parameters'].get('simplified', False)
        self.local_config['parameters']['device'] = self.local_config['parameters'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.local_config['parameters']['n_nodes'] = self.local_config['parameters'].get('n_nodes', np.max(self.dataset.num_nodes_values))
        self.local_config['parameters']['noise_list'] = self.local_config['parameters'].get('noise_list', None)
    