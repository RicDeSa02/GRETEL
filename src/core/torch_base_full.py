import numpy as np
import random
import torch

from src.core.trainable_base import Trainable
from src.utils.cfg_utils import init_dflts_to_of
from src.core.factory_base import get_instance_kvargs
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class TorchBase(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        self.batch_size = self.local_config['parameters']['batch_size']
        
        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        self.device = self.local_config['parameters']['device']
        
        # print(torch.cuda.device_count())
        # print(torch.cuda.current_device())
        # print(torch.cuda.get_device_name(torch.cuda.current_device()))

        self.model.to(self.device)
        for param in self.model.parameters():
            param.data = param.data.to(self.device)
        print(f"Using device: {self.device}") 
        
        self.patience = 0                           
    
    def real_fit(self):
              
        instances = self.dataset.get_torch_instances(fold_id=self.fold_id)
        train_loader, val_loader = None, None
        
        if self.early_stopping_threshold:
            # num_instances = len(self.dataset.instances)
            num_instances = len(instances)
            # print(f'num_instances: {num_instances}')
            # get 5% of training instances and reserve them for validation
            indices = list(range(num_instances))
            random.shuffle(indices)
            val_size = max(int(.05 * len(indices)), self.batch_size)
            train_size = len(indices) - val_size
            # print(f'val_size, train_size: {val_size, train_size}')
            # get the training instances
            # train_instances = Subset(instances, indices[:train_size - 1])
            train_instances = Subset(instances, indices[:train_size])
            val_instances = Subset(instances, indices[train_size:])
            # get the train and validation loaders
            train_loader = DataLoader(train_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(instances, batch_size=self.batch_size, shuffle=True, drop_last=True)

        best_loss = [0,0]
        
        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.model.train()
            self.model.float()
            for batch in train_loader:
                # print("batch.shape:", batch.batch.shape)
                # print("unique graphs in batch:", batch.batch.unique())
                batch = batch.to(self.device)
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.float().to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_weights = batch.edge_attr.float().to(self.device)
                labels = batch.y.to(self.device).long()
                # print(f"labels at torch base: {labels[:5]}")
                # print("labels dtype:", labels.dtype, "min:", labels.min().item(), "max:", labels.max().item())
                
                self.optimizer.zero_grad()

                def batch_full_edge_index_and_weights_good(batch: Data):
                    """
                    Converts a batched PyG Data object into full adjacency matrices
                    with edge weights for all possible edges (0 if missing), suitable
                    for gradient-based CFs.
                    
                    Returns:
                        full_edge_index: [2, N_total_edges] long tensor
                        full_edge_weights: [N_total_edges] float tensor
                    """
                    edge_index_list = []
                    edge_weight_list = []
                    cum_nodes = 0  # to offset node indices for batching

                    # Iterate over each graph in the batch
                    for i in range(batch.num_graphs):
                        mask = batch.batch == i
                        num_nodes = mask.sum().item()
                        
                        # Extract node indices for this graph
                        nodes = torch.arange(num_nodes)
                        
                        # Build full adjacency
                        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.flaot(), device=batch.x.device)
                        
                        # Original edges for this graph
                        graph_edge_mask = (batch.batch[batch.edge_index[0]] == i)
                        edge_idx = batch.edge_index[:, graph_edge_mask] - cum_nodes
                        edge_attr = batch.edge_attr[graph_edge_mask]
                        adj[edge_idx[0], edge_idx[1]] = edge_attr  # fill original edges
                        # adj.fill_diagonal_(1) # ensure self-loops

                        # Flatten full adjacency
                        row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
                        edge_index_list.append(torch.stack([row.flatten(), col.flatten()], dim=0) + cum_nodes)
                        edge_weight_list.append(adj.flatten())

                        cum_nodes += num_nodes

                    # Concatenate all graphs
                    full_edge_index = torch.cat(edge_index_list, dim=1)
                    full_edge_weights = torch.cat(edge_weight_list, dim=0)

                    return full_edge_index.long(), full_edge_weights

                def batch_full_edge_index_and_weights(batch: Data):
                    """
                    Converts a batched PyG Data object into full adjacency matrices
                    with edge weights for all possible edges (0 if missing), suitable
                    for gradient-based CFs. Vectorized for GPU.
                    """
                    device = batch.x.device
                    batch_idx = batch.batch

                    # Compute number of nodes per graph
                    num_nodes_per_graph = torch.bincount(batch_idx).to(device)
                    cum_nodes = torch.cat([torch.tensor([0], device=device), num_nodes_per_graph.cumsum(dim=0)[:-1]])

                    # Prepare flattened adjacency matrix
                    full_edge_index_list = []
                    full_edge_weight_list = []

                    # Create full adjacency for each graph in parallel
                    for i, n_nodes in enumerate(num_nodes_per_graph):
                        # Node offset
                        offset = cum_nodes[i]
                        nodes = torch.arange(n_nodes, device=device)

                        # Build full adjacency indices
                        row, col = torch.meshgrid(nodes, nodes, indexing='ij')
                        full_edge_index_list.append(torch.stack([row.flatten(), col.flatten()], dim=0) + offset)

                        # Original edges for this graph
                        mask = batch_idx[batch.edge_index[0]] == i
                        edge_idx = batch.edge_index[:, mask] - offset
                        edge_attr = batch.edge_attr[mask]

                        # Fill adjacency weights
                        adj = torch.zeros((n_nodes, n_nodes), dtype=batch.x.dtype, device=device)
                        if edge_attr.numel() > 0:
                            adj[edge_idx[0], edge_idx[1]] = edge_attr
                        full_edge_weight_list.append(adj.flatten())

                    full_edge_index = torch.cat(full_edge_index_list, dim=1)
                    full_edge_weights = torch.cat(full_edge_weight_list, dim=0)

                    return full_edge_index.long(), full_edge_weights


                # print(f"edge_index.shape: {edge_index.shape} - edge_weights.shape: {edge_weights.shape}")
                # print(f"edge_index: {edge_index[0][:20]}{edge_index[1][:20]} - edge_weights: {edge_weights[:20]}")
                edge_index, edge_weights = batch_full_edge_index_and_weights(batch)
                # print(f"edge_index.shape: {edge_index.shape} - edge_weights.shape: {edge_weights.shape}")
                # print(f"edge_index: {edge_index[0][:20]}{edge_index[1][:20]} - edge_weights: {edge_weights[:20]}")
                edge_index, edge_weights = edge_index.to(self.device), edge_weights.to(self.device)
                # input()
                
                pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                # print(f'pred.shape, labels.shape: {pred.shape, labels.shape}')
                # print(f'pred: {pred[:5]}')
                if pred.shape[1] > 2: # and labels.dim() > 1:
                    labels = torch.nn.functional.one_hot(labels, num_classes=pred.shape[1]).float()                  
                    # print(f'labels after one hot: {labels[:5]}')
                loss = self.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())
               
                self.optimizer.step()

            accuracy = self.accuracy(labels_list, preds)
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            self.lr_scheduler.step()
            
            # check if we need to do early stopping
            if self.early_stopping_threshold and len(val_loader) > 0:
                self.model.eval()
                self.model.float()
                var_losses, var_labels, var_preds = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch.batch = batch.batch.to(self.device)
                        node_features = batch.x.float().to(self.device)
                        edge_index = batch.edge_index.to(self.device)
                        edge_weights = batch.edge_attr.float().to(self.device)
                        labels = batch.y.to(self.device).long()

                        pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                        if pred.shape[1] > 2:
                            labels = torch.nn.functional.one_hot(labels, num_classes=pred.shape[1]).float()
                        loss = self.loss_fn(pred, labels)
                        
                        var_labels += list(labels.squeeze().to('cpu').numpy())
                        var_preds += list(pred.squeeze().to('cpu').numpy())
                        
                        var_losses.append(loss.item())
                        
                    best_loss.pop(0)
                    var_loss = np.mean(var_losses)
                    best_loss.append(var_loss)
                            
                    accuracy = self.accuracy(var_labels, var_preds)
                    self.context.logger.info(f'epoch = {epoch} ---> valid_loss = {var_loss:.4f}\t valid_accuracy = {accuracy:.4f}')
                
                if abs(best_loss[0] - best_loss[1]) < self.early_stopping_threshold:
                    self.patience += 1
                    
                    if self.patience == 4:
                        self.context.logger.info(f"Early stopped training at epoch {epoch}")
                        break  # terminate the training loop
                
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 200)
        # local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 1)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')
        local_config['parameters']['device'] = local_config['parameters'].get('device', (
            "cuda"
            if torch.cuda.is_available() or torch.cuda.device_count()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"))
        
    def accuracy(self, testy, probs):
        # print(testy[:10], probs[:10])
        # testy_idx = np.argmax(testy, axis=1)
        # probs_idx = np.argmax(probs, axis=1)
        # print(testy_idx[:1], probs_idx[:1])
        # print(len(testy))

        if len(probs[0]) > 2:
            # print(np.argmax(testy, axis=1), np.argmax(testy, axis=1)-1, np.argmax(probs, axis=1))
            acc = accuracy_score(np.argmax(testy, axis=1), np.argmax(probs, axis=1))
        else:
            acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc

    def read(self):
        super().read()
        if isinstance(self.model, list):
            for mod in self.model:
                mod.to(self.device)
        else:
            self.model.to(self.device)
            
    def to(self, device):
        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        elif isinstance(self.model, list):
            for model in self.model:
                if isinstance(model, torch.nn.Module):
                    model.to(self.device)