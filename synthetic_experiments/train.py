''' usage example:
        # 1. Train synthetic graph with linear GCN
        python train.py --synthetic --model_type graffgcn \
        --step_size 0.1 --num_layers 60 --normalize none --linear \
        --base_dataset cora --graph_type random --edge_homo 0.1 --degree_intra 2 \
        --num_graph 3 --lr 0.005 --weight_decay 0.0001 --dropout 0
        
        # 2. Train real graph with linear GCN
        python train.py --model_type graffgcn \
        --step_size 0.1 --num_layers 60 --normalize none --linear \
        --base_dataset cora \
        --num_graph 3 --lr 0.005 --weight_decay 0.0001 --dropout 0
        
        # NOTE: choice of datasets: ['cora', 'citeseer', 'pubmed',
        #                            'cornell', 'texas', 'wisconsin',
        #                            'film', 'chameleon', 'squirrel']
'''

from __future__ import division
from __future__ import print_function

import time
import os 

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from arg_parser import arg_parser
from models import GRAFFNet, GCN
from logger import SyntheticExpLogger
from utils import accuracy, load_synthetic_data, normalize, random_disassortative_splits, load_full_data

logger = SyntheticExpLogger()

# Training settings
args = arg_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.pi = torch.acos(torch.zeros(1)).item() * 2

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

num_edge_same = args.degree_intra * 400

# Log info
model_info = {
    "model_type": args.model_type,
    "graph_type": args.graph_type,
    "num_edge_same": num_edge_same,
    "edge_homo": args.edge_homo,
    "base_dataset": args.base_dataset,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "dropout": args.dropout,
}

run_info = {
    "result": 0,
    "graph_idx": 0
}

record_info = {
    "test_result": 0,
    "test_std": 0,
    "graph_idx": 0
}

# Load data
t_total = time.time()

best_result = 0
best_std = 0
best_dropout = None
best_weight_decay = None

# Path 
if not args.synthetic:
    if args.linear:
        directory = f'ckpt_real_graph_clean/{args.base_dataset}/{args.model_type}/linear/ss_{args.step_size}/nl_{args.num_layers}/norm_{args.normalize}'
    else:
        directory = f'ckpt_real_graph_clean/{args.base_dataset}/{args.model_type}/act_{args.activation}/ss_{args.step_size}/nl_{args.num_layers}/norm_{args.normalize}'
else:
    if args.linear: 
        directory = f'ckpt_syn_graph_clean/{args.base_dataset}/{args.edge_homo}/{args.model_type}/linear/ss_{args.step_size}/nl_{args.num_layers}/norm_{args.normalize}'
    else:
        directory = f'ckpt_syn_graph_clean/{args.base_dataset}/{args.edge_homo}/{args.model_type}/act_{args.activation}/ss_{args.step_size}/nl_{args.num_layers}/norm_{args.normalize}'        

path = os.path.join(os.curdir, directory) 
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)  # Create directory and its parents if they don't exist

result = np.zeros(args.num_graph)
for sample in range(args.num_graph):
    run_info["graph_idx"] = sample
    
    if args.synthetic:
        adj, labels, degree, features = load_synthetic_data(
            args.graph_type, sample, args.edge_homo, args.base_dataset
        )
        nnodes = adj.shape[0]
        adj_dense = adj
        adj_dense[adj_dense != 0] = 1
        adj_dense = adj_dense - torch.diag(torch.diag(adj_dense))
        
        adj_low = torch.tensor(normalize(adj_dense + torch.eye(nnodes)))        
        adj_high = torch.eye(nnodes) - adj_low
        adj_low = adj_low.to_sparse()
        adj_high = adj_high.to_sparse()
    else:
        adj, _, features, labels = load_full_data(args.base_dataset)
        adj_low = adj 
        adj_high = adj_low      # NOT USED
        adj_dense = adj_low     # NOT USED

    if args.cuda:
        features = features.cuda()
        adj_low = adj_low.cuda()
        adj_high = adj_high.cuda()
        labels = labels.cuda()
        adj_dense = adj_dense.cuda()

    def test():  # isolated_mask
        model.eval()
        output = model(features, adj_low)
        output = F.log_softmax(output, dim=1)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test

    # Train model
    idx_train, idx_val, idx_test = random_disassortative_splits(
        labels, labels.max() + 1
    )

    if args.model_type in ["graff", "graffgcn"]:
        model = GRAFFNet(
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            self_loops=True, 
            step_size = args.step_size,
            model_type = args.model_type, 
            linear = args.linear,
            num_layers = args.num_layers,
            activation=args.activation,
            normalize=args.normalize
        )
    else:
        raise NotImplementedError

    if args.cuda:
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, 
        weight_decay=args.weight_decay
    )

    best_training_loss = None
    best_val_acc = 0
    best_val_loss = float("inf")
    val_loss_history = torch.zeros(args.epochs)
    best_test = 0

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_low)
            
        output = F.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
        output = model(features, adj_low)

        output = F.log_softmax(output, dim=1)
        val_loss = F.nll_loss(output[idx_val], labels[idx_val])
        val_acc = accuracy(output[idx_val], labels[idx_val])
        
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_test = test()  # isolated_mask
            best_training_loss = loss_train
            best_ckpt = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        }
        if epoch >= 0:
            val_loss_history[epoch] = val_loss.detach()
        if args.early_stopping > 0 and epoch > args.early_stopping:
            tmp = torch.mean(
                val_loss_history[epoch - args.early_stopping : epoch]
            )
            if val_loss > tmp:
                break

    best_epoch = best_ckpt['epoch']
    torch.save(best_ckpt, os.path.join(path, f'run_idx_{sample}_epoch_{best_epoch}.pt'))

    run_info["result"] = best_test
    logger.log_run(run_info)

    # Testing
    result[sample] = best_test
    del model, optimizer
    if args.cuda:
        torch.cuda.empty_cache()

    if np.mean(result) > best_result:
        record_info["result"] = np.mean(result)
        record_info["std"] = np.std(result)
        record_info["run_idx"] = sample

logger.log_record(model_info, record_info)