import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops
from utils import add_self_loops
from tqdm import tqdm

from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits
from eval import evaluate, eval_acc, eval_rocauc, eval_f1,eval_recall_at_k,eval_w_f1
from parse import parse_method, parser_add_main_args
import dill
import warnings
warnings.filterwarnings('ignore')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Main function for running
def main(args):

    print(args)
    fix_seed(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    ### Load and preprocess data ###
    # 包含graph和 lable
    dataset = load_dataset(args)
    dataset.label = dataset.label.to(device)

    ### get the splits for all runs ###
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        for _ in range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                        for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

    print(f"训练集长度：{len(split_idx_lst[0]['train'])}")
    print(f"验证集长度：{len(split_idx_lst[0]['valid'])}")
    print(f"测试集长度：{len(split_idx_lst[0]['test'])}")

    ### Basic information of datasets ###
    n = dataset.graph['num_binodes']  # number of tokens(节点加超边的总数) tensor(12304)
    num_nodes = dataset.graph['num_nodes']  # number of nodes:3604
    e = dataset.graph['H'].shape[1]  # number of hyperedges:8661
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])  # number of class :1951
    d = dataset.graph['node_feat'].shape[1]  # 特征维度

    print(f"dataset {args.dataset} | num token {n} | num node {num_nodes} | num edge {e}| num node feats {d} | num classes {c}")

    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    dataset.graph['edge_index_bipart'] = dataset.graph['edge_index_bipart'].to(device)
    dataset.graph['H'] = dataset.graph['H'].to(device)

    ### Load method ###
    model = parse_method(args, n, num_nodes, c, d, e, device)

    # criterion = nn.NLLLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss(reduction='sum')
    ### Performance metric  ###
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
    elif args.metric == 'f1':
        eval_func = eval_f1
    elif args.metric == 'recall':
        eval_func = eval_recall_at_k
    elif args.metric == 'w_f1':
        eval_func = eval_w_f1
    else:
        eval_func = eval_acc

    logger = Logger(args.runs, args)
    model.train()
    print('MODEL:', model)

    adjs = []
    adj_bipart, _ = remove_self_loops(dataset.graph['edge_index_bipart'])
    adj_bipart, _ = add_self_loops(adj_bipart, num_nodes=dataset.graph['num_binodes'])
    adjs.append(adj_bipart)
    dataset.graph['adjs'] = adjs  # len2


    ### Training loop ###
    for run in tqdm(range(args.runs)):
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        fix_seed(args.seed)
        args.seed += 1
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        best_val = float('-inf')
        patience = 10  # 设定早停的耐心值
        counter = 0  # 计数器，记录连续未改进的次数

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            if args.use_edge_regular:
                out, link_loss_ = model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'],args.tau)
                loss0 = criterion(out[train_idx], dataset.label[train_idx])/c
                loss1 = args.lamda * sum(link_loss_) / len(link_loss_)
                loss = loss0-loss1

            else:
                out = model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'],args.tau)
                # out1 = torch.softmax(out,-1)
                loss0 = criterion(out[train_idx], dataset.label[train_idx])
                loss = loss0 / c

            loss.backward()
            optimizer.step()

            if epoch % args.eval_step == 0:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
                logger.add_result(run, result[:-1])

                if result[1] > best_val:
                    best_val = result[1]
                    counter = 0  # 验证集结果改进时重置计数器
                    if args.save_model:
                        torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')
                else:
                    counter += 1  # 未改进，计数器加1

                # 打印准确率
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%')

            # 检查早停条件
            if counter >= patience:
                print(f"Early stopping at epoch {epoch:02d}. Best validation accuracy: {100 * best_val:.2f}%")
                break

        logger.print_statistics(run)

    results = logger.print_statistics()


if __name__ == "__main__":
    # Replace this dictionary with any desired parameters
    args = argparse.Namespace(
        dataset="mimic_iii_sorted",
        rand_split=True,
        metric="recall",
        method="hypergt",
        lr=4e-3,
        weight_decay=2e-1,
        num_layers=2,
        hidden_channels=256,
        num_heads=1,
        rb_order=0,
        rb_trans="sigmoid",
        lamda=1.0,
        tau=0.2,
        M=30,
        K=10,
        use_bn=True,
        use_residual=True,
        use_gumbel= False,
        use_act=True,
        use_jk=True,
        runs=10,
        epochs= 100,
        dropout= 0.45,
        device=2,
        data_dir="./data/pyg_data/hypergraph_dataset_updated",
        pe="HEPEHtEPE",
        hefeat="mean",
        feature_noise=1.0,
        seed=2025,
        cpu=False,
        train_prop=0.7,
        valid_prop=0.1,
        eval_step=1,
        save_model=False,
        model_dir="./saved_models/",
        protocol=None,
        label_num_per_class=None,
        rand_split_class=False,
        records_length="IJCAI2025/PVHGT/data/raw_data/mimic_iii_sorted/record_lengths.pkl",
        cooccurrence="IJCAI2025/PVHGT/data/raw_data/mimic_iii_sorted/cooccurrence_matrix.csv",
        feature_dim = 128,
        use_edge_regular= False
    )
    main(args)
