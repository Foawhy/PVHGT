import torch
import os
from data_utils import rand_train_test_idx, class_rand_splits
from utils import ExtractV2E, ConstructH
from convert_datasets_to_pygDataset import dataset_Hypergraph
from tqdm import tqdm

class NCDataset(object):
    def __init__(self, name):
        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}# 存储图结构信息的字典，比如节点数，节点特征，边数，边的特征等等
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=0.5, valid_prop=0.25, label_num_per_class=20):

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_dataset(args):
    if args.dataset in ('mimic_iii', 'mimic_iv','mimic_iii_sorted',"mimic_iv_version9_sorted","mimic_iv_version9_sorted_10000"):
        dataset = load_AllSet_dataset(args, feature_noise=args.feature_noise)
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_AllSet_dataset(args, feature_noise=None):
    name = args.dataset
    if name not in ['mimic_iii', 'mimic_iv','mimic_iii_sorted',"mimic_iv_version9_sorted","mimic_iv_version9_sorted_10000"]:#name就是传入的数据集的名字
        feature_noise = None
    p2raw = '../data/raw_data/'# 存储原始数据的位置
    # data_dir: ../data
    dataset = dataset_Hypergraph(args=args,name=name,root = args.data_dir, feature_noise=feature_noise,p2raw=p2raw)

    data = dataset.data

    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])

    if not hasattr(data, 'num_hyperedges'):
        data.num_hyperedges = torch.tensor([data.edge_index[0].max()-data.n_x[0]+1])

    print(f"data.n_x: {data.n_x}, type: {type(data.n_x)}")
    # num_nodes = data.n_x[0]
    num_nodes = data.n_x
    # num_hyperedges = data.num_hyperedges[0].to(int)
    # num_hyperedges = data.num_hyperedges[0]
    num_hyperedges = data.num_hyperedges

    edge_index = data.edge_index#[V|E;E|V]# 2,10816
    node_feat = data.x
    label = data.y

    V2E = ExtractV2E(edge_index,num_nodes,num_hyperedges)# (2,5408)只包含节点到超边的索引
    H= ConstructH(V2E,num_nodes)

    he_feat = torch.zeros(num_hyperedges, node_feat.shape[1],requires_grad=False)# 超边特征？

    if args.hefeat =='mean':
        for i in tqdm(range(num_hyperedges)):
            he_feat[i] = torch.mean(node_feat[V2E[0,V2E[1,:]==i+num_nodes]],dim=0)
        if not os.path.exists('he_feat'):
            os.makedirs('he_feat')
        torch.save(he_feat,'he_feat/he_feat_mean_'+name+'.pt')

    elif args.hefeat =='rand':
        he_feat = torch.rand(num_hyperedges, node_feat.shape[1],requires_grad=True)
    elif args.hefeat =='zero':
        pass
    elif args.hefeat =='load':
        he_feat=torch.load('he_feat/he_feat_mean_'+name+'.pt')
    else:
        raise ValueError('Invalid hyperedge feature type')
    
    node_feat = torch.cat([node_feat, he_feat], dim=0)# 把节点特征和超边特征按行拼接起来

    # 实例化一个NCDataset对象
    dataset = NCDataset(name)
    # 把前面得到的节点和边的信息都加上
    dataset.graph = {'edge_index_bipart': V2E,
                     'H': H.coalesce(),
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_binodes': data.edge_index[0].max()+1,# 二部图中所有的节点
                     'num_hyperedges': num_hyperedges,
                     'num_nodes': num_nodes}
    print(f"'num_binodes': {data.edge_index[0].max() + 1}")
    dataset.label = label
    return dataset
