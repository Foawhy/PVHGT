import math
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'
import dill
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from utils import SparseLinear

BIG_CONSTANT = 1e8

def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m/d) #0
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)# 每个块通过 QR 分解生成，确保是正交的。
            q = torch.t(q)# 进行转置
        block_list.append(q)
        current_seed += 1# 确保每个块是可复现的，种子加一
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])# 取前remaining行
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        # 对每一行计算其欧几里得范数（L2 范数）
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):
    del is_query
    if projection_matrix is None:
        return F.relu(data) + numerical_stabilizer
    else:
        ratio = 1.0 / torch.sqrt(
            torch.tensor(projection_matrix.shape[0], torch.float32)
        )
        data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, projection_matrix)
        return F.relu(data_dash) + numerical_stabilizer

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    # 使用比例因子对q(k)进行缩放
    data = data_normalizer * data #(1,597,1,64)
    # 计算投影比率 (ratio)：对降维后的特征进行归一化，确保特征在新维度空间的数值范围适中
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix) #(1,597,130)
    # 对输入特征 data 的每个元素平方
    diag_data = torch.square(data)
    # 最后一个维度上进行求和,即四维变成三维，(B,N,H)
    # 结果等价于计算每个节点在每个注意力头下的特征平方和
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)# 又增加一个维度(1,597,1,1)，diag_data 是对 data_dash 的平方和进行归一化
    last_dims_t = len(data_dash.shape) - 1 #得到降维特征在(B,N,H,D)中维度的索引
    attention_dims_t = len(data_dash.shape) - 3 #得到降维后特征的注意力维度在(B,N,H,D)中的索引
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )# 在指数操作 torch.exp 之前，通过减去 diag_data，避免指数值过大导致溢出。
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )# 除了找每个节点在每个头上的最大值，还要找出每个节点的特征值的最大值
    return data_dash

def numerator(qs, ks, vs):
    '''
    分别是q,k,v,分子计算
    '''
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)

def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)

def numerator_gumbel(qs, ks, vs):
    '''
    numerator:分子
    '''
    # 对于每个注意力头 ℎ 和 Gumbel 采样 k，生成了一个 [M×D] 的特征矩阵，表示键和值的组合。
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs) # kvs refers to U_k in the paper
    # 对于每个节点 n、每个注意力头 h、每个 Gumbel 采样 k，生成了加权后的值特征表示，形状为 [N,B,H,K,D]
    return torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)

def denominator_gumbel(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)

def block_softmax(attention_scores, block_size=1024):
    H, N, _ = attention_scores.shape
    attention_weights = torch.zeros_like(attention_scores)  # 初始化结果张量

    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            # 提取当前块
            block = attention_scores[:, i:i+block_size, j:j+block_size]
            # 对当前块做 softmax
            block_softmax = torch.softmax(block, dim=-1)
            # 写回结果张量
            attention_weights[:, i:i+block_size, j:j+block_size] = block_softmax

    return attention_weights


def block_matmul_cpu_to_gpu(query, key_transpose, block_size=1024):
    """
    强制在 CPU 上进行分块矩阵乘法，然后将结果转移回 GPU
    """
    query = query.to('cpu')  # 将输入数据移动到 CPU
    key_transpose = key_transpose.to('cpu')
    H, N, M = query.shape
    result_blocks = []

    for i in range(0, N, block_size):
        row_blocks = []
        for j in range(0, N, block_size):
            query_block = query[:, i:i + block_size, :]
            key_block = key_transpose[:, :, j:j + block_size]
            block_result = torch.matmul(query_block, key_block)  # 在 CPU 上计算
            row_blocks.append(block_result)
        row_result = torch.cat(row_blocks, dim=-1)
        result_blocks.append(row_result)

    # 拼接结果
    attention_scores_cpu = torch.cat(result_blocks, dim=1)

    # 将结果移动到 GPU
    attention_scores_gpu = attention_scores_cpu.to('cuda')
    return attention_scores_gpu


def kernelized_softmax(args,query, key, value, kernel_transformation, projection_matrix=None, edge_index=None, tau=0.25, return_weight=True):

    query = query / math.sqrt(tau) # 使用tau的平方根对q 进行缩放
    key = key / math.sqrt(tau) # 使用tau的平凡根对k进行缩放

    query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M],进行降维
    key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]，进行降维

    query_prime1 = query_prime.permute(1, 0, 2, 3) # 改变q的维度顺序 [N, B, H, M]
    key_prime1 = key_prime.permute(1, 0, 2, 3) # 改变k的维度顺序,[N, B, H, M]
    query = query.squeeze(0)#(N,H,M)
    key = key.squeeze(0)#(N,H,M)
    value=value.squeeze(0)

    # 将节点数 (N) 和特征维度 (M) 的位置调整
    query = query.permute(1, 0, 2)  # (H, N, M)
    key = key.permute(1, 0, 2)  # (H, N, M)
    value = value.permute(1,0,2)

    # 转置 Key 的最后两个维度
    key_transpose = key.transpose(-1, -2)  # (H, N, M) -> (H, M, N)

    #计算注意力分数
    attention_scores = torch.matmul(query, key_transpose)
    # 对注意力分数进行缩放 (Scale)
    M = query.size(-1)  # 特征维度
    attention_scores = attention_scores / math.sqrt(M)  # 缩放


    import pandas as pd
    with open (args.cooccurrence,"rb")as f:
        cooccurrence_matrix= pd.read_csv(f)

    cooccurrence_matrix=cooccurrence_matrix.to_numpy()
    cooccurrence_matrix = cooccurrence_matrix / cooccurrence_matrix.max()  # 最大值归一化到 [0, 1]
    cooccurrence_matrix = torch.tensor(cooccurrence_matrix, dtype=torch.float32)

    cooccurrence_matrix = cooccurrence_matrix.unsqueeze(0)
    # 提取 Attention Scores 中节点与节点的部分
    num_nodes = cooccurrence_matrix.size(1)  # 节点数

    node_attention_scores = attention_scores[:, :num_nodes, :num_nodes]  # (1, 节点数, 节点数)
    # 对节点部分的注意力分数进行归一化
    node_attention_scores = torch.softmax(node_attention_scores, dim=-1)  # 归一化到 [0, 1]

    # 将共现矩阵与注意力分数相加
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")  # 确认设备
    cooccurrence_matrix = cooccurrence_matrix.to(device)  # 将张量移动到指定设备

    weighted_node_attention = node_attention_scores + cooccurrence_matrix  # (1, 节点数, 节点数)
    # 提取其他部分
    other_parts = attention_scores[:, num_nodes:, :]  # 取出其他部分

    # 拼接新张量
    new_attention_scores = torch.cat([
        torch.cat([weighted_node_attention, attention_scores[:, :num_nodes, num_nodes:]], dim=-1),
        other_parts
    ], dim=-2)


    # 对全局 Attention Scores 做归一化 (Softmax)
    attention_weights = torch.softmax(new_attention_scores, dim=-1)
    # attention_weights = torch.softmax(attention_scores, dim=-1)

    z_output = torch.matmul(attention_weights, value)
    z_output =  z_output.permute(1,0,2)
    z_output = z_output.unsqueeze(0)
    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_end, key_start = query_prime1[end], key_prime1[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime1, key_prime1) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

        return z_output, A_weight

    else:
        return z_output


def kernelized_gumbel_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None,
                                K=10, tau=0.25, return_weight=False):
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]

    gumbels = (
        -torch.empty(key_prime.shape[:-1]+(K, ), memory_format=torch.legacy_contiguous_format).exponential_().log()
    ).to(query.device) / tau # [N, B, H, K]

                   #(N,B,H,1,M)             #(N,B,H,K,1)
    # 得到的特征表示在 K个 不同噪声扰乱下得到的节点特征
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4) # [N, B, H, K, M]
    # 注意力机制的分子
    z_num = numerator_gumbel(query_prime, key_t_gumbel, value) # [N, B, H, K, D]
    # 注意力机制的分母
    z_den = denominator_gumbel(query_prime, key_t_gumbel) # [N, B, H, K]

    z_num = z_num.permute(1, 0, 2, 3, 4) # [B, N, H, K, D]
    z_den = z_den.permute(1, 0, 2, 3) # [B, N, H, K]

    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = torch.mean(z_num / z_den, dim=3) # [B, N, H, D]

    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

        return z_output, A_weight

    else:
        return z_output

def add_conv_relational_bias(x, edge_index, b, trans='sigmoid'):
    row, col = edge_index
    d_in = degree(col, x.shape[1]).float()
    d_norm_in = (1. / d_in[col]).sqrt()
    d_out = degree(row, x.shape[1]).float()
    d_norm_out = (1. / d_out[row]).sqrt()
    conv_output = []
    for i in range(x.shape[2]):
        if trans == 'sigmoid':
            b_i = F.sigmoid(b[i])
        elif trans == 'identity':
            b_i = b[i]
        else:
            raise NotImplementedError
        value = torch.ones_like(row) * b_i * d_norm_in * d_norm_out
        adj_i = SparseTensor(row=col, col=row, value=value, sparse_sizes=(x.shape[1], x.shape[1]))
        conv_output.append( matmul(adj_i, x[:, :, i]) )  # [B, N, D]
    conv_output = torch.stack(conv_output, dim=2) # [B, N, H, D]
    return conv_output

class NodeFormerConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, kernel_transformation=softmax_kernel_transformation, projection_matrix_type='a',
                 nb_random_features=10, use_gumbel=True, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=True):
        super(NodeFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)# output层

        if rb_order >= 1:
            # 是 Relational Bias（关系偏置）的参数，通常用于表示多阶邻接关系的权重。
            # 每个注意力头、每阶邻接关系都有独立的偏置值
            self.b = torch.nn.Parameter(torch.FloatTensor(rb_order, num_heads), requires_grad=True)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.projection_matrix_type = projection_matrix_type
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        # Gumbel softmax 能在训练阶段增强模型的探索能力。
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order
        # # relation_bias_transformer的缩写，是一种指定如何处理 self.b 的规则（比如通过 sigmoid 归一化）
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.Wk.weight)  # Xavier 初始化
        nn.init.zeros_(self.Wk.bias)  # 偏置初始化为 0

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.zeros_(self.Wq.bias)

        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.zeros_(self.Wv.bias)

        nn.init.xavier_uniform_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)

        if self.rb_order >= 1:
            if self.rb_trans == 'sigmoid':
                torch.nn.init.constant_(self.b, 0.1)#
            elif self.rb_trans == 'identity':
                torch.nn.init.constant_(self.b, 1.0)

    def forward(self, args, z, adjs, tau):
        # 获取 batch和节点数
        B, N = z.size(0), z.size(1)
        query = self.Wq(z).reshape(-1, N, self.num_heads, self.out_channels)
        key = self.Wk(z).reshape(-1, N, self.num_heads, self.out_channels)
        value = self.Wv(z).reshape(-1, N, self.num_heads, self.out_channels)

        # 使用投影函数对q进行降维，方便计算内积
        if self.projection_matrix_type is None:
            projection_matrix = None
        else:
            dim = query.shape[-1]
            seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
            projection_matrix = create_projection_matrix(
                self.nb_random_features, dim, seed=seed).to(query.device)

        if self.use_gumbel and self.training:  # 确保只在训练时使用 Gumbel 噪声
            if self.use_edge_loss:
                z_next, weight = kernelized_gumbel_softmax(query,key,value,self.kernel_transformation,projection_matrix,adjs[0],
                                                    self.nb_gumbel_sample, tau, self.use_edge_loss)
            else:
                z_next = kernelized_gumbel_softmax(query,key,value,self.kernel_transformation,projection_matrix,adjs[0],
                                                    self.nb_gumbel_sample, tau, self.use_edge_loss)
        else:
            if self.use_edge_loss:
                z_next, weight = kernelized_softmax(args,query, key, value, self.kernel_transformation, projection_matrix, adjs[0],
                                                    tau, self.use_edge_loss)
            else:
                z_next = kernelized_softmax(args,query, key, value, self.kernel_transformation, projection_matrix,
                                                    adjs[0], tau, self.use_edge_loss)
       
        # compute update by relational bias of input adjacency, requires O(E)
        for i in range(self.rb_order):
            z_next += add_conv_relational_bias(value, adjs[i], self.b[i], self.rb_trans)

        # aggregate results of multiple heads
        z_next = self.Wo(z_next.flatten(-2, -1))

        if self.use_edge_loss: # compute edge regularization loss on input adjacency
            row, col = adjs[0]
            d_in = degree(col, query.shape[1]).float()
            d_norm = 1. / d_in[col]
            d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, weight.shape[-1])
            link_loss = torch.mean(weight.log() * d_norm_)
            return z_next, link_loss
        else:
            return z_next

        
class HyperGT(nn.Module):
    def __init__(self,num_tokens, num_nodes, in_channels, hidden_channels, out_channels, num_hes, num_layers=2, num_heads=4, dropout=0.0,
                 kernel_transformation=softmax_kernel_transformation, nb_random_features=30, use_bn=True, use_gumbel=True,
                 use_residual=True, use_act=False, use_jk=False, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=False):
        super(HyperGT, self).__init__()

        self.convs = nn.ModuleList()#这行代码创建了一个空的 nn.ModuleList，通常用于在类的构造函数中初始化动态网络的子模块列表。
        self.fcs = nn.ModuleList()
        self.classfier=nn.Softmax(dim=-1)
        # self.classfier=nn.Sigmoid()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))# 添加一个线性层
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))#添加一个LN层
        # self.activation=torch.nn.Sigmoid()
        for i in range(num_layers):
            self.convs.append(
                NodeFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel_transformation=kernel_transformation,
                              nb_random_features=nb_random_features, use_gumbel=use_gumbel, nb_gumbel_sample=nb_gumbel_sample,
                               rb_order=rb_order, rb_trans=rb_trans, use_edge_loss=use_edge_loss))
            self.bns.append(nn.LayerNorm(hidden_channels))

        if use_jk:
            self.fcs.append(nn.Linear(hidden_channels * num_layers + hidden_channels, out_channels))
        else:
            self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.use_jk = use_jk
        self.use_edge_loss = use_edge_loss
        self.n = num_tokens
        # 用于对超图中的超边特征和节点特征生成位置编码
        self.he_sparse_encoder = SparseLinear(num_hes, hidden_channels)
        self.hte_sparse_encoder = SparseLinear(num_nodes, hidden_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if hasattr(bn, 'weight') and bn.weight is not None:
                nn.init.ones_(bn.weight)  # 将 weight 初始化为全1
            if hasattr(bn, 'bias') and bn.bias is not None:
                nn.init.zeros_(bn.bias)  # 将 bias 初始化为全0
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
        nn.init.xavier_uniform_(self.he_sparse_encoder.weight)
        nn.init.zeros_(self.he_sparse_encoder.bias)
        nn.init.xavier_uniform_(self.hte_sparse_encoder.weight)
        nn.init.zeros_(self.hte_sparse_encoder.bias)

    def forward(self, args, x, adjs, H, tau=1.0):
        x = x.unsqueeze(0) # (1,597,100)
        layer_ = []# 用来保存每一层的输出
        link_loss_=[]
        # 经过第一层的线性变化
        z = self.fcs[0](x)
        he_pe = self.he_sparse_encoder(H).unsqueeze(0)#(1,282,64),节点的位置编码特征
        hte_pe = self.hte_sparse_encoder(H.transpose(0,1)).unsqueeze(0)#(1,315,64)，超边的位置编码特征
        if  'HEPE' in args.pe:# 使用节点的位置编码
            #(1,315,64)，得到全零的超边特征
            padding=torch.zeros(z.shape[0],z.shape[1]-he_pe.shape[1],z.shape[2],requires_grad=False,device=z.device)
            he_pe = torch.cat((he_pe,padding),dim=1)#(1,597,64)
            z = z + he_pe
        if 'HtEPE' in args.pe:# 使用超边的位置编码
            # 得到全零的节点特征
            padding=torch.zeros(z.shape[0],z.shape[1]-hte_pe.shape[1],z.shape[2],requires_grad=False,device=z.device)
            hte_pe = torch.cat((padding,hte_pe),dim=1)
            # 上一步加了位置编码的节点特征，和原来的超边特征加上稀疏编码器的特征形成现在的超边特征
            z = z + hte_pe
        
        if self.use_bn:
            z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        layer_.append(z)# 得到了初步的加了位置编码的节点特征和超边特征

        for i, conv in enumerate(self.convs):
            if self.use_edge_loss:
                z, link_loss, = conv(args, z, adjs, tau)
                link_loss_.append(link_loss)
            else:
                z = conv(args, z, adjs, tau)
            if self.use_residual:# 如果使用残差
                z += layer_[i]
            if self.use_bn:
                z = self.bns[i+1](z)
            if self.use_act:
                z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)# 把每一层卷积后的全部特征矩阵都保存在里面

        if self.use_jk: # use jk connection for each layer
            z = torch.cat(layer_, dim=-1)# 按特征的维度拼接(597,4*hidden)

        # 解析超边嵌入
        num_hyperedges = H.shape[1]
        hyperedge_embeddings = z.squeeze(0)[-num_hyperedges:]  # 超边嵌入部分 (num_hyperedges, hidden_dim)
        # 根据 record_lengths 计算每个患者的嵌入
        patient_embeddings = []
        start_idx = 0
        record_length_filepath=args.records_length
        with open(record_length_filepath,"rb")as file :
            record_lengths=dill.load(file)
        for num_visits in record_lengths:
            end_idx = start_idx + num_visits
            # patient_embed = hyperedge_embeddings[start_idx:end_idx].sum(dim=0)  # 每个患者的住院嵌入相加
            patient_embed = hyperedge_embeddings[start_idx:end_idx].mean(dim=0)  # 平均池化代替求和
            patient_embeddings.append(patient_embed)
            start_idx = end_idx
        patient_embeddings = torch.stack(patient_embeddings, dim=0)  # (num_patients, hidden_dim)

        # 使用患者嵌入进行诊断分类,得到logit
        x_out = self.fcs[-1](patient_embeddings)  # (num_patients, out_channels)
        x_out = self.classfier(x_out)
        if self.use_edge_loss:
            return x_out, link_loss_
        else:
            return x_out
