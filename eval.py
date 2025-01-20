import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def eval_recall_at_k(y_true, y_pred, k=10):
    """
    计算多标签分类任务的 R@k (Recall at k)。

    Args:
        y_true (torch.Tensor): 真实标签，形状为 (num_samples, num_classes)，值为 0 或 1。
        y_pred (torch.Tensor): 模型预测的 logits，形状为 (num_samples, num_classes)。
        k (int): 选择概率前 k 大的标签。

    Returns:
        float: 平均 R@k 值。
    """
    # 将张量转换为 NumPy 格式
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # 存储每个样本的 R@k
    recall_list = []
    for true_labels, pred_logits in zip(y_true, y_pred):
        # 获取前 k 个预测标签的索引
        top_k_indices = np.argsort(pred_logits)[-k:]  # 按概率从小到大排序，选取后 k 个
        # 计算真实标签与预测标签的交集数量
        true_positive = np.sum(true_labels[top_k_indices])# 预测对的个数
        # 计算该样本的召回率
        possible_positive = np.sum(true_labels)# 总共有多少个类别
        recall = true_positive / possible_positive if possible_positive > 0 else 0.0
        recall_list.append(recall)
    # 返回平均召回率
    return np.mean(recall_list)


def eval_w_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    import numpy as np

    # Convert tensors to numpy arrays
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # Sort predictions by descending order of scores
    y_pred_sorted = np.argsort(y_pred, axis=-1)[:, ::-1]  # Sort indices by descending order

    # Create an empty result matrix with the same shape as y_true
    result = np.zeros_like(y_true)

    # Populate the result matrix with top predictions based on the number of true labels
    for i in range(len(result)):
        true_number = np.sum(y_true[i] == 1)  # Number of true labels for this sample
        result[i][y_pred_sorted[i][:true_number]] = 1  # Select top true_number predictions

    # Compute weighted F1 score
    return f1_score(y_true=y_true, y_pred=result, average='weighted', zero_division=0)

def eval_f1(y_true, y_pred, threshold=0.5, average='micro'):
    # 将张量转换为 NumPy 数组
    y_true = y_true.detach().cpu().numpy()
    y_pred = (y_pred.sigmoid().detach().cpu().numpy() > threshold).astype(int)  # 转换为二进制分类结果

    # 计算多标签 F1 分数
    f1 = f1_score(y_true, y_pred, average=average)  # 使用 sklearn 的 F1 计算函数

    return f1


def eval_acc(y_true, y_pred,threshold=0.5):
    y_true = y_true.detach().cpu().numpy()
    y_pred = (y_pred.detach().cpu().numpy() > threshold).astype(int)

    acc_list = []
    for i in range(y_true.shape[0]):
        correct = np.sum(y_true[i] == y_pred[i])  # Number of correct labels per sample
        acc_list.append(correct / y_true.shape[1])  # Accuracy per sample

    return np.mean(acc_list)  # Average accuracy


def eval_rocauc(y_true, y_pred):
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args):
    model.eval()
    # Forward pass
    if args.method == 'hypergt':
        if args.use_edge_regular:
            out, _ = model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'], args.tau)
        else:
            out = model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'], args.tau)
    else:
        out = model(dataset)

    # Calculate validation loss (if needed, currently set to 0)
    valid_loss = 0

    if args.metric == "recall":
        # Calculate multi-label recall for train, validation, and test sets
        train_eva = eval_recall_at_k(dataset.label[split_idx['train']], out[split_idx['train']], 10)
        valid_eva = eval_recall_at_k(dataset.label[split_idx['valid']], out[split_idx['valid']], 10)
        test_eva = eval_recall_at_k(dataset.label[split_idx['test']], out[split_idx['test']], 10)
    else:
        # Calculate W-F1 for train, validation, and test sets
        train_eva = eval_w_f1(dataset.label[split_idx['train']], out[split_idx['train']])
        valid_eva = eval_w_f1(dataset.label[split_idx['valid']], out[split_idx['valid']])
        test_eva = eval_w_f1(dataset.label[split_idx['test']], out[split_idx['test']])

    return train_eva, valid_eva, test_eva, valid_loss, out



@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, result=None):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs_, x = dataset.graph['adjs'], dataset.graph['node_feat']
    adjs = []
    adjs.append(adjs_[0])
    for k in range(args.rb_order - 1):
        adjs.append(adjs_[k + 1])
    out, _ = model(x, adjs)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out
