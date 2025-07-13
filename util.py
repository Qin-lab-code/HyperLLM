import random
import numpy as np
import argparse
import scipy.sparse as sp
import torch


def recall_func(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users


def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len - 1]
        dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)


def add_flags_from_config(parser, config_dict):
    def OrNone(default):
        def func(x):
            if x.lower() == "none":
                return None
            elif default is None:
                return str(x)
            else:
                return type(default)(x)
        return func
    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    parser.add_argument(f"--{param}", action="append", type=type(default[0]), default=default, help=description)
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(f"Could not add flag for param {param} because it was already present.")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
