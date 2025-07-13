import pickle
import argparse
import numpy as np
import torch
from scipy.sparse import load_npz
from model import HGCF
from sampler import WarpSampler, WarpSampler_Tag
from util import add_flags_from_config, set_seed, normalize, sparse_mx_to_torch_sparse_tensor
from logger import *
from datetime import datetime
from config import config_args
import scipy.sparse as sp
from optim import RiemannianSGD
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_flags_from_config(parser, config_args)
    args = parser.parse_args()
    dataset = args.dataset
    data_root = os.path.join("data", dataset)
    args.data_root = data_root
    with open(os.path.join(data_root, "meta.pkl"), "rb") as f:
        meta_data = pickle.load(f)
    num_users, num_items = meta_data["num_users"], meta_data["num_items"]
    args.num_users, args.num_items = num_users, num_items

    now = datetime.now().strftime('%m-%d_%H-%M-%S')
    logger = Logger(args.dataset, now)
    logger.write(f"dataset: {dataset} num_users: {num_users} num_items: {num_items}")
    for arg in vars(args):
        logger.write(arg + '=' + str(getattr(args, arg)) + '\n')
    set_seed(args.seed)

    train_mat_path = os.path.join(data_root, "train_matrix.npz")
    train_mat = load_npz(train_mat_path)
    val_mat_path = os.path.join(data_root, "val_matrix.npz")
    val_mat = load_npz(val_mat_path)
    test_mat_path = os.path.join(data_root, "test_matrix.npz")
    test_mat = load_npz(test_mat_path)
    val_dict_path = os.path.join(data_root, "val_dict.pkl")
    test_dict_path = os.path.join(data_root, "test_dict.pkl")
    if os.path.exists(val_dict_path):
        with open(val_dict_path, 'rb') as f:
            val_dict = pickle.load(f)
    else:
        val_dict = {}
        for user_id in range(val_mat.shape[0]):
            item_indices = val_mat[user_id].nonzero()[1]
            val_dict[user_id] = item_indices.tolist()
        with open(val_dict_path, 'wb') as f:
            pickle.dump(val_dict, f)

    if os.path.exists(test_dict_path):
        with open(test_dict_path, 'rb') as f:
            test_dict = pickle.load(f)
    else:
        test_dict = {}
        for user_id in range(test_mat.shape[0]):
            item_indices = test_mat[user_id].nonzero()[1]
            test_dict[user_id] = item_indices.tolist()
        with open(test_dict_path, 'wb') as f:
            pickle.dump(test_dict, f)

    train_mat_coo = train_mat.tocoo()
    rows = np.concatenate((train_mat_coo.row, train_mat_coo.transpose().row + num_users))
    cols = np.concatenate((train_mat_coo.col + num_users, train_mat_coo.transpose().col))
    data = np.ones((train_mat_coo.nnz * 2,))
    train_adj = sp.coo_matrix((data, (rows, cols))).tocsr().astype(np.float32)
    train_adj_norm = sparse_mx_to_torch_sparse_tensor(normalize(train_adj + sp.eye(train_adj.shape[0]))).coalesce().to(args.device)
    sampler = WarpSampler((num_users, num_items), train_adj, args.batch_size, 1)

    num_pairs = train_adj.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1

    with open(os.path.join(data_root, 'id_to_tag.json'), 'r', encoding='utf-8') as f:
        id_to_tags = json.load(f)
    num_tags = len(id_to_tags)
    args.num_tags = num_tags

    path_adj_tag_tag = os.path.join(data_root, 'train_adj_tag_tag.npz')
    path_adj_tag_item = os.path.join(data_root, 'train_adj_tag_item.npz')
    path_adj_tags = os.path.join(data_root, 'train_adj_tags.npz')
    if os.path.exists(path_adj_tag_tag):
        train_adj_tag_tag = sp.load_npz(path_adj_tag_tag)
        train_adj_tag_item = sp.load_npz(path_adj_tag_item)
        train_adj_tags = sp.load_npz(path_adj_tags)
    else:
        with open(os.path.join(data_root, 'tag_to_tag.json'), 'r', encoding='utf-8') as f:
            tag_to_tag = json.load(f)
        edges_tag_tag = []
        for tag1, tags in tag_to_tag.items():
            for tag2 in tags:
                edges_tag_tag.append((int(tag1), tag2))
                edges_tag_tag.append((tag2, int(tag1)))
        rows_tag_tag = np.array([edge[0] for edge in edges_tag_tag])
        cols_tag_tag = np.array([edge[1] for edge in edges_tag_tag])
        data_tag_tag = np.ones((len(rows_tag_tag),))
        train_adj_tag_tag = sp.coo_matrix((data_tag_tag, (rows_tag_tag, cols_tag_tag)),
                                          shape=(num_tags, num_tags)).tocsr().astype(np.float32)
        train_adj_tag_tag.setdiag(0)
        train_adj_tag_tag.eliminate_zeros()
        sp.save_npz(path_adj_tag_tag, train_adj_tag_tag)
        with open(os.path.join(data_root, 'tag_to_item.json'), 'r', encoding='utf-8') as f:
            tag_to_item = json.load(f)
        edges_tag_item = []
        for tag, items in tag_to_item.items():
            for item in items:
                edges_tag_item.append((int(tag), num_tags + item))
                edges_tag_item.append((num_tags + item, int(tag)))
        rows_tag_item = np.array([edge[0] for edge in edges_tag_item])
        cols_tag_item = np.array([edge[1] for edge in edges_tag_item])
        data_tag_item = np.ones((len(rows_tag_item),))
        train_adj_tag_item = sp.coo_matrix((data_tag_item, (rows_tag_item, cols_tag_item)),
                                           shape=(num_tags + num_items, num_tags + num_items)).tocsr().astype(np.float32)
        sp.save_npz(path_adj_tag_item, train_adj_tag_item)
        edges = []
        edges.extend(edges_tag_tag)
        edges.extend(edges_tag_item)
        rows_tags = np.array([edge[0] for edge in edges])
        cols_tags = np.array([edge[1] for edge in edges])
        data_tags = np.ones((len(rows_tags),))
        train_adj_tags = sp.coo_matrix((data_tags, (rows_tags, cols_tags)),
                                       shape=(num_tags + num_items, num_tags + num_items)).tocsr().astype(np.float32)
        sp.save_npz(path_adj_tags, train_adj_tags)
    count = train_adj_tag_tag.count_nonzero()
    num_pairs_tag_tag = count // num_batches
    if count % num_batches != 0:
        num_pairs_tag_tag += 1
    sampler_tag_tag = WarpSampler_Tag(num_tags, train_adj_tag_tag, num_pairs_tag_tag, 1)
    count = train_adj_tag_item.count_nonzero()
    num_pairs_tag_item = count // num_batches
    if count % num_batches != 0:
        num_pairs_tag_item += 1
    sampler_tag_item = WarpSampler((num_tags, num_items), train_adj_tag_item, num_pairs_tag_item, 1)
    train_adj_norm_tags = sparse_mx_to_torch_sparse_tensor(normalize(
        train_adj_tags + sp.eye(train_adj_tags.shape[0]))).coalesce().to(args.device)

    model = HGCF(args).to(args.device)
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    best = 0.0
    stop_cnt = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        avg_loss = 0.
        for batch in range(num_batches):
            triples = sampler.next_batch()
            triples_tag_tag = sampler_tag_tag.next_batch()
            triples_tag_item = sampler_tag_item.next_batch()
            triples_tag = np.vstack((triples_tag_tag, triples_tag_item))
            optimizer.zero_grad()
            train_loss = model(train_adj_norm, train_adj_norm_tags, triples, triples_tag)
            if torch.isnan(train_loss):
                sampler.close()
                sampler_tag_tag.close()
                sampler_tag_item.close()
                os._exit(0)
            train_loss.backward()
            optimizer.step()
            avg_loss += train_loss / num_batches
        logger.write(f"Epoch {epoch} - Loss: {avg_loss}")
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(train_adj_norm)
            r, n = model.predict(embeddings, train_mat, val_dict, args.eval_batch_num)
            logger.write("Val:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(r[10], r[20], n[10], n[20]))
            score = r[10] + r[20] + n[10] + n[20]
            if score >= best:
                stop_cnt = 0
                best = score
                r, n = model.predict(embeddings, train_mat, test_dict, args.eval_batch_num)
                logger.write("Test:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(r[10], r[20], n[10], n[20]))
            else:
                stop_cnt += 1
            if stop_cnt >= 50:
                break
    sampler.close()
    sampler_tag_tag.close()
    sampler_tag_item.close()
    exit()
