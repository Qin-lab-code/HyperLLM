import numpy as np
import torch
import torch.nn as nn
from util import recall_func, ndcg_func
from geoopt import ManifoldParameter
from manifolds.lorentz import Lorentz
import os
import re


class HGCF(nn.Module):
    def __init__(self, args):
        super(HGCF, self).__init__()
        self.device = args.device
        self.manifold = Lorentz(max_norm=args.max_norm)
        self.num_users, self.num_items, self.num_tags = args.num_users, args.num_items, args.num_tags
        self.num_layers, self.margin = args.num_layers, args.margin
        self.margin1, self.weight, self.t = args.margin1, args.weight, args.t
        dataset = re.sub(r'\d+', '', args.dataset)
        emb_user_path = os.path.join('save', f'{dataset}_user_{args.emb}.pt')
        emb_item_path = os.path.join('save', f'{dataset}_item_{args.emb}.pt')
        self.emb_user = nn.Embedding.from_pretrained(torch.load(emb_user_path), freeze=False)
        self.emb_user.weight = nn.Parameter(self.manifold.expmap0(self.emb_user.state_dict()['weight'], project=True))
        self.emb_user.weight = ManifoldParameter(self.emb_user.weight, self.manifold, True)
        self.emb_item = nn.Embedding.from_pretrained(torch.load(emb_item_path), freeze=False)
        self.emb_item.weight = nn.Parameter(self.manifold.expmap0(self.emb_item.state_dict()['weight'], project=True))
        self.emb_item.weight = ManifoldParameter(self.emb_item.weight, self.manifold, True)

        self.emb_tags = nn.Embedding(num_embeddings=self.num_tags, embedding_dim=self.emb_item.weight.size(1))
        self.emb_tags.state_dict()['weight'].uniform_(-args.scale, args.scale).to(self.device)
        self.emb_tags.weight = nn.Parameter(self.manifold.expmap0(self.emb_tags.state_dict()['weight'], project=True))
        self.emb_tags.weight = ManifoldParameter(self.emb_tags.weight, self.manifold, True)

    def forward(self, adj, adj_tags, triples, triples_tags):
        x_user, x_item, x_tags = self.emb_user.weight, self.emb_item.weight, self.emb_tags.weight
        t_user, t_item, t_tags = self.manifold.logmap0(x_user), self.manifold.logmap0(x_item), self.manifold.logmap0(x_tags)
        t_user_item = torch.cat([t_user, t_item], dim=0)
        loss_user_item, out_user_item = self.margin_loss(t_user_item, adj, self.margin, triples)

        t_tags_items = torch.cat([t_tags, t_item], dim=0)
        loss_tags, out_tag_item = self.margin_loss(t_tags_items, adj_tags, self.margin1, triples_tags)

        _, item_emb1 = torch.split(out_user_item, [self.num_users, self.num_items], dim=0)
        _, item_emb2 = torch.split(out_tag_item, [self.num_tags, self.num_items], dim=0)

        ssl_loss = self.contra_loss(item_emb1, item_emb2)
        loss = loss_user_item + loss_tags + self.weight * ssl_loss
        return loss

    def contra_loss(self, h1, h2):
        pos_score = -1 * self.manifold.sqdist(h1, h2)
        pos_score = torch.exp(pos_score / self.t)
        tot_score = -1 * self.manifold.sqdist_multi(h1, h2)
        tot_score = torch.exp(tot_score / self.t).sum(dim=1)
        ssl_loss = -torch.log(pos_score / tot_score).sum()
        return ssl_loss

    def margin_loss(self, h, adj, margin, triples):
        result = [h]
        for i in range(self.num_layers):
            result.append(torch.spmm(adj, result[i]))
        out = sum(result[1:])
        out = self.manifold.expmap0(out, project=True)
        anchor_embs, pos_embs, neg_embs = out[triples[:, 0], :], out[triples[:, 1], :], out[triples[:, 2], :]
        pos_scores = self.manifold.sqdist(anchor_embs, pos_embs)
        neg_scores = self.manifold.sqdist(anchor_embs, neg_embs)
        loss = pos_scores - neg_scores + margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss, out

    def encode(self, adj):
        t_user, t_item = self.emb_user.weight, self.emb_item.weight
        t_user, t_item = self.manifold.logmap0(t_user), self.manifold.logmap0(t_item)
        t_user_item = torch.cat([t_user, t_item], dim=0)
        result = [t_user_item]
        for i in range(self.num_layers):
            result.append(torch.spmm(adj, result[i]))
        out = sum(result[1:])
        return self.manifold.expmap0(out, project=True)

    def predict(self, h, train_csr, test_dict, eval_batch_num):
        arr = [10, 20]
        recall, ndcg = {}, {}
        item = h[np.arange(self.num_users, self.num_users + self.num_items), :]
        batch_size = (self.num_users // eval_batch_num) + 1
        all_probs = []
        for start in range(0, self.num_users, batch_size):
            end = min(start + batch_size, self.num_users)
            user_batch = h[np.arange(start, end), :]
            probs_batch = -1 * self.manifold.sqdist_multi(user_batch, item).detach().cpu().numpy()
            all_probs.append(probs_batch)
        probs = np.concatenate(all_probs, axis=0)
        probs[train_csr.nonzero()] = np.NINF
        ind = np.argpartition(probs, -20)[:, -20:]
        arr_ind = probs[np.arange(len(probs))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(probs)), ::-1]
        pred_list = ind[np.arange(len(probs))[:, None], arr_ind_argsort]
        all_ndcg = ndcg_func([*test_dict.values()], pred_list)
        for k in arr:
            recall[k] = recall_func(test_dict, pred_list, k)
            ndcg[k] = all_ndcg[k - 1]
        return recall, ndcg
