import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, q_dim, kv_dim, out_features, qk_proj=400, dropout=None, alpha=None, concat=True):
        super(CrossAttentionLayer, self).__init__()
        self.dropout = dropout
        self.qk_proj = qk_proj
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.Wq = nn.Parameter(torch.empty(size=(q_dim, self.qk_proj)))
        nn.init.xavier_uniform_(self.Wq.data, gain=1.)

        self.Wk = nn.Parameter(torch.empty(size=(kv_dim, self.qk_proj)))
        nn.init.xavier_uniform_(self.Wk.data, gain=1.)

        self.Wv = nn.Parameter(torch.empty(size=(kv_dim, out_features)))
        nn.init.xavier_uniform_(self.Wv.data, gain=1.)

        self.a = nn.Parameter(torch.empty(size=(2*(self.qk_proj + self.out_features), 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, q, kv, adj=None):
        # kv = kv[1:,:]
        q_proj = torch.mm(q, self.Wq)
        k_proj = torch.mm(kv, self.Wk)
        v_proj = torch.mm(kv, self.Wv)

        att = torch.mm(q_proj, k_proj.T)
        att = F.softmax(att, dim=1)
        h_prime = torch.matmul(att, v_proj)
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'