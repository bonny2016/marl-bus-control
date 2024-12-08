import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    """
    A Cross-Attention Layer to compute attention between queries and key-value pairs.
    """
    def __init__(self, q_dim, kv_dim, out_features, qk_proj=400):
        super(CrossAttentionLayer, self).__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.out_features = out_features
        self.qk_proj = qk_proj

        # Learnable parameters for projections
        self.Wq = nn.Parameter(torch.empty(size=(q_dim, qk_proj)))
        self.Wk = nn.Parameter(torch.empty(size=(kv_dim, qk_proj)))
        self.Wv = nn.Parameter(torch.empty(size=(kv_dim, out_features)))

        # Initialize parameters using Xavier uniform
        nn.init.xavier_uniform_(self.Wq.data, gain=1.0)
        nn.init.xavier_uniform_(self.Wk.data, gain=1.0)
        nn.init.xavier_uniform_(self.Wv.data, gain=1.0)

    def forward(self, q, kv):
        """
        Forward pass for the Cross-Attention Layer.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, q_dim).
            kv (torch.Tensor): Key-Value tensor of shape (batch_size, kv_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Project queries, keys, and values
        q_proj = torch.mm(q, self.Wq)  # Shape: (batch_size, qk_proj)
        k_proj = torch.mm(kv, self.Wk)  # Shape: (batch_size, qk_proj)
        v_proj = torch.mm(kv, self.Wv)  # Shape: (batch_size, out_features)

        # Compute attention weights
        att = torch.mm(q_proj, k_proj.T)  # Shape: (batch_size, batch_size)
        att = F.softmax(att, dim=1)  # Normalize attention weights

        # Compute weighted sum of values
        h_prime = torch.matmul(att, v_proj)  # Shape: (batch_size, out_features)
        return h_prime
