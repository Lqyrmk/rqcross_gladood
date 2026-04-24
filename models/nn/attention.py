import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from math import inf

class CrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(0.4)

    def forward(self, xq, xk, xv, mask=None):

        N, _ = xq.shape
        M, _ = xk.shape

        Q = self.W_q(xq)
        K = self.W_k(xk)
        V = self.W_v(xv)

        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(M, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(M, self.num_heads, self.head_dim).transpose(0, 1)

        # Q: [H, N, D], KV: [H, M, D]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [H, N, M]

        scores = F.softmax(scores, dim=-1)  # [H, N, M]

        out = scores @ V  # [H, N, D]
        out = out.transpose(0, 1).contiguous()  # [N, H, D]
        out = out.view(N, -1)

        # out = self.W_o(out)
        # out = self.dropout(self.bn(out))

        return out
