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

        return out, scores

class SelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        seq_len, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # [num_heads, seq_len, head_dim]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # scores: [num_heads, seq_len, seq_len]
        # graph mask
        if mask is not None:  # [seq_len, seq_len]
            mask = ~mask.unsqueeze(0)  # [1, seq_len, seq_len]
            scores = scores.masked_fill(mask, float('-inf'))

        scores = F.softmax(scores.float(), dim=-1).type_as(Q)

        out = scores @ V  # [num_heads, seq_len, head_dim]
        out = out.transpose(0, 1).contiguous()  # [seq_len, num_heads, head_dim]
        out = out.view(seq_len, -1)  # [seq_len, num_heads * head_dim]
        out = self.W_o(out)
        return out, scores


class DoubleCrossAttention(nn.Module):

    def __init__(self, d_model, device) -> None:
        super().__init__()
        self.k = torch.sqrt(torch.FloatTensor([d_model])).to(device)

        self.A_projection_network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.A_residual_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.A_w_qs = nn.Linear(d_model, d_model, bias=False)
        self.A_w_ks = nn.Linear(d_model, d_model, bias=False)
        self.A_w_vs = nn.Linear(d_model, d_model, bias=False)
        self.A_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.A_fc_ffn = nn.Linear(d_model, d_model, bias=False)

        # ------------------------------------------- view B --------------------------------------#
        self.B_projection_network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.B_residual_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.B_w_qs = nn.Linear(d_model, d_model, bias=False)
        self.B_w_ks = nn.Linear(d_model, d_model, bias=False)
        self.B_w_vs = nn.Linear(d_model, d_model, bias=False)
        self.B_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.B_fc_ffn = nn.Linear(d_model, d_model, bias=False)


    def forward(self, feat_a, feat_b):
        feat_a = feat_a.unsqueeze(0)
        feat_b = feat_b.unsqueeze(0)

        A_feat = self.A_projection_network(feat_a)
        A_residual_feat = self.A_residual_block(A_feat)

        B_feat = self.B_projection_network(feat_b)
        B_residual_feat = self.B_residual_block(B_feat)

        A_Q = self.A_w_qs(A_feat)
        A_K = self.A_w_ks(A_feat).permute(0, 2, 1)
        A_V = self.A_w_vs(A_feat)

        B_Q = self.B_w_qs(B_feat)
        B_K = self.B_w_ks(B_feat).permute(0, 2, 1)
        B_V = self.B_w_vs(B_feat)

        A_attn = A_Q @ B_K
        B_attn = B_Q @ A_K

        A_attn /= self.k
        B_attn /= self.k

        A_attn = torch.softmax(A_attn, 2)
        A_attention = A_attn / (1e-9 + A_attn.sum(dim=1, keepdims=True))
        A_sc = A_attention @ A_V

        B_attn = torch.softmax(B_attn, 2)
        B_attention = B_attn / (1e-9 + B_attn.sum(dim=1, keepdims=True))
        B_sc = B_attention @ B_V

        A_s = self.A_layer_norm(A_residual_feat + A_sc)
        A_ffn = A_s + self.A_fc_ffn(A_s)

        B_s = self.B_layer_norm(B_residual_feat + B_sc)
        B_ffn = B_s + self.B_fc_ffn(B_s)

        A_output = A_ffn
        B_output = B_ffn

        A_output = A_output.squeeze(0)
        B_output = B_output.squeeze(0)
        return A_output, B_output
