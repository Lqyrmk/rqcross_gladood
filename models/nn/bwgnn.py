import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sympy
import scipy
from torch_scatter import scatter_add

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for k in range(d+1):
            inv_coeff.append(float(coeff[d-k]))
        thetas.append(inv_coeff)
    return thetas


class PolyConv(nn.Module):
    """
    PyG-compatible PolyConv implementing:
      h = sum_k theta_k * (L_sym^k) feat
    where L_sym = I - D^{-1/2} A D^{-1/2}.
    forward(edge_index) expects edge_index with edges (src, dst).
    """
    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu, lin=False, bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias=bias) if lin else None
        self.lin = lin

    def reset_parameters(self):
        if self.linear is not None:
            nn.init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def _L_sym_mul(self, x, edge_index, deg_inv_sqrt):
        """
        compute L_sym * x = x - D^{-1/2} A D^{-1/2} x
        but implemented as: out = x - deg_inv_sqrt * (A * (x * deg_inv_sqrt))
        edge_index: [2, E] (src, dst)
        deg_inv_sqrt: [N, 1]
        """
        # normalized source features: x_src * deg_inv_sqrt[src]
        src, dst = edge_index
        # src_norm = x[src] * deg_inv_sqrt[src]
        src_norm = x[src] * deg_inv_sqrt[src]

        # aggregate sum_j A_ij * src_norm_j to dst nodes
        # out_agg[dst] = sum src_norm[src_index]
        out_agg = scatter_add(src_norm, dst, dim=0, dim_size=x.size(0))

        # result = x - deg_inv_sqrt * out_agg
        return x - deg_inv_sqrt * out_agg

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: LongTensor [2, E]
        """
        N = x.size(0)
        # degree: count incoming edges per node (assuming edges are directed src->dst)
        src, dst = edge_index
        # compute degree for dst? We need graph.in_degrees() equivalent: number of incoming edges per node.
        # We'll compute deg as scatter_add of ones on dst
        ones = torch.ones(src.size(0), device=src.device, dtype=x.dtype)
        deg = scatter_add(ones, dst, dim=0, dim_size=N)  # deg per node (in-degree)
        deg = deg.clamp(min=1.0).unsqueeze(-1)  # avoid zeros
        deg_inv_sqrt = deg.pow(-0.5)

        h_acc = self._theta[0] * x
        feat = x
        for k in range(1, self._k):
            feat = self._L_sym_mul(feat, edge_index, deg_inv_sqrt)
            h_acc = h_acc + self._theta[k] * feat

        if self.lin and (self.linear is not None):
            h_acc = self.linear(h_acc)
            h_acc = self.activation(h_acc)

        return h_acc


class BWGNN(nn.Module):
    """
    Homogeneous BWGNN for PyG.
    - `edge_index` is provided when forward called by model(x, edge_index).
    """
    def __init__(self, in_feats, h_feats, out_feats, d=2, batch=False):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList()
        # convs expect input dim = h_feats and output h_feats (as original)
        for theta in self.thetas:
            self.conv.append(PolyConv(h_feats, h_feats, theta, lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, out_feats)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, x, edge_index):
        h = self.linear(x)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_final = []
        for conv in self.conv:
            h0 = conv(h, edge_index)   # [N, h_feats]
            h_final.append(h0)
        h_cat = torch.cat(h_final, dim=-1)  # [N, h_feats * K]
        h = self.linear3(h_cat)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def testlarge(self, x, edge_index):
        # identical to forward (left for API compatibility)
        return self.forward(x, edge_index=edge_index)

    def batch(self, blocks, in_feat):
        """
        If you want to support neighbor-sampler blocks you'd need to implement block-wise degree calc.
        Placeholder to keep API compatibility with original.
        """
        # blocks: list of Data/Block objects (not implemented here)
        raise NotImplementedError("batch() is not implemented in PyG conversion. Use full-graph forward(x, edge_index).")


class BWGNN_Hetero(nn.Module):
    """
    Simplified heterogeneous support.
    Expect `edge_index_dict` to be a dict mapping relation name (tuple or string) to edge_index Tensor [2, E].
    For homogeneous fallback, pass a single edge_index under key 'default'.
    """
    def __init__(self, in_feats, h_feats, out_feats, edge_index_dict, d=2):
        super(BWGNN_Hetero, self).__init__()
        self.edge_index_dict = edge_index_dict
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        self.conv = nn.ModuleList([PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, out_feats)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index_dict=None):
        # edge_index_dict: dict {relation: edge_index}
        if edge_index_dict is None:
            edge_index_dict = self.edge_index_dict

        h = self.linear(x)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_all = []
        # For each relation, compute relation-specific output and sum them
        for rel, edge_index in edge_index_dict.items():
            h_final = []
            for conv in self.conv:
                h0 = conv(h, edge_index)
                h_final.append(h0)
            h_cat = torch.cat(h_final, dim=-1)
            h_rel = self.linear3(h_cat)
            h_all.append(h_rel)

        # sum over relations
        h_sum = torch.stack(h_all, dim=0).sum(dim=0)
        h_sum = self.act(h_sum)
        out = self.linear4(h_sum)
        return out
