import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .bwgnn import BWGNN
from torch_geometric.utils import subgraph
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class CROSS(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers=5,
                 pooling='mean',
                 readout='concat',
                 mask_threshold=0.4,
                 eps=0.001,
                 scalar=20,
                 explainer_model='gin',
                 explainer_layers=5,
                 explainer_readout='add',
                 **kwargs):
        super(CROSS, self).__init__()

        self.eps = eps
        self.scalar = scalar

        self.pooling = pooling
        self.readout = readout

        self.mask_threshold = mask_threshold

        self.emb = None
        self.emb_dim = num_layers * hid_dim

        self.pool = self.get_pool()

        self.bwgnn_encoder = BWGNN(in_dim, self.emb_dim, self.emb_dim)
        self.gin_encoder = GIN(in_dim, hid_dim, num_layers, pooling, readout)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def process_probability(self, node_prob, batch):
        out, _ = torch_scatter.scatter_max(torch.reshape(node_prob.detach(), (1, -1)), batch)
        out = out.reshape(-1, 1)
        out = out[batch]
        node_prob /= out + self.eps
        node_prob = (2 * node_prob - 1) / (2 * self.scalar) + 1
        return node_prob


    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        elif self.pooling == 'mean' or self.pooling == 'avg':
            pool = global_mean_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool


    def forward(self, data):

        x, edge_index, batch, num_graphs, num_nodes = data.x, data.edge_index, data.batch, data.num_graphs, data.num_nodes

        n_v_1 = self.bwgnn_encoder(x, edge_index)  # [n, md]
        g_v_1 = self.pool(n_v_1, batch)  # [g, md]
        g_v_2, n_v_2 = self.gin_encoder(x, edge_index, batch)  # [g, md], [n, md]

        return g_v_1, g_v_2

    def loss_func(self, emb_list, batch, t):

        g_v_1, g_v_2 = emb_list

        loss = self.calc_gcl_loss_g(g_v_1, g_v_2, t)

        return loss

    def score_func(self, emb_list, batch, t):

        g_v_1, g_v_2 = emb_list

        score = self.calc_gcl_loss_g(g_v_1, g_v_2, t)

        return score

    @staticmethod
    def calc_gcl_loss_n(x, x_aug, batch, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        node_belonging_mask = batch.repeat(batch_size, 1)
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature) * node_belonging_mask
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = global_mean_pool(loss, batch)

        return loss

    @staticmethod
    def calc_gcl_loss_g(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class Projector_MLP(torch.nn.Module):

    def __init__(self, in_dim, hid_dim):
        super(Projector_MLP, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hid_dim, hid_dim))

        self.mlp_aug = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hid_dim, hid_dim))

    def forward(self, x, x_aug):
        return self.mlp(x), self.mlp_aug(x_aug)


class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling='mean', readout='add'):
        super(GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.pooling = pooling
        self.readout = readout

        self.convs = torch.nn.ModuleList()
        self.dim = dim
        self.pool = self.get_pool()

        for i in range(num_gc_layers):
            if i:
                net = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                net = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(net)

            self.convs.append(conv)
        self.dense = nn.Linear(dim, dim * num_gc_layers)

    def forward(self, x, edge_index, batch):

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))

            xs.append(x)  # [l_1, l_3, l_3...], l_k: [n, d]

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)  # [g, d]
            graph_emb = self.dense(graph_emb)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)  # [g, md]
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)  # [g, d]
            graph_emb = self.dense(graph_emb)
        # graph: [g, d] or [g, md]
        # node: [n, md]
        return graph_emb, torch.cat(xs, 1)

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        elif self.pooling == 'mean' or self.pooling == 'avg':
            pool = global_mean_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool


class Explainer_MLP(torch.nn.Module):
    def __init__(self, num_features, dim, n_layers):
        super(Explainer_MLP, self).__init__()

        self.n_layers = n_layers
        self.mlps = torch.nn.ModuleList()

        for i in range(n_layers):
            if i:
                nn = Sequential(Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim))
            self.mlps.append(nn)

        self.final_mlp = Linear(dim, 1)

    def forward(self, x, edge_index, batch):

        for i in range(self.n_layers):
            x = self.mlps[i](x)
            x = F.relu(x)

        node_prob = self.final_mlp(x)
        node_prob = softmax(node_prob, batch)
        return node_prob


class Explainer_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, readout):
        super(Explainer_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.readout = readout

        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.convs.append(conv)

        hid_dim = dim * num_gc_layers if self.readout == 'concat' else dim
        self.mlp = Linear(hid_dim, 1)


    def lift_node_score_to_edge_score(self, node_score, edge_index):
        src = edge_index[0]
        dst = edge_index[1]
        src_score = node_score[src]
        dst_score = node_score[dst]
        edge_score = src_score * dst_score
        # 权重
        edge_prop_by_dst = softmax(edge_score, dst)
        # 加权求和
        new_score = scatter_add(src_score * edge_prop_by_dst, dst, dim=0, dim_size=node_score.size(0))  # [total_nodes, hidden_dim]
        # 聚合
        node_score = node_score + new_score
        return node_score, edge_score

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            if i != self.num_gc_layers - 1:
                x = self.convs[i](x, edge_index)
                x = F.relu(x)
            else:
                x = self.convs[i](x, edge_index)

            xs.append(x)  # [l_1, l_3, l_3...], l_k: [n, d]

        if self.readout == 'last':
            node_prob = xs[-1]  # [n, d]
        elif self.readout == 'concat':
            node_prob = torch.cat(xs, 1)  # [n, md]
        elif self.readout == 'add':
            node_prob = 0
            for x in xs:
                node_prob += x  # [n, d]

        node_prob = self.mlp(node_prob)  # [n, 1]
        node_prob, _ = self.lift_node_score_to_edge_score(node_prob, edge_index)
        node_prob = softmax(node_prob, batch)
        return node_prob