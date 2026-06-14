import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .gin import GIN
from .bwgnn import BWGNN
from .attention import CrossAttention, SelfAttention
from .moe import MMoE
from .vqvae import VectorQuantizerEMA
from torch_geometric.utils import subgraph
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class CROSS(nn.Module):

    def __init__(self, config):
        super(CROSS, self).__init__()

        in_dim = config.in_dim
        in_str_dim = config.in_str_dim
        hid_dim = config.hid_dim
        num_layers = config.num_layers
        num_heads = config.num_heads
        num_experts = config.num_experts

        self.k = config.k
        self.eps = config.eps
        self.scalar = config.scalar
        self.pooling = config.pooling
        self.readout = config.readout

        self.emb_dim = num_layers * hid_dim

        self.bwgnn_encoder = BWGNN(in_dim, self.emb_dim, self.emb_dim)
        self.gin_encoder = GIN(in_dim, hid_dim, num_layers, self.pooling, self.readout)

        self.graph_dense_layers = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim),
                                                 nn.Linear(self.emb_dim, self.emb_dim)])
        self.extractor = Explainer_GIN(self.emb_dim, self.emb_dim, num_layers, self.readout)

        self.prototype_codebooks = nn.Parameter(torch.rand(self.k, self.emb_dim))

        self.cross_attn_low = CrossAttention(d_model=self.emb_dim, num_heads=num_heads)
        self.cross_attn_high = CrossAttention(d_model=self.emb_dim, num_heads=num_heads)

        self.in_self_attn = SelfAttention(d_model=in_dim, num_heads=1)
        self.shared_self_attn = SelfAttention(d_model=self.emb_dim, num_heads=1)
        self.self_attn_low = SelfAttention(d_model=self.emb_dim, num_heads=1)
        self.self_attn_high = SelfAttention(d_model=self.emb_dim, num_heads=1)

        self.moe = MMoE(in_dim=in_dim,
                        in_str_dim=in_str_dim,
                        num_experts=num_experts,
                        expert_dim=hid_dim,
                        num_tasks=2,
                        task_dim=self.emb_dim,
                        hid_dim=hid_dim,
                        out_dim=self.emb_dim,
                        gnn_layers=num_layers)

        self.feat_norm = nn.BatchNorm1d(in_dim)
        self.str_norm = nn.BatchNorm1d(in_str_dim)

        self.norm_low = nn.BatchNorm1d(self.emb_dim)
        self.norm_high = nn.BatchNorm1d(self.emb_dim)

        self.proto_proj_low = nn.Linear(self.emb_dim, self.emb_dim)
        self.proto_proj_high = nn.Linear(self.emb_dim, self.emb_dim)

        self.low_dropout = nn.Dropout(0.5)
        self.high_dropout = nn.Dropout(0.5)

        self.attn_pool_proj = nn.Linear(self.emb_dim, self.emb_dim)

        # self.decoder = GIN(self.emb_dim, hid_dim, num_layers, pooling, readout)
        self.dec_proj_high = nn.Linear(self.emb_dim, self.emb_dim)
        self.dec_proj_low = nn.Linear(self.emb_dim, self.emb_dim)
        self.dec_proj_layers = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim),
                                              nn.Linear(self.emb_dim, self.emb_dim)])

        # VQ
        commitment_cost = config.commitment_cost
        vq_decay = config.vq_decay
        vq_epsilon = config.vq_epsilon
        self.vq = VectorQuantizerEMA(num_embeddings=self.k,
                                     embedding_dim=self.emb_dim,
                                     commitment_cost=commitment_cost,
                                     decay=vq_decay,
                                     epsilon=vq_epsilon)

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

    def allocate_prototype(self, x):
        # x: [N, D]
        # codebook: [K, D]

        # 1.基于内积相似度
        sim = x @ self.prototype_codebooks.T  # [N, K]
        labels = sim.argmax(dim=1)  # [N,]
        # 2.基于 L2 距离
        # distances = (
        #     torch.sum(x ** 2, dim=1, keepdim=True)  # [N, 1]
        #     + torch.sum(self.prototype_codebooks ** 2, dim=1)  # [K,]
        #     - 2 * torch.matmul(x, self.prototype_codebooks.t())  # [N, K]
        # )
        # labels = torch.argmin(distances, dim=1)  # [N,]

        # 计算利用率
        # encodings = F.one_hot(labels, self.k).type(x.dtype)  # [N, K]
        # avg_probs = encodings.mean(dim=0)  # [K,] 利用率
        # print(f"当前 batch 码本利用率：{[(str(p.item() * 100) + '%') for p in avg_probs]}")

        # 返回原型
        p = self.prototype_codebooks[labels]  # [N, D]
        return p

    def intra_graph_mask(self, batch):
        # 1. [N] -> [N, 1] unsqueeze(1)
        # 2. [N] -> [1, N] unsqueeze(0)
        mask = batch.unsqueeze(1) == batch.unsqueeze(0)
        # print(f"mask = {mask.shape}")
        return mask

    def self_attn_pool(self, nx, batch):
        # attn_x, _ = self.self_attn_low(nx, self.intra_graph_mask(batch))
        attn_x, _ = self.shared_self_attn(nx)
        nx = nx + attn_x
        gx = self.attn_pool_proj(global_mean_pool(nx, batch))
        return gx

    def node_masking(self, x, node_prob, batch):
        mask = self.process_probability(node_prob, batch)
        return x * mask

    def forward(self, data):

        x, x_s, edge_index, batch = data.x, data.x_s, data.edge_index, data.batch
        x, x_s = self.feat_norm(x), self.str_norm(x_s)

        # nh = self.bwgnn_encoder(x, edge_index)  # [n, md]
        # gh = self.pool(nh, batch)  # [g, md]
        # gl, nl = self.gin_encoder(x, edge_index, batch)  # [g, md], [n, md]

        _, task_outs_n = self.moe(x, x_s, edge_index, batch)
        # task_sub_g, task_outs_n = self.moe(x, x_s, edge_index, batch)

        task_sub_g = []
        for i, nx in enumerate(task_outs_n):
            node_prob = self.extractor(nx, edge_index, batch)
            sub_nx = self.node_masking(nx, node_prob, batch)
            sub_gx = self.graph_dense_layers[i](global_mean_pool(sub_nx, batch))
            task_sub_g.append(sub_gx)
        gh, gl = task_sub_g

        cross_gh, _ = self.cross_attn_high(gh, self.vq.embedding.weight, self.vq.embedding.weight)
        cross_gl, _ = self.cross_attn_low(gl, self.vq.embedding.weight, self.vq.embedding.weight)

        cross_gh = self.norm_high(cross_gh)
        cross_gl = self.norm_low(cross_gl)

        vq_loss = 0
        perplexity_list = []
        recon_list = []
        z_q_list = []
        for i, z_e in enumerate(task_outs_n):  # [N, D]
            z_q, vq_loss_item, _, perplexity = self.vq(z_e)  # [N, D] -> [N, D]
            rec = self.dec_proj_layers[i](z_q)  # [N, D] -> [N, D]
            # 统计
            vq_loss += vq_loss_item
            perplexity_list.append(perplexity)
            recon_list.append(rec)
            # node -> graph
            g_z_q = self.graph_dense_layers[i](global_mean_pool(z_q, batch))
            z_q_list.append(g_z_q)

        h_rec, l_rec = recon_list  # [N, D]
        recon_loss = F.mse_loss(h_rec, task_outs_n[0]) + F.mse_loss(l_rec, task_outs_n[1])

        return gh, gl, cross_gh, cross_gl, z_q_list, recon_loss, vq_loss, perplexity_list

    def loss_func(self, emb_list, batch, t):

        # gh, gl, cross_gh, cross_gl, hp, lp, attn_gh, attn_gl = emb_list
        # gh, gl, cross_gh, cross_gl, hp, lp, h_rec, l_rec = emb_list
        gh, gl, cross_gh, cross_gl, z_q_list, _, _, _ = emb_list

        hp, lp = z_q_list

        loss_ii = self.gcl_loss_g(cross_gh, cross_gl, t) + self.gcl_loss_g(gh, gl, t)
        # loss_ii += self.gcl_loss_g(attn_gh, attn_gl, t)
        # loss_pp = self.gcl_loss_g(hp, lp, t)
        loss_pp = torch.tensor([0.]).to(batch.device)
        loss_ip = self.gcl_loss_g(cross_gh, hp, t) + self.gcl_loss_g(cross_gl, lp, t)

        # loss_recon = self.recon_loss(cross_gh, h_rec) + self.recon_loss(cross_gl, l_rec)
        loss_recon = torch.tensor([0.]).to(batch.device)

        return loss_ii, loss_pp, loss_ip, loss_recon

    def score_func(self, emb_list, batch, t):

        # gh, gl, cross_gh, cross_gl, hp, lp, attn_gh, attn_gl = emb_list
        # gh, gl, cross_gh, cross_gl, hp, lp, h_rec, l_rec = emb_list
        gh, gl, cross_gh, cross_gl, z_q_list, _, _, _ = emb_list

        hp, lp = z_q_list

        score_ii = self.gcl_loss_g(cross_gh, cross_gl, t) + self.gcl_loss_g(gh, gl, t)
        # score_ii += self.gcl_loss_g(attn_gh, attn_gl, t)
        # score_pp = self.gcl_loss_g(hp, lp, t)
        score_pp = torch.tensor([0.]).to(batch.device)
        score_ip = self.gcl_loss_g(cross_gh, hp, t) + self.gcl_loss_g(cross_gl, lp, t)

        # score_recon = self.recon_loss(cross_gh, h_rec) + self.recon_loss(cross_gl, l_rec)
        score_recon = torch.tensor([0.]).to(batch.device)

        return score_ii, score_pp, score_ip, score_recon

    @staticmethod
    def gcl_loss_n(x, x_aug, batch, temperature=0.2):
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
    def gcl_loss_g(x, x_aug, temperature=0.2):
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