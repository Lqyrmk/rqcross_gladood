import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .gin import GIN
from .bwgnn import BWGNN
from .attention import CrossAttention, SelfAttention, DoubleCrossAttention
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

        self.cross_g = DoubleCrossAttention(d_model=self.emb_dim, device=config.device)
        self.cross_n = DoubleCrossAttention(d_model=self.emb_dim, device=config.device)

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
        vq1 = VectorQuantizerEMA(num_embeddings=self.k,
                                embedding_dim=self.emb_dim,
                                commitment_cost=commitment_cost,
                                decay=vq_decay,
                                epsilon=vq_epsilon)
        vq2 = VectorQuantizerEMA(num_embeddings=self.k,
                                embedding_dim=self.emb_dim,
                                commitment_cost=commitment_cost,
                                decay=vq_decay,
                                epsilon=vq_epsilon)

        self.vq = nn.ModuleList([vq1, vq2])


        self.encoder_feat = GIN(in_dim, hid_dim, num_layers, self.pooling, self.readout)
        self.encoder_str = GIN(in_str_dim, hid_dim, num_layers, self.pooling, self.readout)
        self.Cross_Attention_g = DoubleCrossAttention(d_model=self.emb_dim, device=config.device)
        self.Cross_Attention_n = DoubleCrossAttention(d_model=self.emb_dim, device=config.device)

        self.graph_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim * 2)   # 重构两个图级表示
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim + in_str_dim)
        )

        self.mi_discriminator = nn.Bilinear(self.emb_dim, self.emb_dim, 1)

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

    def fusion(self, x1, x2):
        return x1 + x2

    def forward(self, data):
        x_f, x_s, edge_index, batch = data.x, data.x_s, data.edge_index, data.batch
        g_f, n_f = self.encoder_feat(x_f, edge_index, batch)
        g_s, n_s = self.encoder_str(x_s, edge_index, batch)

        # moe 效果并不好，看看能不能对 gin 层做专家混合，选择特定gin层放到结果里，然后concat readout
        # 层1选自gin1，层2选自gin4，层3选自gin2，层4选自gin5
        # task_outs_g, task_outs_n = self.moe(x_f, x_s, edge_index, batch)
        # g_f, g_s = task_outs_g
        # n_f, n_s = task_outs_n

        g_f_2, g_s_2 = self.Cross_Attention_g(g_f, g_s)

        n_f_2, n_s_2 = self.Cross_Attention_n(n_f, n_s)

        g = self.fusion(g_f_2, g_s_2)
        n = self.fusion(n_f_2, n_s_2)
        g_rec = self.graph_decoder(g)          # 拆分为 (g_f_rec, g_s_rec)
        n_rec = self.node_decoder(n)          # 拆分为 (x_f_rec, x_s_rec)

        return g_f_2, g_s_2, n_f_2, n_s_2, torch.cat([g_f, g_s], dim=1), torch.cat([x_f, x_s], dim=1), g_rec, n_rec


    def loss_func(self, emb_list, batch, t):

        # g_f, g_s, n_f, n_s = emb_list
        g_f, g_s, n_f, n_s, g, n, g_rec, n_rec = emb_list
        loss_g = self.gcl_loss_g(g_f, g_s)
        loss_n = self.gcl_loss_n(n_f, n_s, batch)

        rec_loss_g = F.mse_loss(g_rec, g, reduction='none')
        rec_loss_n = F.mse_loss(n_rec, n, reduction='none')

        mi_loss = self.mi_loss(n_f, g_f, batch) + self.mi_loss(n_s, g_s, batch)

        return loss_g, loss_n, rec_loss_g, rec_loss_n, mi_loss

    def score_func(self, emb_list, batch, t):

        # g_f, g_s, n_f, n_s = emb_list
        g_f, g_s, n_f, n_s, g, n, g_rec, n_rec = emb_list
        score_g = self.gcl_loss_g(g_f, g_s)
        score_n = self.gcl_loss_n(n_f, n_s, batch)

        recon_score_g = F.mse_loss(g_rec, g, reduction='none').mean(dim=1)
        node_mse = F.mse_loss(n_rec, n, reduction='none').mean(dim=1)
        recon_score_n = scatter_add(node_mse, batch) / scatter_add(torch.ones_like(node_mse), batch)

        return score_g, score_n, recon_score_g, recon_score_n

    def mi_loss(self, n, g, batch):
        # 正样本得分
        pos_score = self.mi_discriminator(n, g[batch])  # g[batch] 将图表示广播到对应节点
        # 负样本：打乱 batch 顺序
        shuffle_idx = torch.randperm(g.size(0))
        neg_g = g[shuffle_idx][batch]
        neg_score = self.mi_discriminator(n, neg_g)
        mi_loss = -F.logsigmoid(pos_score) - F.logsigmoid(-neg_score)
        return mi_loss

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