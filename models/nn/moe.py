import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .gin import GIN
from .bwgnn import BWGNN
from torch_geometric.nn import global_mean_pool

class HfEncoder(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.bw_encoder = BWGNN(in_dim, hid_dim, out_dim)

    def forward(self, x, edge_index, batch):
        node_emb = self.bw_encoder(x, edge_index)  # [n, md]
        graph_emb = global_mean_pool(node_emb, batch)  # [g, md]
        return graph_emb


class LfEncoder(nn.Module):

    def __init__(self, in_dim, hid_dim, num_layers):
        super().__init__()
        self.gin_encoder = GIN(in_dim, hid_dim, num_layers)

    def forward(self, x, edge_index, batch):
        graph_emb, node_emb = self.gin_encoder(x, edge_index, batch)  # [g, md], [n, md]
        return graph_emb


class Expert(nn.Module):
    def __init__(self, encoder, enc_out_dim, out_dim):
        super().__init__()
        self.encoder = encoder
        self.bn = nn.BatchNorm1d(enc_out_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(enc_out_dim, out_dim)

    def forward(self, x, edge_index, batch):
        feat = self.encoder(x, edge_index, batch)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)
        out = self.proj(feat)
        return out

class MMoE(nn.Module):

    def __init__(
        self,
        in_dim,
        num_experts,
        expert_dim,
        num_tasks,
        task_dim,
        hid_dim,
        out_dim,
        gnn_layers
    ):
        super().__init__()

        self.num_tasks = num_tasks

        self.num_lf = num_experts // 2
        self.num_hf = num_experts - self.num_lf

        self.experts = nn.ModuleList()

        # concat dimension
        enc_out_dim = hid_dim * gnn_layers

        for _ in range(self.num_hf):
            hf_encoder = HfEncoder(in_dim, hid_dim, enc_out_dim)
            self.experts.append(Expert(hf_encoder, enc_out_dim, expert_dim))
        for _ in range(self.num_lf):
            lf_encoder = LfEncoder(in_dim, hid_dim, gnn_layers)
            self.experts.append(Expert(lf_encoder, enc_out_dim, expert_dim))

        # 共享专家
        shared_hf_expert = HfEncoder(in_dim, hid_dim, enc_out_dim)
        self.experts.append(Expert(shared_hf_expert, enc_out_dim, expert_dim))
        num_experts += 1

        # 全 bwgnn 效果也不错
        # for _ in range(num_experts):
        #     hf_encoder = HfEncoder(in_dim, hid_dim, enc_out_dim)
        #     self.experts.append(Expert(hf_encoder, enc_out_dim, expert_dim))

        self.gates = nn.ModuleList([
            nn.Linear(in_dim, num_experts) for _ in range(num_tasks)
        ])

        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, task_dim),
                # nn.ReLU(),
                nn.BatchNorm1d(task_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(task_dim, out_dim)
            ) for _ in range(num_tasks)
        ])

        self.expert_norm = nn.BatchNorm1d(enc_out_dim)
        self.gate_dropout = nn.Dropout(0.2)
        self.out_dropout = nn.Dropout(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):

        expert_outs = [expert(x, edge_index, batch) for expert in self.experts]  # expert(x): [B, d]
        expert_outs = torch.stack(expert_outs, dim=1)  # [B, E, d]

        g_x = global_mean_pool(x, batch)

        task_outs = []
        for i in range(self.num_tasks):

            gate_out = self.gates[i](g_x)
            gate_weight = F.softmax(gate_out, dim=-1)  # [B, E]

            # 防止专家饥饿
            gate_weight = self.gate_dropout(gate_weight)

            # method 1: matrix multiplication
            # [B, 1, E] @ [B, E, d] -> [B, 1, d] -> [B, d]
            # fused = torch.matmul(gate_weight.unsqueeze(1), expert_outs).squeeze(1)  # or torch.bmm
            # method 2: element-wise product and sum
            # [B, E, 1] * [B, E, d] -> [B, E, d] -> [B, d]
            fused = torch.sum(gate_weight.unsqueeze(-1) * expert_outs, dim=1)

            out = self.task_towers[i](fused)  # [B, dt]
            out = self.out_dropout(self.expert_norm(out))
            task_outs.append(out)

        return task_outs