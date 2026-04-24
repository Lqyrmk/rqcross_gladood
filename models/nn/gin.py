import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool

class GIN(nn.Module):
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
                net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim)
                )
            else:
                net = nn.Sequential(
                    nn.Linear(num_features, dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim)
                )
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