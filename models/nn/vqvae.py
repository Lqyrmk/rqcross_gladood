import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
 # parents[0]=当前目录, [1]=上层, [2]=上上层
ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))
from config import parse_args

class Encoder(nn.Module):

    def __init__(self, in_dim, hid_dim):
        super().__init__()

        self.enc_proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, x):
        return self.enc_proj(x)  # [B, D] -> [B, H]


class Decoder(nn.Module):

    def __init__(self, hid_dim, out_dim):
        super().__init__()

        self.dec_proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, z_q):
        return self.dec_proj(z_q)  # [B, H] -> [B, D]


class VectorQuantizerEMA(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 commitment_cost,
                 decay,
                 epsilon
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # codebook: [K, D]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # EMA 更新时无需梯度
        self.embedding.weight.requires_grad = False

        # EMA 移动平均变量
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))  # [K,]
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, z_e):
        """
        z_e -> z_q
        Args:
            z_e: 编码器输出的潜向量，shape=(B, D)
        Returns:
            z_q: 量化后的潜向量，shape=(B, D)
            loss: commitment loss
            indices: 码本索引，shape=(B,)
            perplexity: 码本困惑度（衡量码本利用率，越高越好）
        """
        N, D = z_e.shape

        # ||x-y|| ^ 2 = ||x||^2 + ||y||^2 - 2xy
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)  # [N, 1]
            + torch.sum(self.embedding.weight ** 2, dim=1)  # [K,]
            - 2 * torch.matmul(z_e, self.embedding.weight.t())  # [N, K]
        )
        indices = torch.argmin(distances, dim=1)  # [N,]

        # 量化
        z_q = self.embedding(indices)  # [N, D]

        # 直通梯度估计（Straight-Through Estimator）
        z_q = z_e + (z_q - z_e).detach()

        # commitment loss
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        loss = self.commitment_cost * e_latent_loss

        # 统计每个 code 在当前 batch 中被使用的次数
        encodings = F.one_hot(indices, self.num_embeddings).type(z_e.dtype)  # [N, K]

        # EMA 更新码本（训练中执行）
        if self.training:
            # 使用次数
            cluster_size = encodings.sum(0)  # [K,]

            # 指数移动平均更新簇大小
            self._ema_cluster_size = self._ema_cluster_size * self.decay + cluster_size * (1 - self.decay)

            # 拉普拉斯平滑，防止除零
            n = torch.sum(self._ema_cluster_size)  # 总数
            self._ema_cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )

            # 指数移动平均更新簇中心
            # 求和：[K, N] @ [N, D] -> [K, D]
            dw = torch.matmul(encodings.t(), z_e)
            self._ema_w = self._ema_w * self.decay + dw * (1 - self.decay)  # [K, D]

            # 更新码本权重
            # 簇内平均：[K, D] / [K, 1] -> [K, D]
            self.embedding.weight.data.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # 码本困惑度，衡量码本利用率，最大值为 num_embeddings
        avg_probs = torch.mean(encodings, dim=0)  # [K,] 利用率
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  # 标量

        return z_q, loss, indices, perplexity

class VQVAE(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config.in_dim, config.hid_dim)

        self.vq = VectorQuantizerEMA(config.num_embeddings,
                                    config.hid_dim,
                                    config.commitment_cost,
                                    config.decay,
                                    config.epsilon)

        self.decoder = Decoder(config.hid_dim, config.in_dim)

    def forward(self, x):
        z_e = self.encoder(x)  # [B, D] -> [B, H]
        z_q, vq_loss, indices, perplexity = self.vq(z_e)
        x_recon = self.decoder(z_q)  # [B, H] -> [B, D]
        # 重建
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, recon_loss, vq_loss, indices, perplexity

if __name__ == "__main__":

    config = parse_args()

    config.in_dim = 10
    config.hid_dim = 5
    config.num_embeddings = 6
    config.embedding_dim = 5
    config.commitment_cost = 0.25
    config.decay = 0.99
    config.epsilon = 1e-5

    x = torch.rand(5, 10)
    model = VQVAE(config)

    x_recon, recon_loss, vq_loss, indices, perplexity = model(x)
    print(f"x_recon: {x_recon}")
    print(f"recon loss: {recon_loss:.2f}")
    print(f"commitment loss: {vq_loss:.2f}")
    print(f"indices: {indices}")
    print(f"perplexity: {perplexity}")