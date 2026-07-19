import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from tqdm import tqdm
from ..nn import cross
from utils.path import get_model_save_path, clear_directory
from utils.metrics import ood_auc
from visualization.attention_heatmap import visualize_attention
from sklearn.cluster import KMeans
class CrossDetector:

    def __init__(self, config):

        self.config = config
        self.device = config.device
        self.lr = config.lr
        self.temperature = config.temperature

        self.path = get_model_save_path(config)
        clear_directory(self.path)
        self.max_auc = 0

    def init_model(self):
        return cross.CROSS(self.config).to(self.device)

    def initialize_codebook(self, model, dataloader):
        config = self.config

        if not config.kmeans_init:
            print(f"跳过 KMeans 码本初始化")
            return

        print(f"开始 KMeans 码本初始化...")
        print(f"使用前 {config.kmeans_init_num_batches} 个 batch 进行初始化...")

        model.eval()
        all_z_e = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                if i >= config.kmeans_init_num_batches:
                    break

                data = data.to(self.device)
                x = data.x  # [B, D]
                edge_index = data.edge_index
                z_e = model.bwgnn_encoder(x, edge_index)  # [B, H]
                all_z_e.append(z_e.cpu().numpy())

                print(f"\nBatch {i}, z_e = {z_e.shape}")

        all_z_e = np.concatenate(all_z_e, axis=0)
        print(f"潜向量个数：{len(all_z_e)}")
        print(f"开始聚类...")

        kmeans = KMeans(n_clusters=config.k, random_state=config.seed)
        kmeans.fit(all_z_e)

        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        # # 初始化码本
        model.vq.embedding.weight.data.copy_(cluster_centers)
        # # 初始化 EMA 的移动平均变量
        model.vq._ema_w.data.copy_(cluster_centers)

        print(f"KMeans 聚类完成")

        # 切换回去
        model.train()

    def fit(self, dataloader, dataloader_val):
        model = self.init_model()

        # 训练前，先做一次KMeans码本初始化
        # self.initialize_codebook(model, dataloader)

        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        patience = 80
        counter = 0

        for epoch in range(1, self.config.num_epoch + 1):
            model.train()
            total_mean_loss = 0
            total_mean_rec_loss = 0
            total_mean_vq_loss = 0
            total_loss = 0

            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                emb = model(data)

                # vq_loss = emb[-1]
                # recon_loss = emb[-2]

                # loss_ii, loss_pp, loss_ip = model.loss_func(emb, data.batch, self.temperature)
                # loss_ii, loss_pp = model.loss_func(emb, data.batch, self.temperature)
                loss_ii, loss_pp, rec_loss_g, rec_loss_n, mi_loss = model.loss_func(emb, data.batch, self.temperature)
                # loss = loss_ii.mean() + loss_pp.mean() + loss_ip.mean() + vq_loss + recon_loss
                loss = loss_ii.mean() + loss_pp.mean() + rec_loss_g.mean() + rec_loss_n.mean() + mi_loss.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_mean_loss += loss.item()
                # total_mean_vq_loss += vq_loss
                # total_mean_rec_loss += recon_loss
                total_loss += loss.item() * data.num_graphs

            if epoch % self.config.eval_freq == 0:
                model.eval()
                y, score = [],[]
                for data in dataloader_val:
                    data = data.to(self.device)
                    emb = model(data)
                    # vq = emb[-1]
                    # recon = emb[-2]
                    # s_ii, s_pp, s_ip = model.score_func(emb, data.batch, self.temperature)
                    # s_ii, s_pp = model.score_func(emb, data.batch, self.temperature)
                    s_ii, s_pp, rec_s_g, rec_s_n = model.score_func(emb, data.batch, self.temperature)
                    # score.extend( s.cpu().tolist() )
                    # score.extend( (s_ii + s_pp + s_ip + vq + recon).cpu().tolist() )
                    # score.extend( (s_ii + s_pp).cpu().tolist() )
                    score.extend( (s_ii + s_pp + rec_s_g + rec_s_n).cpu().tolist() )
                    y.extend(data.y.cpu().tolist())

                auc = ood_auc(y, score)
                if auc > self.max_auc:
                    self.max_auc = auc
                    counter = 0
                    torch.save(model, os.path.join(self.path,'model.pth'))
                else:
                    counter +=1
                log_info = f"[Epoch {epoch:03d}] Val AUC: {auc:.4f} | Best: {self.max_auc:.4f} | Total Mean Loss: {total_mean_loss:.4f}"
                vq_info = f" | Recon Loss: {total_mean_rec_loss:.4f} | Commitment Loss: {total_mean_vq_loss:.4f} "
                print(log_info + vq_info)
                if counter >= patience or self.max_auc > 0.999:
                    print(f"Early stop triggered.")
                    break

    def predict(self, dataloader):
        model = torch.load(os.path.join(self.path,'model.pth'))
        model.eval()
        y, score = [],[]

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                emb = model(data)
                # h_self_scores, l_self_scores, h_cross_scores, l_cross_scores = emb[-4], emb[-3], emb[-2], emb[-1]
                # visualize_attention(h_self_scores, title="High Self Attention")
                # visualize_attention(l_self_scores, title="Low Self Attention")
                # visualize_attention(h_cross_scores, title="High Cross Attention")
                # visualize_attention(l_cross_scores, title="Low Cross Attention")

                # vq = emb[-1]
                # recon = emb[-2]

                # s_ii, s_pp, s_ip = model.score_func(emb, data.batch, self.temperature)
                # s_ii, s_pp = model.score_func(emb, data.batch, self.temperature)
                s_ii, s_pp, rec_s_g, rec_s_n = model.score_func(emb, data.batch, self.temperature)
                # score.extend( s.cpu().tolist() )
                # score.extend( (s_ii + s_pp + s_ip + vq + recon).cpu().tolist() )
                # score.extend( (s_ii + s_pp).cpu().tolist() )
                score.extend( (s_ii + s_pp + rec_s_g + rec_s_n).cpu().tolist() )
                y.extend(data.y.cpu().tolist())

        return score, y