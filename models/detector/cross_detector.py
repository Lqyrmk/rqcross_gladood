import torch
import torch.nn as nn
import os

from tqdm import tqdm
from ..nn import cross
from utils.path import get_model_save_path, clear_directory
from utils.metrics import ood_auc
from visualization.attention_heatmap import visualize_attention

class CrossDetector:

    def __init__(self, config):

        self.config = config
        self.device = config.device

        self.in_dim = config.dataset_num_features
        self.hid_dim = config.hidden_dim
        self.num_layers = config.num_layer
        self.lr = config.lr
        self.temperature = config.temperature
        self.dropout = config.dropout
        self.eps = config.eps
        self.scalar = config.scalar
        self.k = config.k
        self.num_heads = config.num_heads
        self.num_experts = config.num_experts
        self.pooling = config.pooling
        self.readout = config.readout

        self.path = get_model_save_path(config)
        clear_directory(self.path)
        self.max_auc = 0


    def init_model(self):
        return cross.CROSS(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            eps=self.eps,
            scalar=self.scalar,
            k=self.k,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            pooling=self.pooling,
            readout=self.readout
        ).to(self.device)


    def fit(self, dataloader, dataloader_val):
        model = self.init_model()
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        patience = 80
        counter = 0

        for epoch in range(1, self.config.num_epoch + 1):
            model.train()
            total_loss = 0

            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                emb = model(data)
                loss_ii, loss_pp, loss_ip = model.loss_func(emb, data.batch, self.temperature)
                loss = loss_ii.mean() + loss_pp.mean() + loss_ip.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            if epoch % self.config.eval_freq == 0:
                model.eval()
                y, score = [],[]
                for data in dataloader_val:
                    data = data.to(self.device)
                    emb = model(data)
                    s_ii, s_pp, s_ip = model.score_func(emb, data.batch, self.temperature)
                    # score.extend( s.cpu().tolist() )
                    score.extend( (s_ii + s_pp + s_ip).cpu().tolist() )
                    y.extend(data.y.cpu().tolist())

                auc = ood_auc(y, score)
                if auc > self.max_auc:
                    self.max_auc = auc
                    counter = 0
                    torch.save(model, os.path.join(self.path,'model.pth'))
                else:
                    counter +=1
                print(f"[Epoch {epoch:03d}] Val AUC: {auc:.4f} | Best: {self.max_auc:.4f}")
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
                h_self_scores, l_self_scores, h_cross_scores, l_cross_scores = emb[-4], emb[-3], emb[-2], emb[-1]
                visualize_attention(h_self_scores, title="High Self Attention")
                visualize_attention(l_self_scores, title="Low Self Attention")
                visualize_attention(h_cross_scores, title="High Cross Attention")
                visualize_attention(l_cross_scores, title="Low Cross Attention")
                s_ii, s_pp, s_ip = model.score_func(emb, data.batch, self.temperature)
                # score.extend( s.cpu().tolist() )
                score.extend( (s_ii + s_pp + s_ip).cpu().tolist() )
                y.extend(data.y.cpu().tolist())

        return score, y