import torch
import os

from tqdm import tqdm
from ..nn import cross
from utils.path import get_model_save_path, clear_directory
from utils.metrics import ood_auc

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
            dropout = self.dropout,
            eps = self.eps,
            scalar = self.scalar,
            pooling = self.pooling,
            readout = self.readout
        ).to(self.device)


    def fit(self, dataloader, dataloader_val):
        model = self.init_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        patience = 80
        counter = 0

        for epoch in range(1, self.config.num_epoch + 1):
            model.train()
            total_loss = 0

            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                emb = model(data)
                l = model.loss_func(emb, data.batch, self.temperature)
                loss = l.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            if epoch % self.config.eval_freq == 0:
                model.eval()
                y, score = [],[]
                for data in dataloader_val:
                    data = data.to(self.device)
                    emb = model(data)
                    s = model.score_func(emb, data.batch, self.temperature)
                    score.extend( s.cpu().tolist() )
                    y.extend(data.y.cpu().tolist())

                auc = ood_auc(y, score)
                if auc > self.max_auc:
                    self.max_auc = auc
                    counter = 0
                    torch.save(model, os.path.join(self.path,'model.pth'))
                else:
                    counter +=1

                if counter >= patience: break

    def predict(self, dataloader):
        model = torch.load(os.path.join(self.path,'model.pth'))
        model.eval()
        y, score = [],[]

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                emb = model(data)
                s = model.score_func(emb, data.batch, self.temperature)
                score.extend( s.cpu().tolist() )
                y.extend(data.y.cpu().tolist())

        return score, y