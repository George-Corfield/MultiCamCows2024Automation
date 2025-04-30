import pytorch_lightning as pl
import torchvision.models as M
import torch.nn as nn

import torch
from pytorch_metric_learning import losses, miners


class SimCLR(pl.LightningModule):

    def __init__(self, epochs, lr=0.001, weight_decay=0.01, out_dim=128):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.save_hyperparameters(self.hparams)
        # Base
        self.base = M.resnet18(
            weights=M.ResNet18_Weights.DEFAULT)
        # Projection
        self.base.fc = nn.Sequential(
            self.base.fc,
            nn.ReLU(inplace=True),
            nn.Linear(1000, out_dim)
        )

    def forward(self, x):
        return self.base(x)

    def calculate_loss(self, batch):
        (images1, images2), labels, _, _ = batch
        images = torch.cat([images1, images2], dim=0)

        features = self.base(images)
        features = nn.functional.normalize(features, p=2, dim=1)

        all_labels = torch.cat([labels, labels], dim=0)

        lf = losses.NTXentLoss()
        miner = miners.DistanceWeightedMiner()

        pairings = miner(features, all_labels)
        loss = lf(features, all_labels, pairings)

        return loss

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs, verbose=True)
        return {
            'optimizer': optimiser,
            "lr_scheduler": {
                "scheduler": lr_schedule,
                "monitor": "Total_Validation_NTXent_loss",
                "frequency": 1,
            },
        }

    def training_step(self, batch):
        loss = self.calculate_loss(batch)
        self.log('Total_Training_NTXent_loss', loss,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch):
        loss = self.calculate_loss(batch)

        self.log('Total_Validation_NTXent_loss', loss,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def set_camera(self, cam):
        self.camera = cam
