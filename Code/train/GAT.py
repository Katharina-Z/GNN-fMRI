import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import functional as FM

import torch
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool, global_mean_pool

from BrainDataset import BrainDataset2

from functools import partial
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


class LitGAT(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.dim = config['dim']
        self.lr = config['lr']

        num_features = 103
        num_classes = 1
        dim = self.dim

        self.conv1 = GATConv(num_features, dim, heads = 1)
        self.conv2 = GATConv(dim, num_classes, heads = 1)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = global_max_pool(x, batch)
        x = torch.sigmoid(x)
        return x

    def train_dataloader(self) :
        return DataLoader(train_dataset, batch_size=4, num_workers=16)

    def val_dataloader(self) :
        return DataLoader(val_dataset, batch_size=4, num_workers=16)

    def training_step(self, batch, batch_idx):
        batch, edge_attr, edge_index, x, y = batch
        x_tensor = x[1]
        edge_index_tensor = edge_index[1]
        batch_tensor = batch[1]
        y_tensor = y[1].unsqueeze(1).float()
        logits = self(x_tensor, edge_index_tensor, batch_tensor)

        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(logits, y_tensor)

        y_pred = (logits > 0.5).float()
        acc = FM.accuracy(y_pred.cpu(), y_tensor.cpu(), num_classes=2)
        f1 = FM.f1_score(y_pred.cpu(), y_tensor.cpu(), num_classes=2)

        return {'loss': loss, 'train_acc': acc, 'train_f1': f1}

    def validation_step(self, batch, batch_idx):
        batch, edge_attr, edge_index, x, y = batch
        x_tensor = x[1]
        edge_index_tensor = edge_index[1]
        batch_tensor = batch[1]
        y_tensor = y[1].unsqueeze(1).float()
        logits = self(x_tensor, edge_index_tensor, batch_tensor)

        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(logits, y_tensor)

        y_pred = (logits > 0.5).float()
        acc = FM.accuracy(y_pred.cpu(), y_tensor.cpu(), num_classes=2)
        f1 = FM.f1_score(y_pred.cpu(), y_tensor.cpu(), num_classes=2)

        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_accuracy = torch.stack([x['train_acc'] for x in outputs]).mean()
        train_f1 = torch.stack([x['train_f1'] for x in outputs]).mean()

        tensorboard_logs = {'loss': avg_loss, 'train_acc': train_accuracy, 'train_f1': train_f1}
        return {'avg_training_loss': avg_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
        val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_accuracy, 'val_f1': val_f1}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]



dataset = BrainDataset2('/home/ubuntu/ADHD200')

train_dataset, val_dataset = random_split(dataset, [((len(dataset))-100), 100])



config = {
    'dim': tune.grid_search([32, 64, 128]),
    'lr': tune.grid_search([1e-3, 1e-4, 1e-5])
}


def GAT_run(config, num_epochs, num_gpus):
    np.random.seed(42)
    torch.manual_seed(42)
    GAT = LitGAT(config)
    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")
    trainer = pl.Trainer(gpus = num_gpus, max_epochs = num_epochs, logger=logger,
                         callbacks=[TuneReportCallback({'loss': 'val_loss', 'mean_accuracy': 'val_acc'},
                                                       on='validation_end')])
    trainer.fit(GAT)


tune.run(partial(GAT_run, num_epochs=100, num_gpus=1),
         resources_per_trial={'cpu': 16, 'gpu': 1}, config=config, name='tune_GAT_0802', num_samples = 1)