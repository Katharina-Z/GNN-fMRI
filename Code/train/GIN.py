import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import functional as FM


import torch
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from sklearn.metrics import f1_score, roc_auc_score



from BrainDataset import BrainDataset2


class LitGIN(pl.LightningModule):

    def __init__(self):
        super().__init__()

        num_features = 103
        dim = 100

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 2) #second value is dataset.num_classes

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index)) #can try tanh
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch) #could try global max pool
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
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
        y_tensor = y[1]
        logits = self(x_tensor, edge_index_tensor, batch_tensor)
        loss = F.nll_loss(logits, y_tensor) #can try other loss function

        pred = logits.max(dim=1)[1]
        acc = FM.accuracy(pred.cpu(), y_tensor.cpu(), num_classes=2)
        f1 = FM.f1_score(pred.cpu(), y_tensor.cpu())

        return {'loss': loss, 'train_acc': acc, 'train_f1': f1}

    def validation_step(self, batch, batch_idx):
        batch, edge_attr, edge_index, x, y = batch
        x_tensor = x[1]
        edge_index_tensor = edge_index[1]
        batch_tensor = batch[1]
        y_tensor = y[1]
        logits = self(x_tensor, edge_index_tensor, batch_tensor)
        loss = F.nll_loss(logits, y_tensor)

        pred = logits.max(dim=1)[1]
        acc = FM.accuracy(pred.cpu(), y_tensor.cpu(), num_classes=2)
        f1 = FM.f1_score(pred.cpu(), y_tensor.cpu())

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer) #change scheduler
        return [optimizer], [scheduler]



dataset = BrainDataset2('/home/ubuntu/abidegraphs')

train_dataset, val_dataset = random_split(dataset, [((len(dataset))-100), 100])


GIN = LitGIN()
logger = TensorBoardLogger('tb_logs', name='all_scans')
trainer = pl.Trainer(gpus =1, max_epochs=100, logger=logger)
trainer.fit(GIN)