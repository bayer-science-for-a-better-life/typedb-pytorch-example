import torch
import torch.nn as nn
import torch.functional as F
from torch_geometric.data import DataLoader
from kglib.kgcn.examples.diagnosis.diagnosis import get_query_handles
from grakn_pytorch_geometric.data.dataset import GraknPytorchGeometricDataSet
from grakn_pytorch_geometric.models.core import KGCN
from grakn_pytorch_geometric.metrics import metrics, lightning_metrics

from transforms import (
    networkx_transform,
    node_types,
    edge_types,
    CATEGORICAL_ATTRIBUTES,
    CONTINUOUS_ATTRIBUTES,
)

import pytorch_lightning as pl


class GraphModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = KGCN(
            node_types=node_types,
            edge_types=edge_types,
            categorical_attributes=CATEGORICAL_ATTRIBUTES,
            continuous_attributes=CONTINUOUS_ATTRIBUTES,
        )

        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.train_accuracy_node = lightning_metrics.Accuracy(ignore_index=0)
        self.train_accuracy_edge = lightning_metrics.Accuracy(ignore_index=0)
        self.val_accuracy_node = lightning_metrics.Accuracy(ignore_index=0)
        self.val_accuracy_edge = lightning_metrics.Accuracy(ignore_index=0)

    def forward(self, x):
        predictions = self.model(x)
        return F.softmax(predictions)

    def training_step(self, batch, batch_idx):
        node_prediction, edge_prediction = self.model(batch)
        node_loss = self.loss_function(node_prediction, batch.y)
        edge_loss = self.loss_function(edge_prediction, batch.y_edge)
        loss = (node_loss + edge_loss) * 0.5


        self.train_accuracy_node(node_prediction, batch.y)
        self.train_accuracy_edge(edge_prediction, batch.y_edge)

        fraction_solved = metrics.fraction_solved(
            node_prediction, batch.y, batch.batch, ignore_index=0
        )

        self.log("train_accuracy_node", self.train_accuracy_node)
        self.log("train_accuracy_edge", self.train_accuracy_edge)

        self.log("train_loss", loss)
        self.log("train_node_loss", node_loss)
        self.log("train_edge_loss", edge_loss)
        self.log("train_fraction_solved", fraction_solved)
        return loss

    def validation_step(self, batch, batch_idx):
        node_prediction, edge_prediction = self.model(batch)
        node_loss = self.loss_function(node_prediction, batch.y)
        edge_loss = self.loss_function(edge_prediction, batch.y_edge)
        loss = (node_loss + edge_loss) * 0.5

        self.val_accuracy_node(node_prediction, batch.y)
        self.val_accuracy_edge(edge_prediction, batch.y_edge)

        fraction_solved = metrics.fraction_solved(
            node_prediction, batch.y, batch.batch, ignore_index=0
        )

        self.log("val_accuracy_node", self.train_accuracy_node)
        self.log("val_accuracy_edge", self.train_accuracy_edge)

        self.log("val_loss", loss)
        self.log("val_node_loss", node_loss)
        self.log("val_edge_loss", edge_loss)
        self.log("val_fraction_solved", fraction_solved)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)
        return optimizer

model = GraphModel()

# train and test split
example_indices = list(range(80))
val_indices = list(range(80, 100))


# Create a dataset
grakn_dataset = GraknPytorchGeometricDataSet(
    example_indices=example_indices,
    get_query_handles_for_id=get_query_handles,
    infer=True,
    uri="localhost:48555",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

# Use the Dataset to create a Pytorch (geometric) DataLoader
train_dataloader = DataLoader(grakn_dataset, batch_size=5, num_workers=0, shuffle=True)

# Create a validation dataset
val_dataset = GraknPytorchGeometricDataSet(
    example_indices=val_indices,
    get_query_handles_for_id=get_query_handles,
    infer=True,
    uri="localhost:48555",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

# Use the test dataset to create a validation dataloader
val_dataloader = DataLoader(val_dataset, batch_size=5, num_workers=0, shuffle=True)

trainer = pl.Trainer()


if __name__ == '__main__':
    #met = Metrics()


    print(model)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
