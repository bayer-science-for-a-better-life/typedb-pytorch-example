import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import F1, Accuracy
from torch_geometric.data import DataLoader

from grakn.client import *

from kglib.kgcn_data_loader.transform.standard_kgcn_transform import (
    StandardKGCNNetworkxTransform,
)
from kglib.kgcn_data_loader.utils import (
    get_edge_types_for_training,
    get_node_types_for_training,
)

from typedb_pytorch_geometric.data.dataset import GraknPytorchGeometricDataSet
from typedb_pytorch_geometric.models.core import KGCN
from typedb_pytorch_geometric.utils.lightning_metrics import (
    FractionSolved,
    IgnoreIndexMetric,
)
from typedb_pytorch_geometric.utils.loss import MultiStepLoss


from about_this_graph import (
    get_query_handles,
    CATEGORICAL_ATTRIBUTES,
    CONTINUOUS_ATTRIBUTES,
    TYPES_AND_ROLES_TO_OBFUSCATE,
    TYPES_TO_IGNORE,
    ROLES_TO_IGNORE,
)


client = Grakn.core_client("localhost:1729")
session = client.session(database="diagnosis", session_type=SessionType.DATA)
node_types = get_node_types_for_training(session, TYPES_TO_IGNORE)
edge_types = get_edge_types_for_training(session, ROLES_TO_IGNORE)


class Metrics(nn.Module):
    """
    Just grouping together all metrics we are interested in logging
    so we can easily log the same metrics for the train and the
    validation set.
    """

    def __init__(self, prepend=""):
        super().__init__()
        self._prepend = prepend
        self.node_accuracy = IgnoreIndexMetric(Accuracy(), ignore_index=-1)
        self.edge_accuracy = IgnoreIndexMetric(Accuracy(), ignore_index=-1)
        self.fraction_solved = FractionSolved(ignore_index=-1)
        self.f1 = IgnoreIndexMetric(F1(num_classes=2), ignore_index=-1)
        self._metrics = {
            "node_accuracy": self.node_accuracy,
            "edge_accuracy": self.edge_accuracy,
            "fraction_solved": self.fraction_solved,
            "f1": self.f1,
        }
        self._metrics = self._prepend_dict_keys(self._metrics, prepend)

    def forward(self, node_pred, edge_pred, node_target, edge_target, batch):
        self.node_accuracy(node_pred, node_target)
        self.edge_accuracy(edge_pred, edge_target)
        self.fraction_solved(node_pred, node_target, batch)
        self.f1(node_pred, node_target)
        return self._metrics

    def _prepend_dict_keys(self, dictionary, prepend):
        return {
            "{}_{}".format(prepend, key): value for key, value in dictionary.items()
        }


class GraphModel(pl.LightningModule):
    """
    Pytorch Lightning Module with the model and training
    abstractions. See https://www.pytorchlightning.ai/ .
    """

    def __init__(self):
        super().__init__()
        self.model = KGCN(
            node_types=node_types,
            edge_types=edge_types,
            categorical_attributes=CATEGORICAL_ATTRIBUTES,
            continuous_attributes=CONTINUOUS_ATTRIBUTES,
            edge_output_size=2,
            node_output_size=2,
        )

        self.loss_function = MultiStepLoss(torch.nn.CrossEntropyLoss(ignore_index=-1))
        self.train_metrics = Metrics(prepend="train")
        self.val_metrics = Metrics(prepend="val")

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        node_logits, edge_logits = self.model(batch, steps=5, return_all_steps=True)
        node_loss = self.loss_function(node_logits, batch.y, steps=5)
        edge_loss = self.loss_function(edge_logits, batch.y_edge, steps=5)
        loss = (node_loss + edge_loss) * 0.5

        self.log_dict(
            self.train_metrics(
                F.softmax(node_logits[:, :, -1]),
                F.softmax(edge_logits[:, :, -1]),
                batch.y,
                batch.y_edge,
                batch.batch,
            ),
        )
        self.log("train_loss", loss)
        self.log("train_node_loss", node_loss)
        self.log("train_edge_loss", edge_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        node_logits, edge_logits = self.model(batch)
        node_loss = self.loss_function(node_logits, batch.y)
        edge_loss = self.loss_function(edge_logits, batch.y_edge)
        loss = (node_loss + edge_loss) * 0.5

        self.log_dict(
            self.val_metrics(
                F.softmax(node_logits),
                F.softmax(edge_logits),
                batch.y,
                batch.y_edge,
                batch.batch,
            ),
        )
        self.log("val_loss", loss)
        self.log("val_node_loss", node_loss)
        self.log("val_edge_loss", edge_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)
        return optimizer


model = GraphModel()

# train and test split
example_indices = list(range(80))
val_indices = list(range(80, 100))

# create the transformation applied to
# the networkx graph before it is ingested
# into Pytorch Geometric
networkx_transform = StandardKGCNNetworkxTransform(
    node_types=node_types,
    edge_types=edge_types,
    target_name="solution",
    obfuscate=TYPES_AND_ROLES_TO_OBFUSCATE,
    categorical=CATEGORICAL_ATTRIBUTES,
    continuous=CONTINUOUS_ATTRIBUTES,
)


# Create a dataset
grakn_dataset = GraknPytorchGeometricDataSet(
    example_indices=example_indices,
    get_query_handles_for_id=get_query_handles,
    infer=True,
    uri="localhost:1729",
    database="diagnosis",
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
    uri="localhost:1729",
    database="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

# Use the test dataset to create a validation dataloader
val_dataloader = DataLoader(val_dataset, batch_size=5, num_workers=0, shuffle=True)

# running one batch though the network
# so that lazy layers can be initialized.
model(next(iter(train_dataloader)))

if __name__ == "__main__":
    print(model)
    trainer = pl.Trainer(fast_dev_run=False)
    trainer.fit(
        model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
    )
