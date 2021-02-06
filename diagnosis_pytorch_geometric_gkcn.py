"""
An example where I took the GINEConv layer and changed it a little bit so that
also edge features are transformed (and therefore edge classification becomes
possible in the way it is done in kgcn. It is not completely how I want it yet.
"""

import torch
from torch_geometric.data import DataLoader
from kglib.kgcn.examples.diagnosis.diagnosis import get_query_handles
from grakn_pytorch_geometric.data.dataset import GraknPytorchGeometricDataSet
from grakn_pytorch_geometric.models.core import KGCN

from transforms import (
    networkx_transform,
    node_types,
    edge_types,
    CATEGORICAL_ATTRIBUTES,
    CONTINUOUS_ATTRIBUTES,
)


# train and test split
example_indices = list(range(80))
test_indices = list(range(80, 100))


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
dataloader = DataLoader(grakn_dataset, batch_size=5, num_workers=0, shuffle=True)

# Create a test dataset
test_dataset = GraknPytorchGeometricDataSet(
    example_indices=test_indices,
    get_query_handles_for_id=get_query_handles,
    infer=True,
    uri="localhost:48555",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

# Use the test dataset to create a test dataloader
test_loader = DataLoader(test_dataset, batch_size=5, num_workers=0, shuffle=True)

# Define a  los function.
# Samples of class 0 are ignored for loss
# calculation (element already exists in graph).
# this is the done in the diagnosis examplein kgcn as well.
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)

model = KGCN(
    node_types=node_types,
    edge_types=edge_types,
    categorical_attributes=CATEGORICAL_ATTRIBUTES,
    continuous_attributes=CONTINUOUS_ATTRIBUTES,
)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


def train():
    for epoch in range(1000):
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            model.train()
            node_prediction, edge_prediction = model(data)
            node_loss = loss_function(node_prediction, data.y)
            edge_loss = loss_function(edge_prediction, data.y_edge)
            loss = (node_loss + edge_loss) * 0.5
            print(
                "epoch: {}, step: {}, loss {}, node loss: {}, edge_loss: {}".format(
                    epoch, i, loss, node_loss, edge_loss
                )
            )
            loss.backward()
            optimizer.step()

        test_node_loss = 0
        test_edge_loss = 0
        for i, data in enumerate(test_loader):
            model.eval()
            node_prediction, edge_prediction = model(data)
            test_node_loss += loss_function(node_prediction, data.y)
            test_edge_loss += loss_function(edge_prediction, data.y_edge)
        print(
            "epoch: {},Test, loss {}, node loss: {}, edge_loss: {}".format(
                epoch,
                (test_node_loss + test_edge_loss) / (2 * (i + 1)),
                test_node_loss / (i + 1),
                test_edge_loss / (i + 1),
            )
        )


if __name__ == "__main__":
    train()
