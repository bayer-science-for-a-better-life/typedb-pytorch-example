"""
KGCN in Pytorch Geometric
"""

import torch
from torch_geometric.data import DataLoader

from grakn.client import GraknClient
from grakn.rpc.session import SessionType

from kglib.kgcn_data_loader.transform.standard_kgcn_transform import StandardKGCNNetworkxTransform
from kglib.kgcn_data_loader.utils import get_edge_types_for_training, get_node_types_for_training

from grakn_pytorch_geometric.data.dataset import GraknPytorchGeometricDataSet
from grakn_pytorch_geometric.models.core import KGCN
from grakn_pytorch_geometric.utils import metrics

from about_this_graph import (
    get_query_handles,
    CATEGORICAL_ATTRIBUTES,
    CONTINUOUS_ATTRIBUTES,
    TYPES_AND_ROLES_TO_OBFUSCATE,
    TYPES_TO_IGNORE,
    ROLES_TO_IGNORE
)


client = GraknClient.core(address="localhost:1729")
session = client.session(SessionType.DATA, database="diagnosis")
node_types = get_node_types_for_training(session, TYPES_TO_IGNORE)
edge_types = get_edge_types_for_training(session, ROLES_TO_IGNORE)

# train and test split
example_indices = list(range(80))
test_indices = list(range(80, 100))

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
    uri="localhost:1729",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

# Use the test dataset to create a test dataloader
test_loader = DataLoader(test_dataset, batch_size=5, num_workers=0, shuffle=True)

# Define a  los function.
# Samples of class 0 are ignored for loss
# calculation (element already exists in graph).
# this is the done in the diagnosis examplein kgcn_tensorflow as well.
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
            acc = metrics.existence_accuracy(node_prediction, data.y, ignore_index=0)
            fraction_solved = metrics.fraction_solved(
                node_prediction, data.y, data.batch, ignore_index=0
            )
            print(
                "epoch: {}, step: {}, loss {}, node loss: {}, edge_loss: {}, node accuracy: {}, fraction_solved {}".format(
                    epoch, i, loss, node_loss, edge_loss, acc, fraction_solved
                )
            )

            loss.backward()
            optimizer.step()

        test_node_loss = 0
        test_edge_loss = 0
        test_acc_sum = 0
        fraction_solved = 0
        for i, data in enumerate(test_loader):
            model.eval()
            node_prediction, edge_prediction = model(data)
            test_node_loss += loss_function(node_prediction, data.y)
            test_edge_loss += loss_function(edge_prediction, data.y_edge)
            test_acc_sum += metrics.existence_accuracy(
                node_prediction, data.y, ignore_index=0
            )
            fraction_solved += metrics.fraction_solved(
                node_prediction, data.y, data.batch, ignore_index=0
            )

        print(
            "epoch: {},Test, loss {}, node loss: {}, edge_loss: {}, node accuracy: {}, fraction_solved {}".format(
                epoch,
                (test_node_loss + test_edge_loss) / (2 * (i + 1)),
                test_node_loss / (i + 1),
                test_edge_loss / (i + 1),
                test_acc_sum / (i + 1),
                fraction_solved / (i + 1),
            )
        )


if __name__ == "__main__":
    train()
