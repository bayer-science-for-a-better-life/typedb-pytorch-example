"""
This is a very minimalistic example of using the GraknPytorchGeometricDataSet class
to train a network with a very basic GCN. This is not for production but to
give a very minimal example.
"""

import torch
import torch.nn.functional as F
from grakn.client import GraknClient
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from kglib.kgcn.examples.diagnosis.diagnosis import get_query_handles

from grakn_pytorch_geometric.data.dataset import GraknPytorchGeometricDataSet
from grakn_pytorch_geometric.data.transforms import StandardKGCNNetworkxTransform

from about_this_graph import (
    get_node_types,
    get_edge_types,
    CATEGORICAL_ATTRIBUTES,
    CONTINUOUS_ATTRIBUTES,
    TYPES_AND_ROLES_TO_OBFUSCATE,
)


client = GraknClient(uri="localhost:48555")
session = client.session(keyspace="diagnosis")
node_types = get_node_types(session)
edge_types = get_edge_types(session)

example_indices = list(range(100))

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
    uri="localhost:48555",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

# Use the Dataset to create a Pytorch (geometric) DataLoader
dataloader = DataLoader(grakn_dataset, batch_size=5, num_workers=0, shuffle=True)

# Define a  los function.
# Samples of class 0 are ignored for loss
# calculation (element already exists in graph).
# this is the done in the diagnosis examplein kgcn as well.
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)


class Net(torch.nn.Module):
    """Very Simple GCN with two graph convolution layers.
    This model only takes into account node features.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(grakn_dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():

    for epoch in range(20):
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            model.train()
            prediction = model(data)
            loss = loss_function(prediction, data.y)
            print("epoch: {}, step: {}, loss: {}".format(epoch, i, loss))
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    train()
