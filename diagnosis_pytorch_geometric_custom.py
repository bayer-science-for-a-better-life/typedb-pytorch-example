import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from kglib.kgcn.examples.diagnosis.diagnosis import get_query_handles
from grakn_dataloading.pytorch_geometric import GraknPytorchGeometricDataSet
from transforms import networkx_transform


example_indices = list(range(80))
test_indices = list(range(80, 100))

grakn_dataset = GraknPytorchGeometricDataSet(
    example_indices=example_indices,
    get_query_handles_for_id=get_query_handles,
    infer=True,
    uri="localhost:48555",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

# TODO: multiple workers crashes
dataloader = DataLoader(grakn_dataset, batch_size=5, num_workers=0, shuffle=True)

test_dataset = GraknPytorchGeometricDataSet(
    example_indices=test_indices,
    get_query_handles_for_id=get_query_handles,
    infer=True,
    uri="localhost:48555",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

# TODO: multiple workers crashes
test_loader = DataLoader(test_dataset, batch_size=5, num_workers=0, shuffle=True)

# Samples of class 0 are ignored for loss
# calculation (element already exists in graph).
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)


class TestConv(MessagePassing):
    """Just checking out how to incorporate edge features."""

    def __init__(self, in_channels, out_channels):
        super(TestConv, self).__init__(aggr="add")  # "Add" aggregation (Step 5).
        self.lin_node = torch.nn.Linear(in_channels, out_channels)
        self.lin_edge = torch.nn.Linear(2 * out_channels + in_channels, out_channels)

    def forward(self, x, edge_attr, edge_index):
        x = self.lin_node(x)
        src, dst = edge_index
        deg = degree(dst, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        edge_repr = torch.cat([x[src], edge_attr, x[dst]], dim=-1)
        edge_repr = self.lin_edge(edge_repr)
        return self.propagate(edge_index, x=x, norm=norm), edge_repr

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = TestConv(3, 16)
        self.conv2 = TestConv(16, 16)
        self.node_net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 3))
        self.edge_net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, data):
        x_node, x_edge, edge_index, = (
            data.x,
            data.edge_attr,
            data.edge_index,
        )
        x_node, x_edge = self.conv1(x_node, x_edge, edge_index)
        x_node = F.relu(x_node)
        x_edge = F.relu(x_edge)
        x_node, x_edge = self.conv2(x_node, x_edge, edge_index)
        x_node = F.relu(x_node)
        x_edge = F.relu(x_edge)
        x_node = self.node_net(x_node)
        x_edge = self.edge_net(x_edge)
        return x_node, x_edge


model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


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
