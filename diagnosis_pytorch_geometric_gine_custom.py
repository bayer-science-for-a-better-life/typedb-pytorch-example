"""
An example where I took the GINEConv layer and changed it a little bit so that
also edge features are transformed (and therefore edge classification becomes
possible in the way it is done in kgcn. It is not completely how I want it yet.
"""

from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_sparse import SparseTensor

import torch.nn.functional as F
from torch_geometric.data import DataLoader
from kglib.kgcn.examples.diagnosis.diagnosis import get_query_handles
from grakn_dataloading.pytorch_geometric import GraknPytorchGeometricDataSet
from transforms import networkx_transform


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


class GINEdgeConv(MessagePassing):
    r"""The modified from GINEConv in PyTorch Geometric.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn_node (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        nn_edge (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.

        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        nn_node: Callable,
        nn_edge: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(GINEdgeConv, self).__init__(**kwargs)
        self.nn_node = nn_node
        self.nn_edge = nn_edge
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_node)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""

        src, dst = edge_index
        edge_repr = torch.cat([x[src], edge_attr, x[dst]], dim=-1)
        edge_repr = self.nn_edge(edge_repr)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn_node(out), edge_repr

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn_node)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GINEdgeConv(
            nn_node=nn.Sequential(
                nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            ),
            nn_edge=nn.Sequential(
                nn.Linear(3 + 3 + 3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            ),
        )
        self.conv2 = GINEdgeConv(
            nn_node=nn.Sequential(
                nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            ),
            nn_edge=nn.Sequential(
                nn.Linear(16 + 16 + 16, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            ),
        )
        self.conv3 = GINEdgeConv(
            nn_node=nn.Sequential(
                nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            ),
            nn_edge=nn.Sequential(
                nn.Linear(16 + 16 + 16, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            ),
        )
        self.node_net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 3))
        self.edge_net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, data):
        x_node, x_edge, edge_index, = (
            data.x,
            data.edge_attr,
            data.edge_index,
        )

        x_node, x_edge = self.conv1(x_node, edge_index, x_edge)
        x_node, x_edge = self.conv2(x_node, edge_index, x_edge)
        x_node, x_edge = self.conv3(x_node, edge_index, x_edge)
        x_node = self.node_net(x_node)
        x_edge = self.edge_net(x_edge)
        return x_node, x_edge


model = Net()
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
