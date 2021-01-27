import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv

from grakn_pytorch_geometric.dataloader import GraknPytorchGeometricDataSet
from queries import get_query_handles
from transforms import networkx_transform


def train(example_indices):
    grakn_dataset = GraknPytorchGeometricDataSet(
        example_indices=list(range(20)),
        get_query_handles_for_id=get_query_handles,
        infer=True,
        uri="localhost:48555",
        keyspace="diagnosis",
        networkx_transform=networkx_transform
    )


    dataloader = DataLoader(grakn_dataset, batch_size=2)

    for data in dataloader:
        print(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    example_inices = list(range(20))
    train(example_inices)
