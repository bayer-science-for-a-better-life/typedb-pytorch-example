"""
Example in DGL that I started but did not finish. Focussed on Pytorch Geometric for now.
"""

import torch
import dgl
from grakn.client import GraknClient
from grakn_dataloading.networkx import GraknNetworkxDataSet
from grakn_pytorch_geometric.data.transforms import StandardKGCNNetworkxTransform
from torch.utils.data import DataLoader
from kglib.kgcn.examples.diagnosis.diagnosis import get_query_handles

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

example_indices = list(range(20))


class GraknDGLDataSet(torch.utils.data.Dataset):
    """
    Pytorch Geometric DataSet:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset
    Using the more generic GraknNetworkxDataSet but returns Pytorch Geometric Data objects
    instead of networkx graphs.

    """

    def __init__(
        self,
        example_indices,
        get_query_handles_for_id,
        keyspace,
        uri="localhost:48555",
        infer=True,
        networkx_transform=None,
        caching=False,
    ):
        super(GraknDGLDataSet, self).__init__()
        self._networkx_dataset = GraknNetworkxDataSet(
            example_indices=example_indices,
            get_query_handles_for_id=get_query_handles_for_id,
            keyspace=keyspace,
            uri=uri,
            infer=infer,
            transform=networkx_transform,
        )

        self.caching = caching
        self._cache = {}

    def __len__(self):
        return len(self._networkx_dataset)

    def __getitem__(self, idx):
        if self.caching and idx in self._cache:
            return self._cache[idx]
        graph = dgl.from_networkx(self._networkx_dataset[idx])
        if self.caching:
            self._cache[idx] = graph
        return graph


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

grakn_dataset = GraknDGLDataSet(
    example_indices=example_indices,
    get_query_handles_for_id=get_query_handles,
    infer=True,
    uri="localhost:48555",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True,
)

dataloader = DataLoader(
    grakn_dataset, batch_size=2, num_workers=0, shuffle=True, collate_fn=dgl.batch
)


def train():
    for epoch in range(20):
        for i, data in enumerate(dataloader):
            print(i, data)


if __name__ == "__main__":

    train()
