import dgl
from torch.utils.data import DataLoader
from kglib.kgcn.examples.diagnosis.diagnosis import get_query_handles
from grakn_dataloading.dgl import GraknDGLDataSet
from transforms import networkx_transform


example_indices = list(range(20))

grakn_dataset = GraknDGLDataSet(
    example_indices=example_indices,
    get_query_handles_for_id=get_query_handles,
    infer=True,
    uri="localhost:48555",
    keyspace="diagnosis",
    networkx_transform=networkx_transform,
    caching=True
)

dataloader = DataLoader(grakn_dataset, batch_size=2, num_workers=0, shuffle=True, collate_fn=dgl.batch)

def train():
    for epoch in range(20):
        for i, data in enumerate(dataloader):
            print(i, data)


if __name__ == '__main__':

    train()