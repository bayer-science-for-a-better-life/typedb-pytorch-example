import torch
import torch.nn as nn


class ContinuousAttribute(nn.Module):
    def __init__(self, attr_embedding_dim):
        super(ContinuousAttribute, self).__init__()
        # for now input is size 1 (when pytorch 1.8 comes out can be made dependent on input)
        # todo: when pytorch 1.8 comes out, change first Linear to LazyLinear
        self.embedder = nn.Sequential(nn.Linear(1, attr_embedding_dim), nn.ReLU())

    def forward(self, attribute_value):
        # todo: tensorboard histogram here of continuous attributes
        embedding = self.embedder(attribute_value)
        # todo: tensorboard histogram of continuous attibute embedding
        return embedding


class CategoricalAttribute(nn.Module):
    def __init__(self, num_categories, attr_embedding_dim):
        super(CategoricalAttribute, self).__init__()
        self.embedder = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=attr_embedding_dim
        )

    def forward(self, attribute_value):
        # todo: tensorboard histogram here of continuous attributes
        embedding = self.embedder(attribute_value.squeeze().long())
        # todo: tensorboard histogram of continuous attibute embedding
        return embedding


class BlankAttribute(nn.Module):
    def __init__(self, attr_embedding_dim):
        super(BlankAttribute, self).__init__()
        self._attr_embedding_dim = attr_embedding_dim

    def forward(self, attribute_value):
        shape = (attribute_value.size(0), self._attr_embedding_dim)
        return torch.zeros(shape)
