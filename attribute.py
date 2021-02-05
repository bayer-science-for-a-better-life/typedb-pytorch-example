import abc
import torch
import torch.nn as nn


class Attribute(nn.Module, abc.ABC):
    """
    Abstract base class for Attribute value embedding models
    """
    def __init__(self, attr_embedding_dim, name=None):
        super(Attribute, self).__init__()
        self._attr_embedding_dim = attr_embedding_dim
        self._type_name = name

    def extra_repr(self) -> str:
        if self._type_name:
            return "name={}, out_features={}".format(
                self._type_name, self._attr_embedding_dim
            )
        return "out_features={}".format(self._attr_embedding_dim)


class ContinuousAttribute(Attribute):
    def __init__(self, attr_embedding_dim, name=None):
        super(ContinuousAttribute, self).__init__(attr_embedding_dim, name)
        # for now input is size 1 (when pytorch 1.8 comes out can be made dependent on input)
        # a three layer MLP is used like in KGCN
        # todo: when pytorch 1.8 comes out, change first Linear to LazyLinear
        self.embedder = nn.Sequential(
            nn.Linear(1, attr_embedding_dim),
            nn.ReLU(),
            nn.Linear(attr_embedding_dim, attr_embedding_dim),
            nn.ReLU(),
            nn.Linear(attr_embedding_dim, attr_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, attribute_value):
        # todo: tensorboard histogram here of continuous attributes
        embedding = self.embedder(attribute_value)
        # todo: tensorboard histogram of continuous attibute embedding
        return embedding


class CategoricalAttribute(Attribute):
    def __init__(self, num_categories, attr_embedding_dim, name=None):
        super(CategoricalAttribute, self).__init__(attr_embedding_dim, name)
        self.embedder = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=attr_embedding_dim
        )

    def forward(self, attribute_value):
        # todo: tensorboard histogram here of continuous attributes
        embedding = self.embedder(attribute_value.squeeze().long())
        # todo: tensorboard histogram of continuous attibute embedding
        return embedding


class BlankAttribute(Attribute):
    def __init__(self, attr_embedding_dim, name=None):
        super(BlankAttribute, self).__init__(attr_embedding_dim, name)

    def forward(self, attribute_value):
        shape = (attribute_value.size(0), self._attr_embedding_dim)
        return torch.zeros(shape)
