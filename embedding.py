"""Pytorch version of kglib/kgcn/models/embedding.py trying to keep interface similar"""

import torch
import torch.nn as nn
from attribute import ContinuousAttribute, CategoricalAttribute, BlankAttribute


class ThingEmbedder(nn.Module):
    def __init__(
        self,
        node_types,
        type_embedding_dim,
        attr_embedding_dim,
        categorical_attributes,
        continuous_attributes,
    ):
        """

        :param node_types:
        :param type_embedding_dim: size of the type embedding
        :param attr_embedding_dim: size of the attribute embedding
        :param categorical_attributes: dict of {"attribute_name": ["catergory_1", "category_2", ...]}
        :param continuous_attributes: dict of {"attribute_name": (min_value, max_value)}
        """

        super(ThingEmbedder, self).__init__()
        self.type_embedder = nn.Embedding(
            num_embeddings=len(node_types), embedding_dim=type_embedding_dim
        )
        self.attr_embeddder = TypewiseEncoder(
            node_types=node_types,
            embedding_dim=attr_embedding_dim,
            categorical_attributes=categorical_attributes,
            continuous_attributes=continuous_attributes,
        )

    def forward(self, X):
        preexistence_feat = X[:, 0:1]
        type_feat = self.type_embedder(X[:, 1].long())
        attribute_feat = self.attr_embeddder(X[:, 1], X[:, 2:])
        print("sizes: ", preexistence_feat.shape, type_feat.shape, attribute_feat.shape)
        return torch.cat([preexistence_feat, type_feat, attribute_feat], dim=-1)


class TypewiseEncoder(nn.Module):
    def __init__(
        self, node_types, embedding_dim, categorical_attributes, continuous_attributes
    ):
        super(TypewiseEncoder, self).__init__()
        self._node_types = node_types
        self._embedding_dim = embedding_dim

        self.embedders = [BlankAttribute(attr_embedding_dim=self._embedding_dim)] * len(node_types)
        self.construct_categorical_embedders(categorical_attributes)
        self.construct_continuous_embedders(continuous_attributes)
        self.embedders = nn.ModuleList(self.embedders)

    def forward(self, types, features):
        shape = (features.size(0), self._embedding_dim)
        print(shape)
        return torch.zeros(shape)

    def construct_categorical_embedders(self, categorical_attributes):
        if not categorical_attributes:
            return
        for attribute_type, categories in categorical_attributes.items():
            attr_typ_index = self._node_types.index(attribute_type)
            embedder = CategoricalAttribute(
                num_categories=len(categories), attr_embedding_dim=self._embedding_dim
            )
            self.embedders[attr_typ_index] = embedder

    def construct_continuous_embedders(self, continuous_attributes):
        if not continuous_attributes:
            return
        for attribute_type in continuous_attributes.keys():
            attr_typ_index = self._node_types.index(attribute_type)
            embedder = ContinuousAttribute(attr_embedding_dim=self._embedding_dim)
            self.embedders[attr_typ_index] = embedder
