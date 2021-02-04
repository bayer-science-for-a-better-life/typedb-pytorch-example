"""Pytorch version of kglib/kgcn/models/embedding.py trying to keep interface similar"""

import torch
import torch.nn as nn
from attribute import ContinuousAttribute, CategoricalAttribute, BlankAttribute


class Embedder(nn.Module):
    def __init__(
        self,
        types,
        encode_preexistence=True,
        type_embedding_dim=0,
        attr_embedding_dim=0,
        categorical_attributes=None,
        continuous_attributes=None,
    ):
        """
        Embed the 3-vector (preexistence, type, value) from Grakn. Each type
        has its own embedder to embed the value.

        :param types: list of node or edge types
        :param encode_preexistence: to start each embedding with the existence bit
        :param type_embedding_dim: size of the type embedding
        :param attr_embedding_dim: size of the attribute embedding
        :param categorical_attributes: dict of {"attribute_name": ["catergory_1", "category_2", ...]}
        :param continuous_attributes: dict of {"attribute_name": (min_value, max_value)}
        """

        super(Embedder, self).__init__()

        self.type_embedder = None
        self.attr_embeddder =None

        self._encode_preexistence = encode_preexistence
        if type_embedding_dim:
            self.type_embedder = nn.Embedding(
                num_embeddings=len(types), embedding_dim=type_embedding_dim
            )
        if attr_embedding_dim:
            self.attr_embeddder = TypewiseEncoder(
                types=types,
                embedding_dim=attr_embedding_dim,
                categorical_attributes=categorical_attributes,
                continuous_attributes=continuous_attributes,
            )

    def forward(self, X):
        embedding = [torch.zeros((X.size(0), 0))]
        if self._encode_preexistence:
            embedding.append(X[:, 0:1])
        if self.type_embedder:
            embedding.append(self.type_embedder(X[:, 1].long()))
        if self.attr_embeddder:
            embedding.append(self.attr_embeddder(X[:, 1], X[:, 2:]))
        return torch.cat(embedding, dim=-1)


class TypewiseEncoder(nn.Module):
    def __init__(
        self, types, embedding_dim, categorical_attributes, continuous_attributes
    ):
        super(TypewiseEncoder, self).__init__()
        self._types = types
        self._embedding_dim = embedding_dim
        self._encoders_for_types = [BlankAttribute(attr_embedding_dim=self._embedding_dim)] * len(types)
        self._construct_categorical_embedders(categorical_attributes)
        self._construct_continuous_embedders(continuous_attributes)
        self._encoders_for_types = nn.ModuleList(self._encoders_for_types)

    def forward(self, types, features):
        shape = (features.size(0), self._embedding_dim)
        types = types.long()
        encoded_features = torch.zeros(shape)

        for i, encoder in enumerate(self._encoders_for_types):
            mask = types == i
            encoded_features[mask] = encoder(features[mask])
        return encoded_features

    def _construct_categorical_embedders(self, categorical_attributes):
        if not categorical_attributes:
            return
        for attribute_type, categories in categorical_attributes.items():
            attr_typ_index = self._types.index(attribute_type)
            embedder = CategoricalAttribute(
                num_categories=len(categories), attr_embedding_dim=self._embedding_dim
            )
            self._encoders_for_types[attr_typ_index] = embedder

    def _construct_continuous_embedders(self, continuous_attributes):
        if not continuous_attributes:
            return
        for attribute_type in continuous_attributes.keys():
            attr_typ_index = self._types.index(attribute_type)
            embedder = ContinuousAttribute(attr_embedding_dim=self._embedding_dim)
            self._encoders_for_types[attr_typ_index] = embedder
