import networkx as nx

from grakn.client import GraknClient

from kglib.utils.grakn.type.type import get_thing_types, get_role_types
from kglib.utils.graph.iterate import (
    multidigraph_node_data_iterator,
    multidigraph_data_iterator,
    multidigraph_edge_data_iterator,
)
from kglib.kgcn.pipeline.encode import encode_types, encode_values, stack_features
from kglib.kgcn.pipeline.utils import duplicate_edges_in_reverse

# All

# Categorical Attribute types and the values of their categories
CATEGORICAL_ATTRIBUTES = {
    "name": [
        "Diabetes Type II",
        "Multiple Sclerosis",
        "Blurred vision",
        "Fatigue",
        "Cigarettes",
        "Alcohol",
    ]
}
# Continuous Attribute types and their min and max values
CONTINUOUS_ATTRIBUTES = {"severity": (0, 1), "age": (7, 80), "units-per-week": (3, 29)}


TYPES_TO_IGNORE = [
    "candidate-diagnosis",
    "example-id",
    "probability-exists",
    "probability-non-exists",
    "probability-preexists",
]
ROLES_TO_IGNORE = ["candidate-patient", "candidate-diagnosed-disease"]


# The learner should see candidate relations the same as the ground truth relations, so adjust these candidates to
# look like their ground truth counterparts
TYPES_AND_ROLES_TO_OBFUSCATE = {
    "candidate-diagnosis": "diagnosis",
    "candidate-patient": "patient",
    "candidate-diagnosed-disease": "diagnosed-disease",
}

client = GraknClient(uri="localhost:48555")
session = client.session(keyspace="diagnosis")

with session.transaction().read() as tx:
    """
    Directly taken from diagnosis.py from kglib example.
    """
    # Change the terminology here onwards from thing -> node and role -> edge
    node_types = get_thing_types(tx)
    [node_types.remove(el) for el in TYPES_TO_IGNORE]

    edge_types = get_role_types(tx)
    [edge_types.remove(el) for el in ROLES_TO_IGNORE]
    print(f"Found node types: {node_types}")
    print(f"Found edge types: {edge_types}")


def networkx_transform(graph):
    """Transform of the networkx graph as it comes out of Grakn
    to a networkx graph that pytorch geometric likes to ingest.
    Now this is very much geared to pytorch geometric especially
    because I set the attribute names to things like "x" and
    "edge_attr" which are the standard names in pytorch geometric.

    One thing I encountered when trying to load the graph form the
    original kglib example directly in pytorch geometric is that
    since in the original example the feature vector on the nodes
    and the edges were both called "features", the stock function
    from pytorch geometric: torch_geometric.utils.from_networkx
    does not deal with this well (it ends up overwriting node
    features with edge features).

    :arg graph: networkx graph object
    :returns: networkx graph object

    """

    obfuscate_labels(graph, TYPES_AND_ROLES_TO_OBFUSCATE)
    # Encode attribute values as number
    graph = encode_values(graph, CATEGORICAL_ATTRIBUTES, CONTINUOUS_ATTRIBUTES)

    indexed_graph = nx.convert_node_labels_to_integers(graph, label_attribute="concept")
    graph = duplicate_edges_in_reverse(indexed_graph)

    # Node or Edge Type as int
    graph = encode_types(graph, multidigraph_node_data_iterator, node_types)
    graph = encode_types(graph, multidigraph_edge_data_iterator, edge_types)

    for data in multidigraph_node_data_iterator(graph):
        features = create_feature_vector(data)
        target = data["solution"]
        data.clear()
        data["x"] = features
        data["y"] = target

    for data in multidigraph_edge_data_iterator(graph):
        features = create_feature_vector(data)
        target = data["solution"]
        data.clear()
        data["edge_attr"] = features
        data["y_edge"] = target

    return graph


def create_feature_vector(node_or_edge_data_dict):
    """Make a feature 3-dimensional feature vector,

    Factored out of kglib.kgcn.pipeline.encode.create_input_graph.

    Args:
        node_or_edge_dict: the dict coming describing a node or edge
        obtained from an element of graph.nodes(data=True) or graph.edges(data=True)
        of a networkx graph.

    Returns:
        Numpy array (vector) of stacked features

    """
    if node_or_edge_data_dict["solution"] == 0:
        preexists = 1
    else:
        preexists = 0
    features = stack_features(
        [
            preexists,
            node_or_edge_data_dict["categorical_type"],
            node_or_edge_data_dict["encoded_value"],
        ]
    )
    return features


def obfuscate_labels(graph, types_and_roles_to_obfuscate):
    """Taken directly from diagnosis.py from the kglib example"""
    # Remove label leakage - change type labels that indicate candidates into non-candidates
    for data in multidigraph_data_iterator(graph):
        for label_to_obfuscate, with_label in types_and_roles_to_obfuscate.items():
            if data["type"] == label_to_obfuscate:
                data.update(type=with_label)
                break
