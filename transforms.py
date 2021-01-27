import networkx as nx

from grakn.client import GraknClient

from kglib.utils.grakn.type.type import get_thing_types, get_role_types
from kglib.utils.graph.iterate import multidigraph_node_data_iterator, multidigraph_data_iterator, \
    multidigraph_edge_data_iterator
from kglib.kgcn.pipeline.encode import encode_types, create_input_graph, create_target_graph, encode_values
from kglib.kgcn.pipeline.utils import duplicate_edges_in_reverse



# Categorical Attribute types and the values of their categories
CATEGORICAL_ATTRIBUTES = {'name': ['Diabetes Type II', 'Multiple Sclerosis', 'Blurred vision', 'Fatigue', 'Cigarettes',
                                   'Alcohol']}
# Continuous Attribute types and their min and max values
CONTINUOUS_ATTRIBUTES = {'severity': (0, 1), 'age': (7, 80), 'units-per-week': (3, 29)}


TYPES_TO_IGNORE = ['candidate-diagnosis', 'example-id', 'probability-exists', 'probability-non-exists', 'probability-preexists']
ROLES_TO_IGNORE = ['candidate-patient', 'candidate-diagnosed-disease']


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
    # Change the terminology here onwards from thing -> node and role -> edge
    node_types = get_thing_types(tx)
    [node_types.remove(el) for el in TYPES_TO_IGNORE]

    edge_types = get_role_types(tx)
    [edge_types.remove(el) for el in ROLES_TO_IGNORE]
    print(f'Found node types: {node_types}')
    print(f'Found edge types: {edge_types}')


def networkx_transform(graph):
    '''Transform of the graph as it comes out
       of Grakn.
    '''

    obfuscate_labels(graph, TYPES_AND_ROLES_TO_OBFUSCATE)

    # Encode attribute values
    graph = encode_values(graph, CATEGORICAL_ATTRIBUTES, CONTINUOUS_ATTRIBUTES)

    indexed_graph = nx.convert_node_labels_to_integers(graph, label_attribute='concept')
    graph = duplicate_edges_in_reverse(indexed_graph)

    graph = encode_types(graph, multidigraph_node_data_iterator, node_types)
    graph = encode_types(graph, multidigraph_edge_data_iterator, edge_types)

    #return graph


    input_graph = create_input_graph(graph)
    target_graph = create_target_graph(graph)

    return input_graph

    #return input_graph, target_graph


def obfuscate_labels(graph, types_and_roles_to_obfuscate):
    # Remove label leakage - change type labels that indicate candidates into non-candidates
    for data in multidigraph_data_iterator(graph):
        for label_to_obfuscate, with_label in types_and_roles_to_obfuscate.items():
            if data["type"] == label_to_obfuscate:
                data.update(type=with_label)
                break