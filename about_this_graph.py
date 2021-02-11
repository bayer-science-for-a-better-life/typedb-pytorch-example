from kglib.utils.grakn.type.type import get_thing_types, get_role_types

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


# TODO: the following should probably go somewhere else:


def get_node_types(session):
    with session.transaction().read() as tx:
        # Change the terminology here onwards from thing -> node and role -> edge
        node_types = get_thing_types(tx)
        [node_types.remove(el) for el in TYPES_TO_IGNORE]
    print(f"Found node types: {node_types}")
    return node_types


def get_edge_types(session):
    with session.transaction().read() as tx:
        edge_types = get_role_types(tx)
        [edge_types.remove(el) for el in ROLES_TO_IGNORE]
    print(f"Found edge types: {edge_types}")
    return edge_types
