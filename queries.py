import inspect

from kglib.utils.graph.query.query_graph import QueryGraph
from kglib.utils.graph.thing.queries_to_graph import build_graph_from_queries


# Existing elements in the graph are those that pre-exist in the graph, and should be predicted to continue to exist
PREEXISTS = 0
# Candidates are neither present in the input nor in the solution, they are negative samples
CANDIDATE = 1
# Elements to infer are the graph elements whose existence we want to predict to be true, they are positive samples
TO_INFER = 2


def get_query_handles(example_id):
    """
    Creates an iterable, each element containing a Graql query, a function to sample the answers, and a QueryGraph
    object which must be the Grakn graph representation of the query. This tuple is termed a "query_handle"

    Args:
        example_id: A uniquely identifiable attribute value used to anchor the results of the queries to a specific
                    subgraph

    Returns:
        query handles
    """

    # === Hereditary Feature ===
    hereditary_query = inspect.cleandoc(
        f"""match
           $p isa person, has example-id {example_id};
           $par isa person;
           $ps(child: $p, parent: $par) isa parentship;
           $diag(patient:$par, diagnosed-disease: $d) isa diagnosis;
           $d isa disease, has name $n;
           get;"""
    )

    vars = p, par, ps, d, diag, n = "p", "par", "ps", "d", "diag", "n"
    hereditary_query_graph = (
        QueryGraph()
        .add_vars(vars, PREEXISTS)
        .add_role_edge(ps, p, "child", PREEXISTS)
        .add_role_edge(ps, par, "parent", PREEXISTS)
        .add_role_edge(diag, par, "patient", PREEXISTS)
        .add_role_edge(diag, d, "diagnosed-disease", PREEXISTS)
        .add_has_edge(d, n, PREEXISTS)
    )

    # === Consumption Feature ===
    consumption_query = inspect.cleandoc(
        f"""match
           $p isa person, has example-id {example_id};
           $s isa substance, has name $n;
           $c(consumer: $p, consumed-substance: $s) isa consumption, 
           has units-per-week $u; get;"""
    )

    vars = p, s, n, c, u = "p", "s", "n", "c", "u"
    consumption_query_graph = (
        QueryGraph()
        .add_vars(vars, PREEXISTS)
        .add_has_edge(s, n, PREEXISTS)
        .add_role_edge(c, p, "consumer", PREEXISTS)
        .add_role_edge(c, s, "consumed-substance", PREEXISTS)
        .add_has_edge(c, u, PREEXISTS)
    )

    # === Age Feature ===
    person_age_query = inspect.cleandoc(
        f"""match 
            $p isa person, has example-id {example_id}, has age $a; 
            get;"""
    )

    vars = p, a = "p", "a"
    person_age_query_graph = (
        QueryGraph().add_vars(vars, PREEXISTS).add_has_edge(p, a, PREEXISTS)
    )

    # === Risk Factors Feature ===
    risk_factor_query = inspect.cleandoc(
        f"""match 
            $d isa disease; 
            $p isa person, has example-id {example_id}; 
            $r(person-at-risk: $p, risked-disease: $d) isa risk-factor; 
            get;"""
    )

    vars = p, d, r = "p", "d", "r"
    risk_factor_query_graph = (
        QueryGraph()
        .add_vars(vars, PREEXISTS)
        .add_role_edge(r, p, "person-at-risk", PREEXISTS)
        .add_role_edge(r, d, "risked-disease", PREEXISTS)
    )

    # === Symptom ===
    vars = p, s, sn, d, dn, sp, sev, c = "p", "s", "sn", "d", "dn", "sp", "sev", "c"

    symptom_query = inspect.cleandoc(
        f"""match
           $p isa person, has example-id {example_id};
           $s isa symptom, has name $sn;
           $d isa disease, has name $dn;
           $sp(presented-symptom: $s, symptomatic-patient: $p) isa symptom-presentation, has severity $sev;
           $c(cause: $d, effect: $s) isa causality;
           get;"""
    )

    symptom_query_graph = (
        QueryGraph()
        .add_vars(vars, PREEXISTS)
        .add_has_edge(s, sn, PREEXISTS)
        .add_has_edge(d, dn, PREEXISTS)
        .add_role_edge(sp, s, "presented-symptom", PREEXISTS)
        .add_has_edge(sp, sev, PREEXISTS)
        .add_role_edge(sp, p, "symptomatic-patient", PREEXISTS)
        .add_role_edge(c, s, "effect", PREEXISTS)
        .add_role_edge(c, d, "cause", PREEXISTS)
    )

    # === Diagnosis ===

    diag, d, p, dn = "diag", "d", "p", "dn"

    diagnosis_query = inspect.cleandoc(
        f"""match
           $p isa person, has example-id {example_id};
           $d isa disease, has name $dn;
           $diag(patient: $p, diagnosed-disease: $d) isa diagnosis;
           get;"""
    )

    diagnosis_query_graph = (
        QueryGraph()
        .add_vars([diag], TO_INFER)
        .add_vars([d, p, dn], PREEXISTS)
        .add_role_edge(diag, d, "diagnosed-disease", TO_INFER)
        .add_role_edge(diag, p, "patient", TO_INFER)
    )

    # === Candidate Diagnosis ===
    candidate_diagnosis_query = inspect.cleandoc(
        f"""match
           $p isa person, has example-id {example_id};
           $d isa disease, has name $dn;
           $diag(candidate-patient: $p, candidate-diagnosed-disease: $d) isa candidate-diagnosis; 
           get;"""
    )

    candidate_diagnosis_query_graph = (
        QueryGraph()
        .add_vars([diag], CANDIDATE)
        .add_vars([d, p, dn], PREEXISTS)
        .add_role_edge(diag, d, "candidate-diagnosed-disease", CANDIDATE)
        .add_role_edge(diag, p, "candidate-patient", CANDIDATE)
    )

    return [
        (symptom_query, lambda x: x, symptom_query_graph),
        (diagnosis_query, lambda x: x, diagnosis_query_graph),
        (candidate_diagnosis_query, lambda x: x, candidate_diagnosis_query_graph),
        (risk_factor_query, lambda x: x, risk_factor_query_graph),
        (person_age_query, lambda x: x, person_age_query_graph),
        (consumption_query, lambda x: x, consumption_query_graph),
        (hereditary_query, lambda x: x, hereditary_query_graph),
    ]