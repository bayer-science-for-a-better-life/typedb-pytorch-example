from kglib.kgcn_data_loader.dataset.grakn_networkx_dataset import GraknNetworkxDataSet
from about_this_graph import get_query_handles

from graph_dash_app.app import create_app

dataset = GraknNetworkxDataSet(
    example_indices=list(range(100)),
    get_query_handles_for_id=get_query_handles,
    database="diagnosis",
    uri="localhost:1729",
    infer=True,
    transform=None,
)


if __name__ == "__main__":
    create_app(dataset).run_server(debug=True)
