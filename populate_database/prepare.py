from kglib.utils.grakn.synthetic.examples.diagnosis.generate import (
    generate_example_graphs
)

from grakn.client import GraknClient
from grakn.client import SessionType
from grakn.client import TransactionType

def main(number_of_examples=100):

    client = GraknClient.core("localhost:1729")
    with open ("schema.gql", "r") as schema_file:
        schema = schema_file.readlines()
        schema_as_str = " ".join(schema)
        with client.session("diagnosis", SessionType.SCHEMA) as session:
            with session.transaction(TransactionType.WRITE) as txn:
                txn.query().define(schema_as_str)
                txn.commit()

    with open ("data.gql", "r") as data_file:
        data = data_file.readlines()
        data_as_str = " ".join(data)
        with client.session("diagnosis", SessionType.DATA) as session:
            with session.transaction(TransactionType.WRITE) as txn:
                txn.query().insert(data_as_str)
                txn.commit()

    client.close()
    generate_example_graphs(number_of_examples)


if __name__ == "__main__":
    main()
