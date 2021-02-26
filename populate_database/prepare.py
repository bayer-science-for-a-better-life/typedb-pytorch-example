from kglib.utils.grakn.synthetic.examples.diagnosis.generate import (
    generate_example_graphs
)

from grakn.client import GraknClient
from grakn.client import SessionType
from grakn.client import TransactionType

import subprocess

DATABASENAME = "diagnosis"


def main(number_of_examples=100):

    client = GraknClient.core("localhost:1729")

    if DATABASENAME in client.databases().all():
        subprocess.call(
            ["grakn", "console", "--script",
             "schema_reload.script"]
        )
    else:
        subprocess.call(
            ["grakn", "console", "--script",
             "schema_load.script"]
        )

    with open ("data.gql", "r") as data_file:
        data = data_file.readlines()
        with client.session(DATABASENAME, SessionType.DATA) as session:
            for line in data:
                with session.transaction(TransactionType.WRITE) as txn:
                    txn.query().insert(line)
                    txn.commit()

    client.close()
    generate_example_graphs(number_of_examples)


if __name__ == "__main__":
    main()
