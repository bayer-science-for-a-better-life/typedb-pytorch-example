from kglib.utils.grakn.synthetic.examples.diagnosis.generate import (
    generate_example_graphs,
)

from grakn.client import *

import subprocess

DATABASENAME = "diagnosis"


def main(number_of_examples=100):

    client = Grakn.core_client("localhost:1729")

    if any(DATABASENAME in str(db) for db in client.databases().all()):
        subprocess.call(["/Users/henning.kuich@bayer.com/tools/grakn-core-all-mac-2.0.1/grakn", "console", "--script", "schema_reload.script"])
    else:
        subprocess.call(["/Users/henning.kuich@bayer.com/tools/grakn-core-all-mac-2.0.1/grakn", "console", "--script", "schema_load.script"])

    with open("data.gql", "r") as data_file:
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
