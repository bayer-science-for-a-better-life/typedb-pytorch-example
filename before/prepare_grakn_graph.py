import subprocess
from kglib.utils.grakn.synthetic.examples.diagnosis.generate import (
    generate_example_graphs,
)


def main(number_of_examples=100):
    process = subprocess.Popen(
        ["grakn", "console", "-k", "diagnosis", "-f", "schema.yml"]
    )
    generate_example_graphs(number_of_examples)


if __name__ == "__main__":
    main()
