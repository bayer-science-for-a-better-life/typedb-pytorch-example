from grakn.client import *
import json

q = """match $x isa {}; get $x; limit 5; """


def execute_query(query: str):
    with Grakn.core_client("localhost:1729") as client:
        with client.session("diagnosis", SessionType.DATA) as session:
            with session.transaction(TransactionType.READ) as txn:
                answer_it = txn.query().match(query)
                for answer in answer_it:
                    print(json.dumps(answer, indent=2, default=str))
                print("-----")


print("person: ")
execute_query(q.format("person"))
print("symptom: ")
execute_query(q.format("symptom"))
print("disease: ")
execute_query(q.format("disease"))
print("diagnosis: ")
execute_query(q.format("diagnosis"))
