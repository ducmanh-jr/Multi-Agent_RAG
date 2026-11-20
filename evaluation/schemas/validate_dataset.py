import json
import jsonschema
from pathlib import Path

def validate():
    schema_path = Path("evaluation/schemas/test_case.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    dataset_dir = Path("evaluation/dataset")
    dataset_files = [
        "general_qa.json",
        "document_qa.json",
        "web_research.json",
        "mixed_qa.json",
        "edge_cases.json"
    ]

    total_cases = 0
    for fname in dataset_files:
        fpath = dataset_dir / fname
        cases = json.loads(fpath.read_text(encoding="utf-8"))
        for case in cases:
            total_cases += 1
            jsonschema.validate(instance=case, schema=schema)
            print(f"Validated {case['id']}: {case['query'][:50]}...")

    print(f"\nSUCCESS: All {total_cases} test cases strictly conform to test_case.schema.json!")

if __name__ == "__main__":
    validate()
