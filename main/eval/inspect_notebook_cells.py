import json

for p in [
    "training_lab/agent2_reader/Train_Agent2_Reader_T4x2.ipynb",
    "training_lab/agent5_writer/Train_Agent5_Writer_T4x2.ipynb",
    "training_lab/agent4_fact_checker/Train_Agent4_FactChecker_T4x2.ipynb"
]:
    print(f"\n==================== {p} ====================")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for i, c in enumerate(nb.get("cells", [])):
        print(f"--- Cell {i} ({c.get('cell_type')}) ---")
        print("".join(c.get("source", []))[:500])
