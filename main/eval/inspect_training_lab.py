import os
import json

def inspect_training_lab():
    nb_path = "training_lab/Train_Master_T4x2.ipynb"
    if os.path.exists(nb_path):
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        print(f"=== {nb_path} ({len(cells)} cells) ===")
        for i, c in enumerate(cells[:20]):
            txt = "".join(c.get("source", []))[:100].replace("\n", " ")
            print(f"  Cell {i} [{c.get('cell_type')}]: {txt}")

    # Inspect other notebooks
    for item in os.listdir("training_lab"):
        item_path = os.path.join("training_lab", item)
        if os.path.isdir(item_path):
            for f in os.listdir(item_path):
                if f.endswith(".ipynb"):
                    f_path = os.path.join(item_path, f)
                    with open(f_path, "r", encoding="utf-8") as nbf:
                        c_data = json.load(nbf)
                    print(f"=== {f_path} ({len(c_data.get('cells', []))} cells) ===")

if __name__ == "__main__":
    inspect_training_lab()
