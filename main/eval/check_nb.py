import json
for nb_path in ['training_lab/agent2_reader/Train_Agent2_Reader_T4x2.ipynb', 'training_lab/agent5_writer/Train_Agent5_Writer_T4x2.ipynb']:
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    print(nb_path, '--> num cells:', len(nb['cells']))
    for idx, c in enumerate(nb['cells']):
        print(f"Cell {idx}: type={c.get('cell_type')}, chars={len(''.join(c.get('source', [])))}")
        print("--- CONTENT SNIPPET ---")
        print(''.join(c.get('source', []))[:300])
        print("-----------------------")
