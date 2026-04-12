import json
with open('notebooks/01_phase1_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'STAGE 2B' in src and 'SFTTrainer' in src:
        for line in cell['source']:
            if any(kw in line for kw in ['packing', 'truncate_to_tokens', 'dataset_text_field', 'hf_dataset']):
                print(line.rstrip())
