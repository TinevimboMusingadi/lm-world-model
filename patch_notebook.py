import json, re

nb_path = r'c:\Users\Tinevimbo\lm-world-model\notebooks\01_phase1_pipeline.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

patched = False
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'STAGE 2B' not in src or 'SFTTrainer' not in src:
        continue

    lines = cell['source']
    new_source = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Replace old one-liner dataset build with pre-truncation version ──
        if '# Build dataset' in line and 'train_recs' not in line:
            new_source += [
                '# Build dataset — filter to train split, then format\n',
                '# Pre-truncate each text so every seq fits in MAX_SEQ_LEN tokens.\n',
                '# This prevents the Unsloth padding_free + max_seq_length ValueError.\n',
                'train_recs = [r for r in records if r["split"] == "train"]\n',
                '\n',
                'def truncate_to_tokens(text, max_tokens=MAX_SEQ_LEN):\n',
                '    ids = tokenizer(\n',
                '        text, truncation=True, max_length=max_tokens,\n',
                '        add_special_tokens=False)["input_ids"]\n',
                '    return tokenizer.decode(ids, skip_special_tokens=False)\n',
                '\n',
                'trunc_texts = [truncate_to_tokens(format_for_sft(r, CONDITION, EOS)) for r in train_recs]\n',
                'hf_dataset  = Dataset.from_dict({"text": trunc_texts})\n',
            ]
            # Skip the next 2 old lines (they built hf_dataset the old way)
            while i < len(lines) and 'hf_dataset' not in lines[i]:
                i += 1
            i += 1  # skip the hf_dataset line itself
            continue

        # ── Inject packing params right after dataset_text_field ──
        if 'dataset_text_field' in line:
            new_source.append(line)
            new_source.append("        packing=True,            # Fix: Unsloth padding_free requires packing with max_seq_length\n")
            new_source.append("        packing_strategy='bfd',  # Best-Fit Decreasing — also ~20% faster\n")
            i += 1
            continue

        new_source.append(line)
        i += 1

    cell['source'] = new_source
    patched = True
    print(f'Patched Stage 2B cell ({len(new_source)} lines)')
    break

if not patched:
    print('ERROR: Could not find Stage 2B cell with SFTTrainer')
    exit(1)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook saved successfully.')
