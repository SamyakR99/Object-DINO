import json

my_outputs = "/scratch/bcyh/samyakr99/middle_layers_indicating_hallucinations/pope_results_qwen2vl_adversarial.json"    # your current file (list of dicts)
out_file = "pope_results_adversarial_for_eval_greedy_qwen2vl.jsonl"  # new file in correct format
mode = "greedy"  # or "sampling"


def normalize_answer(text: str) -> str:
    """Convert free-form text to 'yes' or 'no'."""
    # Only keep the first sentence
    if '.' in text:
        text = text.split('.')[0]

    # Clean up punctuation
    text = text.replace(',', '').strip()

    words = text.split()
    # If 'no' or 'not' appears, treat as 'no'
    if any(w.lower() in ['no', 'not'] for w in words):
        return 'no'
    else:
        return 'yes'

# load your results (assumed to be a list of dicts)
with open(my_outputs, "r") as f:
    my_data = json.load(f)

# reformat
with open(out_file, "w") as f:
    for entry in my_data:
        ans_text = entry[mode]   # pick greedy/sampling
        ans_norm = normalize_answer(ans_text)
        f.write(json.dumps({
            "question": entry["question"],
            "answer": ans_norm
        }) + "\n")

print(f"Saved reformatted file to: {out_file}")
