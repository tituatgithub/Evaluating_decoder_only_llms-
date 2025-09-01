import os
import json
import pandas as pd
import sacrebleu
from bert_score import score

# ----------------------------
# Configurations
# ----------------------------
DATA_PATH = "/home/ethicaldevice/Desktop/Titu_exp/mt_comilingua_test.json"
OUTPUT_PATH = "/home/ethicaldevice/Desktop/Titu_exp/llama_output_en/llama_translations.json"
RESULTS_PATH = "/home/ethicaldevice/Desktop/Titu_exp/llama_output_en/eval_results_en.json"

# ----------------------------
# Load dataset and outputs
# ----------------------------
df_gold = pd.read_json(DATA_PATH, lines=True)
df_pred = pd.read_json(OUTPUT_PATH, lines=True)

# Merge on Id to ensure alignment
df = pd.merge(df_gold, df_pred[["Id", "llama_translation"]], on="Id", how="inner")

print(f"Loaded {len(df)} examples for evaluation.")

# ----------------------------
# Collect predictions & references
# ----------------------------
system_outputs = []
references_list = []

for _, row in df.iterrows():
    pred = row["llama_translation"]

    # Collect multiple human references if available
    refs = []
    for i in range(1, 4):
        key = f"Annotator_{i}_en_translation"
        if key in row and isinstance(row[key], str):
            refs.append(row[key])

    if refs:
        system_outputs.append(pred)
        references_list.append(refs)

# ----------------------------
# Compute BLEU
# ----------------------------
bleu = sacrebleu.corpus_bleu(system_outputs, list(zip(*references_list)))
bleu_score = bleu.score

# ----------------------------
# Compute BERTScore
# ----------------------------
P, R, F1 = score(system_outputs, [refs[0] for refs in references_list], lang="en", verbose=True)
bert_score = {
    "precision": float(P.mean()),
    "recall": float(R.mean()),
    "f1": float(F1.mean())
}

# ----------------------------
# Save results
# ----------------------------
results = {
    "BLEU": bleu_score,
    "BERTScore": bert_score
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Evaluation finished.")
print(f"Results saved to {RESULTS_PATH}")
print(json.dumps(results, indent=2, ensure_ascii=False))
