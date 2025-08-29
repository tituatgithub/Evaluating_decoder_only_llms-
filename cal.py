import pandas as pd
import sacrebleu
from bert_score import score

# Load predictions + refs
df = pd.read_json("llama_translations.json", lines=True)

refs = df["Annotator_1_en_translation"].astype(str).tolist()
hyps = df["llama_translation"].astype(str).tolist()

# --- BLEU ---
bleu = sacrebleu.corpus_bleu(hyps, [refs])
print(f"BLEU: {bleu.score:.2f}")

'''# --- BERTScore ---
P, R, F1 = score(hyps, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli")
print(f"BERTScore Precision: {P.mean().item():.4f}")
print(f"BERTScore Recall:    {R.mean().item():.4f}")
print(f"BERTScore F1:        {F1.mean().item():.4f}")'''
# Compute BERTScore in batches to avoid OOM errors
import math
batch_size = 10  # You can adjust this based on your memory
P_scores, R_scores, F1_scores = [], [], []

total = len(hyps)
num_batches = math.ceil(total / batch_size)

for i in range(num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, total)
    batch_hyps = hyps[start:end]
    batch_refs = refs[start:end]
    P, R, F1 = score(batch_hyps, batch_refs, lang="en", model_type="microsoft/deberta-xlarge-mnli")
    P_scores.append(P.mean().item())
    R_scores.append(R.mean().item())
    F1_scores.append(F1.mean().item())
    print(f"Batch {i+1}/{num_batches}: BERTScore F1 = {F1.mean().item():.4f}")

mean_P = sum(P_scores) / len(P_scores)
mean_R = sum(R_scores) / len(R_scores)
mean_F1 = sum(F1_scores) / len(F1_scores)
print(f"Mean BERTScore Precision: {mean_P:.4f}")
print(f"Mean BERTScore Recall:    {mean_R:.4f}")
print(f"Mean BERTScore F1:        {mean_F1:.4f}")
