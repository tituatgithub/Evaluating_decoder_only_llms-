import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path

# Get the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
print(ROOT_DIR)

# ----------------------------
# Configurations
# ----------------------------
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH = ROOT_DIR / "data/mt_comilingua_test.json"
OUTPUT_DIR = ROOT_DIR / "out/llama_output_hi"  # Hindi directory

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = OUTPUT_DIR / "progress_hi.json"
CHECKPOINT_PATH = OUTPUT_DIR / "llama_checkpoint_hi.json"
LIVE_STATUS_PATH = OUTPUT_DIR / "live_status_hi.json"
FINAL_OUTPUT_PATH = OUTPUT_DIR / "llama_translations_hi.json"

# ----------------------------
# Setup
# ----------------------------
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

# ----------------------------
# Load model and tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

translator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ----------------------------
# Prompt template
# ----------------------------
def make_prompt(text, src_lang="Hinglish", tgt_lang="Hindi"):
    return f"""You are a helpful translation assistant.
Translate the following text from {src_lang} to {tgt_lang}.

Text: {text}

Translation:"""

# ----------------------------
# Main translation loop
# ----------------------------
def main():
    # Load dataset
    df = pd.read_json(DATA_PATH, lines=True)

    translations = []

    # Clean old logs if rerunning
    for path in [PROGRESS_PATH, CHECKPOINT_PATH, LIVE_STATUS_PATH]:
        if path.exists():
            path.unlink()  # Remove old logs

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Translating"):
        prompt = make_prompt(row["Sentences"], src_lang="Code-mixed Hindi-English", tgt_lang="Hindi")
        out = translator(prompt, max_new_tokens=128, temperature=0.2, do_sample=False)
        generated = out[0]["generated_text"].split("Translation:")[-1].strip()

        # Save in-memory
        translations.append(generated)

        # Append to progress_hi.json (streaming log)
        progress_record = {
            "Id": int(row["Id"]),
            "Input": row["Sentences"],
            "llama_translation_hi": generated
        }
        with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(progress_record, ensure_ascii=False) + "\n")

        # Update live_status_hi.json
        live_status = {
            "total_sentences": len(df),
            "completed": idx + 1
        }
        with open(LIVE_STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(live_status, f, ensure_ascii=False, indent=2)

        # Every 50 rows → save checkpoint_hi.json
        if (idx + 1) % 50 == 0:
            temp_df = df.iloc[: idx + 1].copy()
            temp_df["llama_translation_hi"] = translations
            temp_df.to_json(CHECKPOINT_PATH, orient="records", lines=True, force_ascii=False)

    # Final save
    df["llama_translation_hi"] = translations
    df.to_json(FINAL_OUTPUT_PATH, orient="records", lines=True, force_ascii=False)

    print("✅ Finished.")
    print(f"Live log → {PROGRESS_PATH}")
    print(f"Checkpoints → {CHECKPOINT_PATH}")
    print(f"Final output → {FINAL_OUTPUT_PATH}")
    print(f"Live status → {LIVE_STATUS_PATH}")


if __name__ == "__main__":
    main()
