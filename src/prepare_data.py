from datasets import load_dataset
import json, re, os
from huggingface_hub import login

# ── 0. Login to Hugging Face ─────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")  
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️  No HF_TOKEN found, continuing without login...")

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading dataset...")
train_raw = load_dataset("stanfordnlp/imdb", split="train[:4000]")
test_raw  = load_dataset("stanfordnlp/imdb", split="test[:1000]")

print(f"Train size: {len(train_raw)} | Test size: {len(test_raw)}")

# ── 2. Cleaning function ──────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── 3. Apply cleaning ─────────────────────────────────────────────────────────
print("\nCleaning text...")
train_clean = train_raw.map(lambda x: {"text": clean_text(x["text"]), "label": x["label"]})
test_clean  = test_raw.map( lambda x: {"text": clean_text(x["text"]), "label": x["label"]})

print("\n--- BEFORE cleaning ---")
print(train_raw[0]["text"][:200])
print("\n--- AFTER cleaning ---")
print(train_clean[0]["text"][:200])

# ── 4. Save id2label mapping ──────────────────────────────────────────────────
id2label = {"0": "NEGATIVE", "1": "POSITIVE"}
os.makedirs("data", exist_ok=True)
with open("data/id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

print("\n✅ Saved data/id2label.json")
print("✅ Data preparation complete!")