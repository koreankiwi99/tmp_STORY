import csv
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import ast

CSV_PATH = "data/movie_metadata.csv"
FAISS_INDEX_PATH = "data/overview_faiss_combined.index"
FAISS_META_PATH = "data/overview_metadata_combined.pkl"

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Helper to safely parse stringified Python lists
def safe_parse_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

combined_texts = []
metadata = []

# Count total rows for tqdm
with open(CSV_PATH, encoding="utf-8") as f:
    total_rows = sum(1 for _ in f) - 1  # subtract header

# Process rows
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader, total=total_rows, desc="Processing movies"):
        try:
            if not row["overview"].strip():
                continue

            keywords = safe_parse_list(row["keywords"])
            emotions = safe_parse_list(row["emotions"])

            combined_text = (
                f"{row['overview']} "
                f"Keywords: {' '.join(keywords)}. "
                f"Emotions: {' '.join(emotions)}."
            )

            combined_texts.append(combined_text)

            # Only store ID and title
            metadata.append({
                "id": int(row["id"]),
                "title": row["title"]
            })

        except Exception as e:
            print(f"Skipping row due to error: {e} → {row.get('title', '')}")

# Encode and build FAISS index
print("Encoding combined content...")
embeddings = model.encode(combined_texts, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))

# Save index and metadata
faiss.write_index(index, FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAISS index and metadata saved.")