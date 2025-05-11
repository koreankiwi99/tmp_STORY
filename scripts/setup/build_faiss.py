import csv
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

CSV_PATH = "data/moviemetadata.csv"
FAISS_INDEX_PATH = "data/overview_faiss_combined.index"
FAISS_META_PATH = "data/overview_metadata_combined.pkl"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

combined_texts = []
metadata = []

# Count total rows first for tqdm
with open(CSV_PATH, encoding="utf-8") as f:
    total_rows = sum(1 for _ in f) - 1  # subtract header

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader, total=total_rows, desc="Processing movies"):
        try:
            if not row["overview"].strip():
                continue

            keywords = eval(row["keywords"]) if row["keywords"] else []
            emotions = eval(row["emotions"]) if row["emotions"] else []

            combined_text = (
                f"{row['overview']} "
                f"Keywords: {' '.join(keywords)}. "
                f"Emotions: {' '.join(emotions)}."
            )

            combined_texts.append(combined_text)
            metadata.append({
                "id": int(row["id"]),
                "title": row["title"],
                "overview": row["overview"],
                "year": int(row["year"]) if row["year"] else None,
                "genres": eval(row["genres"]) if row["genres"] else [],
                "emotions": emotions,
                "actors": eval(row["actors"]) if row["actors"] else [],
                "keywords": keywords,
                "poster_path": row.get("poster_path", ""),
                "director": row.get("director", "")
            })
        except Exception as e:
            print(f"Skipping row due to error: {e} → {row.get('title', '')}")

# Embed and build FAISS
print("Encoding combined content...")
embeddings = model.encode(combined_texts, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))

# Save everything
faiss.write_index(index, FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index and metadata saved.")