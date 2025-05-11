import csv
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# File paths
CSV_PATH = "data/moviemetadata.csv"
FAISS_INDEX_PATH = "data/overview_faiss.index"
FAISS_META_PATH = "data/overview_metadata.pkl"

# Load sentence embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Read movie overviews and metadata
overviews = []
metadata = []

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            if not row["overview"].strip():
                continue  # Skip movies with empty overview

            metadata.append({
                "id": int(row["id"]),
                "title": row["title"],
                "overview": row["overview"],
                "year": int(row["year"]) if row["year"] else None,
                "genres": eval(row["genres"]) if row["genres"] else [],
                "emotions": eval(row["emotions"]) if row["emotions"] else [],
                "actors": eval(row["actors"]) if row["actors"] else [],
                "keywords": row["keywords"].split(", ") if row["keywords"] else [],
                "poster_path": row.get("poster_path", ""),
                "director": row.get("director", "")
            })

            overviews.append(row["overview"])
        except Exception as e:
            print(f"⚠️ Skipping movie due to error: {e}\n→ {row.get('title', '')}")

# Encode overviews and build FAISS index
print("🧠 Encoding overviews...")
embeddings = model.encode(overviews, show_progress_bar=True)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# Save index and metadata
faiss.write_index(index, FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"✅ FAISS index saved to: {FAISS_INDEX_PATH}")
print(f"✅ Metadata saved to: {FAISS_META_PATH}")