import csv
import pickle
import numpy as np
import ast
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

CSV_PATH = "data/movie_metadata.csv"
FAISS_INDEX_PATH = "data/overview_faiss_full.index"
FAISS_META_PATH = "data/overview_metadata_full.pkl"

# Load Sentence-BERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Safely parse stringified lists from CSV
def safe_parse_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

# Containers
texts = []
metadata = []

# Count total rows
with open(CSV_PATH, encoding="utf-8") as f:
    total_rows = sum(1 for _ in f) - 1

# Parse CSV and construct semantic text per movie
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader, total=total_rows, desc="Building FAISS input"):

        if not row["overview"].strip():
            continue

        try:
            title = row["title"]
            overview = row["overview"].strip()
            genres = safe_parse_list(row.get("genres", "[]"))
            keywords = safe_parse_list(row.get("keywords", "[]"))
            emotions = safe_parse_list(row.get("emotions", "[]"))
            directors = safe_parse_list(row.get("directors", "[]"))
            actors = safe_parse_list(row.get("actors", "[]"))
            poster_path = row.get("poster_path", "")

            # Create a combined text representation for embeddings
            semantic_summary = (
                f"Title: {title}. "
                f"Overview: {overview} "
                f"Genres: {' '.join(genres)}. "
                f"Keywords: {' '.join(keywords)}. "
                f"Emotions: {' '.join(emotions)}. "
                f"Directed by {' '.join(directors)}. "
                f"Starring {' '.join(actors)}."
            )

            texts.append(semantic_summary)

            # Save full metadata
            metadata.append({
                "id": int(row["id"]),
                "title": title,
                "genres": genres,
                "keywords": keywords,
                "emotions": emotions,
                "directors": directors,
                "actors": actors,
                "poster_path": poster_path
            })

        except Exception as e:
            print(f"Skipping movie due to error: {e} → {row.get('title', '')}")

# Encode with Sentence-BERT
print("🔍 Encoding movie metadata...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine if normalized
index.add(np.array(embeddings).astype("float32"))

# Save
faiss.write_index(index, FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAISS index and metadata saved.")