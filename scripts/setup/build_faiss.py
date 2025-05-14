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
            actors = safe_parse_list(row.get("actors", "[]"))
            director = row.get("director", "").strip()
            poster_path = row.get("poster_path", "").strip()


            # Build semantic summary string
            semantic_summary = f"Title: {title}. Overview: {overview} "

            if actors:
                actor_list = ", ".join(actors[:5])  # limit to top 5
                semantic_summary += f"Main cast includes {actor_list}. "

            if director:
                semantic_summary += f"Directed by {director}. "

            if genres:
                genre_list = ", ".join(genres)
                semantic_summary += f"Genres: {genre_list}. "

            if keywords:
                keyword_list = ", ".join(keywords)
                semantic_summary += f"Tags include: {keyword_list}. "

            if emotions:
                emotion_list = ", ".join(emotions)
                semantic_summary += f"Emotions evoked: {emotion_list}."

            texts.append(semantic_summary)

            # Save full metadata
            metadata.append({
                "id": int(row["id"]),
                "title": title,
                "genres": genres,
                "keywords": keywords,
                "emotions": emotions,
                "directors": director,
                "actors": actors,
                "overview" : overview,
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