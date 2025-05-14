import argparse
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === Config ===
FAISS_INDEX_PATH = "data/overview_faiss_full.index"
FAISS_META_PATH = "data/overview_metadata_full.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# === Load FAISS + Metadata + Model ===
index = faiss.read_index(FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "rb") as f:
    metadata = pickle.load(f)
model = SentenceTransformer(MODEL_NAME)

# === Retrieval ===
def retrieve(query: str, top_k: int = TOP_K):
    embedding = model.encode([query])
    scores, indices = index.search(np.array(embedding).astype("float32"), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx]
        results.append({
            "id": item["id"],
            "title": item["title"],
            "genres": item.get("genres", []),
            "keywords": item.get("keywords", []),
            "emotions": item.get("emotions", []),
            "directors": item.get("directors", []),
            "actors": item.get("actors", []),
            "poster_path": item.get("poster_path", ""),
            "score": float(score)
        })
    return results

# === CLI Argument Parser ===
def main():
    parser = argparse.ArgumentParser(description="Retrieve similar movies using FAISS.")
    parser.add_argument("--query", type=str, help="A single user query.")
    parser.add_argument("--json_file", type=str, help="A JSON file with list of queries.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of top results to return.")
    args = parser.parse_args()

    elif args.query:
        print(f"\n🔎 Query: {args.query}")
        results = retrieve(args.query, top_k=args.top_k)
        for r in results:
            print(f"\n🎬 {r['title']} (Score: {r['score']:.4f})")
            print(f"   ID: {r['id']}")
            print(f"   Genres: {', '.join(r['genres'])}")
            print(f"   Keywords: {', '.join(r['keywords'])}")
            print(f"   Emotions: {', '.join(r['emotions'])}")
            print(f"   Directors: {', '.join(r['directors'])}")
            print(f"   Actors: {', '.join(r['actors'])}")
            print(f"   Poster Path: {r['poster_path']}")

    elif args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        for item in queries:
            print(f"\n🔎 Query: {item['query']}")
            results = retrieve(item['query'], top_k=args.top_k)
            for r in results:
                print(f"\n🎬 {r['title']} (Score: {r['score']:.4f})")
                print(f"   ID: {r['id']}")
                print(f"   Genres: {', '.join(r['genres'])}")
                print(f"   Keywords: {', '.join(r['keywords'])}")
                print(f"   Emotions: {', '.join(r['emotions'])}")
                print(f"   Directors: {', '.join(r['directors'])}")
                print(f"   Actors: {', '.join(r['actors'])}")
                print(f"   Poster Path: {r['poster_path']}")

    else:
        print("❗ Please provide --query, --json_file, or --print_all_metadata.")

if __name__ == "__main__":
    main()