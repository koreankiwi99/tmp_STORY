import argparse
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# === Config ===
FAISS_INDEX_PATH = "data/overview_faiss_full.index"
FAISS_META_PATH = "data/overview_metadata_full.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
RERANK_K = 20

# === Load FAISS index, metadata, and model ===
index = faiss.read_index(FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "rb") as f:
    metadata = pickle.load(f)
model = SentenceTransformer(MODEL_NAME)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# === Retrieval API for external use ===
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


def rerank(query, candidates, top_k=5):
    # Combine as pairs
    pairs = [[query, f"{c['title']} {c.get('overview', '')}"] for c in candidates]
    scores = reranker.predict(pairs)
    
    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
    
    sorted_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_results[:top_k]


# === Pretty printer for retrieval results ===
def print_results(results, query=None):
    if query:
        print(f"\n🔎 Query: {query}")
    for r in results:
        print(f"\n🎬 {r['title']} (Score: {r['score']:.4f})")
        print(f"   ID: {r['id']}")
        print(f"   Genres: {', '.join(r['genres'])}")
        print(f"   Keywords: {', '.join(r['keywords'])}")
        print(f"   Emotions: {', '.join(r['emotions'])}")
        print(f"   Director: {r['directors']}")
        print(f"   Actors: {', '.join(r['actors'])}")
        print(f"   Poster Path: {r['poster_path']}")


# === Command-line interface ===
def main():
    parser = argparse.ArgumentParser(description="Retrieve similar movies using FAISS.")
    parser.add_argument("--query", type=str, help="A single user query.")
    parser.add_argument("--json_file", type=str, help="A JSON file with a list of queries.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of top results to return.")
    parser.add_argument("--rerank_k", type=int, default=RERANK_K, help="Number of top results to rerank.")
    args = parser.parse_args()

    if args.query:
        results = retrieve(args.query, top_k=args.rerank_k)  # Expand FAISS top_k
        results = rerank(args.query, results, top_k=args.top_k)
        print_results(results, query=args.query)

    elif args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        for item in queries:
            q = item["query"]
            results = retrieve(q, top_k=args.rerank_k)  # Expand FAISS top_k
            results = rerank(q, results, top_k=args.top_k)
            print_results(results, query=q)

    else:
        print("❗ Please provide --query or --json_file.")


if __name__ == "__main__":
    main()