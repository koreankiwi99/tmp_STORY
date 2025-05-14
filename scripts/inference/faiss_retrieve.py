import argparse
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# === Config ===
FAISS_INDEX_PATH = "data/overview_faiss_full.index"
FAISS_META_PATH = "data/overview_metadata_full.pkl"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5
RERANK_K = 20

# === Load index, metadata, models ===
print("🔧 Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "rb") as f:
    metadata = pickle.load(f)

print("🔧 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("🔧 Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# === Core Retrieval ===
def retrieve(query: str, top_k: int = TOP_K):
    formatted_query = f"<|user|>\n{query}"
    embedding = model.encode([formatted_query], normalize_embeddings=True)
    scores, indices = index.search(np.array(embedding).astype("float32"), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx]
        results.append({
            "id": item["id"],
            "title": item["title"],
            "overview": item.get("overview", ""),
            "genres": item.get("genres", []),
            "keywords": item.get("keywords", []),
            "emotions": item.get("emotions", []),
            "directors": item.get("directors", ""),
            "actors": item.get("actors", []),
            "poster_path": item.get("poster_path", ""),
            "score": float(score)
        })
    return results

def rerank(query, candidates, top_k=TOP_K):
    pairs = [[query, f"{c['title']} {c['overview']}"] for c in candidates]
    scores = reranker.predict(pairs)
    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
    sorted_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_results[:top_k]

def print_results(results, query=None):
    if query:
        print(f"\n🔎 Query: {query}")
    for r in results:
        print(f"\n🎬 {r['title']} (FAISS: {r['score']:.4f}, Rerank: {r.get('rerank_score', 0):.4f})")
        print(f"   ID: {r['id']}")
        print(f"   Genres: {', '.join(r['genres'])}")
        print(f"   Keywords: {', '.join(r['keywords'])}")
        print(f"   Emotions: {', '.join(r['emotions'])}")
        print(f"   Director: {r['directors']}")
        print(f"   Actors: {', '.join(r['actors'][:10])}")
        print(f"   Poster Path: {r['poster_path']}")

# === CLI Entry Point ===
def main():
    parser = argparse.ArgumentParser(description="Retrieve similar movies using FAISS + reranker.")
    parser.add_argument("--query", type=str, help="A user query.")
    parser.add_argument("--json_file", type=str, help="JSON file containing a list of queries.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Final top-k results to return.")
    parser.add_argument("--rerank_k", type=int, default=RERANK_K, help="FAISS candidate size before rerank.")
    parser.add_argument("--no_rerank", action="store_true", help="Skip reranking step.")
    args = parser.parse_args()

    if args.query:
        results = retrieve(args.query, top_k=args.rerank_k if not args.no_rerank else args.top_k)
        if not args.no_rerank:
            results = rerank(args.query, results, top_k=args.top_k)
        print_results(results, query=args.query)

    elif args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        for item in queries:
            q = item["query"]
            results = retrieve(q, top_k=args.rerank_k if not args.no_rerank else args.top_k)
            if not args.no_rerank:
                results = rerank(q, results, top_k=args.top_k)
            print_results(results, query=q)
    else:
        print("❗ Please provide either --query or --json_file.")

if __name__ == "__main__":
    main()