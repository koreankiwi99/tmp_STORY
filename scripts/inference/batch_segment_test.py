import json
import argparse
from tqdm import tqdm
from pathlib import Path
from segment_query import MovieQuerySegmenter

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=str, required=True, help="Path to test queries JSON")
    parser.add_argument("--tag_file", type=str, default="data/tag_lists.json", help="Tag DB JSON file")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--embed_threshold", type=float, default=0.35)
    parser.add_argument("--fuzz_threshold", type=int, default=80)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    tag_db = load_json(args.tag_file)
    queries = load_json(args.query_file)

    segmenter = MovieQuerySegmenter(
        tag_db,
        model_path=args.model,
        embed_threshold=args.embed_threshold,
        fuzz_threshold=args.fuzz_threshold,
        top_k=args.top_k
    )

    results = []
    for entry in tqdm(queries, desc="Segmenting queries", dynamic_ncols=True):
        query = entry["query"]
        tqdm.write(f"🔍 {query}")
        segments = segmenter.segment(query)
        results.append({
            "query": query,
            "segments": segments
        })

    input_name = Path(args.query_file).name
    output_path = Path("data/test_queries") / input_name
    save_json(results, output_path)

    print(f"\n✅ Segmentation results saved to: {output_path}")

if __name__ == "__main__":
    main()