import argparse
import json
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
import numpy as np


class MovieQuerySegmenter:
    def __init__(self, tag_db, model_path="all-MiniLM-L6-v2", embed_threshold=0.35, top_k=3, fuzz_threshold=80):
        self.tag_db = tag_db
        self.embed_threshold = embed_threshold
        self.top_k = top_k
        self.fuzz_threshold = fuzz_threshold
        self.model = SentenceTransformer(model_path)

        # Precompute embeddings for all categories
        self.tag_values = {}
        self.embeddings = {}

        for category, tags in tag_db.items():
            self.tag_values[category] = tags
            self.embeddings[category] = self.model.encode(tags, normalize_embeddings=True)

    def embed_match(self, query, category):
        """Fallback embedding match for fuzzy or corrected results."""
        tag_list = self.tag_values[category]
        tag_embeds = self.embeddings[category]
        query_vec = self.model.encode(query, normalize_embeddings=True)

        scores = np.dot(tag_embeds, query_vec)
        top_indices = scores.argsort()[-self.top_k:][::-1]
        return [tag_list[i] for i in top_indices if scores[i] >= self.embed_threshold]

    def fuzzy_match(self, query, category):
        """Fuzzy match for genres, checking for partial or misspelled genre names."""
        results = process.extract(query, self.tag_values[category], scorer=fuzz.token_set_ratio, limit=5)
        return [match for match, score, _ in results if score >= self.fuzz_threshold]

    def literal_match(self, query, category):
        """Exact match for movie titles, genres, directors, etc."""
        results = []
        query_lower = query.lower()

        for tag in self.tag_values[category]:
            tag_clean = tag.replace("Movie: ", "").lower()

            if category in ["actors", "directors"]:
                name_parts = tag_clean.split(" ")
                if len(name_parts) > 1:  # Handle full name
                    last_name = name_parts[-1]
                    if last_name.lower() in query_lower:
                        results.append(tag)
                if tag_clean in query_lower:  # Handle full name match
                    results.append(tag)
            else:
                if tag_clean in query_lower:  # Regular matching
                    results.append(tag)

        return results

    def segment(self, user_query):
        # First, try exact matching
        return {
            "mentioned_movie": self.literal_match(user_query, "titles"),  # exact match
            "actors": self.literal_match(user_query, "actors"),
            "genres": self.fuzzy_match(user_query, "genres") or self.embed_match(user_query, "genres"),  # fuzzy + embedding
            "directors": self.literal_match(user_query, "directors"),
            "keywords": self.literal_match(user_query, "keywords"),
            "emotions": self.literal_match(user_query, "emotions"),  # exact match for emotions
            # Fallback to embedding match if no exact match was found
            "similar_keywords": self.embed_match(user_query, "keywords") if not self.literal_match(user_query, "keywords") else [],
            "similar_emotions": self.embed_match(user_query, "emotions") if not self.literal_match(user_query, "emotions") else []
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a movie query into structured fields.")
    parser.add_argument("--query", type=str, required=True, help="User query to segment.")
    parser.add_argument("--tag_file", type=str, default="data/tag_lists.json", help="Path to tag list JSON file.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Embedding similarity threshold.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches per category.")
    parser.add_argument("--fuzz_threshold", type=int, default=80, help="Fuzzy matching threshold for genres.")
    args = parser.parse_args()

    with open(args.tag_file, "r", encoding="utf-8") as f:
        tag_db = json.load(f)

    segmenter = MovieQuerySegmenter(tag_db, embed_threshold=args.threshold, top_k=args.top_k, fuzz_threshold=args.fuzz_threshold)
    result = segmenter.segment(args.query)

    print(json.dumps(result, indent=2))