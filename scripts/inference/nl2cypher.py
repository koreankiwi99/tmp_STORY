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

        self.embeddings = {
            cat: self.model.encode(tags, normalize_embeddings=True)
            for cat, tags in tag_db.items()
        }

    def fuzzy_match(self, query, category):
        """Fuzzy match using rapidfuzz for categories like genres."""
        return [match for match, score, _ in process.extract(
            query, self.tag_db[category], scorer=fuzz.token_set_ratio, limit=self.top_k
        ) if score >= self.fuzz_threshold]

    def embed_match(self, query, category):
        """Semantic match using embedding similarity."""
        query_vec = self.model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings[category], query_vec)
        return [
            self.tag_db[category][i]
            for i in scores.argsort()[-self.top_k:][::-1]
            if scores[i] >= self.embed_threshold
        ]
    
    def literal_match(self, query, category):
        """Match whole tag strings (not substrings) using word boundaries."""
        query_lower = query.lower()
        results = []
        for tag in self.tag_db[category]:
            tag_words = tag.lower().split()
            if any(word in query_lower.split() for word in tag_words):
                results.append(tag)
            elif tag.lower() in query_lower:
                results.append(tag)
        return results

    def segment(self, user_query):
        return {
            "titles": self.literal_match(user_query, "titles"),
            "genres": self.fuzzy_match(user_query, "genres") or self.embed_match(user_query, "genres"),
            "actors": self.literal_match(user_query, "actors"),
            "directors": self.literal_match(user_query, "directors"),
            "keywords": self.literal_match(user_query, "keywords") or self.embed_match(user_query, "keywords"),
            "emotions": self.literal_match(user_query, "emotions") or self.embed_match(user_query, "emotions")
        }


def build_cypher_query(segments, top_k=10):
    match_clauses = []
    where_clauses = []

    if segments["titles"]:
        title_str = ' OR '.join([f'm.title = "{t}"' for t in segments["titles"]])
        where_clauses.append(f"({title_str})")

    if segments["genres"]:
        match_clauses.append("MATCH (m)-[:HAS_GENRE]->(g:Genre)")
        genre_str = ' OR '.join([f'g.name = "{g}"' for g in segments["genres"]])
        where_clauses.append(f"({genre_str})")

    if segments["actors"]:
        match_clauses.append("MATCH (m)<-[:ACTED_IN]-(a:Actor)")
        actor_str = ' OR '.join([f'a.name = "{a}"' for a in segments["actors"]])
        where_clauses.append(f"({actor_str})")

    if segments["directors"]:
        match_clauses.append("MATCH (m)<-[:DIRECTED]-(d:Director)")
        director_str = ' OR '.join([f'd.name = "{d}"' for d in segments["directors"]])
        where_clauses.append(f"({director_str})")

    if segments["keywords"]:
        match_clauses.append("MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)")
        keyword_str = ' OR '.join([f'k.name = "{k}"' for k in segments["keywords"]])
        where_clauses.append(f"({keyword_str})")

    if segments["emotions"]:
        match_clauses.append("MATCH (m)-[:HAS_EMOTION]->(e:Emotion)")
        emotion_str = ' OR '.join([f'e.name = "{e}"' for e in segments["emotions"]])
        where_clauses.append(f"({emotion_str})")

    match_section = '\n'.join(match_clauses)
    where_section = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    final_query = f"{match_section}\n{where_section}\nRETURN DISTINCT m LIMIT {top_k}"
    return final_query.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--tag_file", type=str, default="data/tag_lists.json")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--embed_threshold", type=float, default=0.35)
    parser.add_argument("--fuzz_threshold", type=int, default=80)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    with open(args.tag_file, "r", encoding="utf-8") as f:
        tag_db = json.load(f)

    segmenter = MovieQuerySegmenter(
        tag_db,
        model_path=args.model,
        embed_threshold=args.embed_threshold,
        fuzz_threshold=args.fuzz_threshold,
        top_k=args.top_k
    )

    segments = segmenter.segment(args.query)
    print("Extracted Segments:")
    print(json.dumps(segments, indent=2, ensure_ascii=False))

    query = build_cypher_query(segments, top_k=args.top_k)
    print("\nGenerated Cypher Query:")
    print(query)


if __name__ == "__main__":
    main()