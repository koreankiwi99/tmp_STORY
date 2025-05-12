import re
import numpy as np
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer


class MovieQuerySegmenter:
    def __init__(self, tag_db, model_path="all-MiniLM-L6-v2", embed_threshold=0.35, top_k=3, fuzz_threshold=80):
        self.tag_db = tag_db
        self.embed_threshold = embed_threshold
        self.top_k = top_k
        self.fuzz_threshold = fuzz_threshold
        self.model = SentenceTransformer(model_path)

        # Precompute embeddings
        self.embeddings = {
            cat: self.model.encode(tags, normalize_embeddings=True)
            for cat, tags in tag_db.items()
        }

    def literal_match(self, query, category, min_word_len=3, min_word_count=2):
        """Strict literal matching using word boundaries and minimum length."""
        query_lower = query.lower()
        results = []
        for tag in self.tag_db[category]:
            tag_lower = tag.lower()
            if len(tag_lower) < min_word_len or len(tag_lower.split()) < min_word_count:
                continue  # Skip short or generic names
            if re.search(rf"\b{re.escape(tag_lower)}\b", query_lower):
                results.append(tag)
        return results

    def fuzzy_match(self, query, category):
        """RapidFuzz fuzzy matching."""
        return [match for match, score, _ in process.extract(
            query, self.tag_db[category], scorer=fuzz.token_set_ratio, limit=self.top_k
        ) if score >= self.fuzz_threshold]

    def embed_match(self, query, category):
        """Embedding similarity matching."""
        query_vec = self.model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings[category], query_vec)
        return [
            self.tag_db[category][i]
            for i in scores.argsort()[-self.top_k:][::-1]
            if scores[i] >= self.embed_threshold
        ]

    def segment(self, user_query):
        return {
            "titles": self.literal_match(user_query, "titles", min_word_len=4, min_word_count=2),
            "genres": self.fuzzy_match(user_query, "genres") or self.embed_match(user_query, "genres"),
            "actors": self.literal_match(user_query, "actors", min_word_len=4, min_word_count=2) or self.fuzzy_match(user_query, "actors") or self.embed_match(user_query, "actors"),
            "directors": self.literal_match(user_query, "directors", min_word_len=4, min_word_count=2) or self.fuzzy_match(user_query, "directors") or self.embed_match(user_query, "directors"),
            "keywords": self.literal_match(user_query, "keywords", min_word_len=5, min_word_count=2) or self.embed_match(user_query, "keywords"),
            "emotions": self.literal_match(user_query, "emotions", min_word_len=4, min_word_count=1) or self.embed_match(user_query, "emotions")
        }