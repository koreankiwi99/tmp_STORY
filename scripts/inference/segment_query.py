import re
import json
import numpy as np
from pathlib import Path
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer


class MovieQuerySegmenter:
    def __init__(
        self,
        tag_db,
        model_path="all-MiniLM-L6-v2",
        embed_threshold=0.35,
        top_k=3,
        fuzz_threshold=80,
        popular_path="data/tag_lists_popular.json"
    ):
        self.tag_db = tag_db
        self.embed_threshold = embed_threshold
        self.top_k = top_k
        self.fuzz_threshold = fuzz_threshold
        self.model = SentenceTransformer(model_path)

        with open(popular_path, "r", encoding="utf-8") as f:
            pop_tags = json.load(f)
        self.popular_actors = set(pop_tags.get("actors", []))
        self.popular_directors = set(pop_tags.get("directors", []))

        self.embeddings = {
            cat: self.model.encode(tags, normalize_embeddings=True)
            for cat, tags in tag_db.items()
        }

    def literal_match(self, query, category, min_word_len=3, min_word_count=2):
        query_lower = query.lower()
        results = []
        for tag in self.tag_db[category]:
            tag_lower = tag.lower()
            if len(tag_lower) < min_word_len or len(tag_lower.split()) < min_word_count:
                continue
            if re.search(rf"\b{re.escape(tag_lower)}\b", query_lower):
                results.append(tag)
        return results

    def soft_literal_match(self, query, category, popular_set):
        query_lower = query.lower()
        results = []
        for tag in self.tag_db[category]:
            if tag not in popular_set:
                continue
            tag_parts = tag.lower().split()
            if any(part in query_lower for part in tag_parts):
                results.append(tag)
        return results

    def keyword_fuzzy_match(self, query, category, filter_set=None):
        keywords = re.findall(r"[A-Z][a-z]+", query)
        candidates = [t for t in self.tag_db[category] if not filter_set or t in filter_set]
        results = set()
        for word in keywords:
            matches = process.extract(word, candidates, scorer=fuzz.token_set_ratio, limit=self.top_k)
            results.update([match for match, score, _ in matches if score >= self.fuzz_threshold])
        return list(results)

    def segment(self, query):
        return {
            "titles": self.literal_match(query, "titles", min_word_len=4, min_word_count=2)
                      or self.keyword_fuzzy_match(query, "titles"),

            "genres": self.literal_match(query, "genres", min_word_len=4, min_word_count=1)
                      or self.keyword_fuzzy_match(query, "genres"),

            "actors": (
                self.literal_match(query, "actors", min_word_len=4, min_word_count=2)
                or self.soft_literal_match(query, "actors", self.popular_actors)
                or self.keyword_fuzzy_match(query, "actors", self.popular_actors)
            ),

            "directors": (
                self.literal_match(query, "directors", min_word_len=4, min_word_count=2)
                or self.soft_literal_match(query, "directors", self.popular_directors)
                or self.keyword_fuzzy_match(query, "directors", self.popular_directors)
            ),

            "keywords": self.literal_match(query, "keywords", min_word_len=5, min_word_count=2)
                        or self.keyword_fuzzy_match(query, "keywords"),

            "emotions": self.literal_match(query, "emotions", min_word_len=4, min_word_count=1)
                        or self.keyword_fuzzy_match(query, "emotions")
        }