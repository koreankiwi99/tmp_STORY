import csv
import argparse
import ast
import time
import random
from neo4j import GraphDatabase
from tqdm import tqdm

BATCH_SIZE = 100  # Adjust based on your network and database performance


def safe_parse_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def create_movie_batch(tx, movies):
    tx.run("""
        UNWIND $movies AS movie
        MERGE (m:Movie {id: movie.id})
        SET m.title = movie.title,
            m.year = movie.year

        WITH m, movie
        UNWIND movie.actors AS actor
        MERGE (a:Actor {name: actor})
        MERGE (m)-[:HAS_ACTOR]->(a)

        WITH m, movie
        UNWIND movie.genres AS genre
        MERGE (g:Genre {name: genre})
        MERGE (m)-[:HAS_GENRE]->(g)

        WITH m, movie
        UNWIND movie.emotions AS emotion
        MERGE (e:Emotion {name: emotion})
        MERGE (m)-[:HAS_EMOTION]->(e)

        WITH m, movie
        UNWIND movie.keywords AS keyword
        MERGE (k:Keyword {name: keyword})
        MERGE (m)-[:HAS_KEYWORD]->(k)
    """, {"movies": movies})


def _write_with_retries(session, batch, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            session.execute_write(create_movie_batch, batch)
            return
        except Exception as e:
            wait_time = 2 ** attempt + random.uniform(0, 1)
            print(f"⚠️ Batch write failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"⏳ Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print("❌ Max retries reached — skipping this batch.")


def upload_to_neo4j(csv_path, uri, user, password):
    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
        max_connection_lifetime=600,
        max_transaction_retry_time=30,
        connection_timeout=30
    )

    with open(csv_path, encoding="utf-8") as f:
        total_rows = sum(1 for _ in f) - 1  # subtract header

    with driver.session() as session, open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in tqdm(reader, total=total_rows, desc=f"Uploading {csv_path}"):
            try:
                movie = {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "year": int(row["year"]) if row["year"] else None,
                    "genres": safe_parse_list(row["genres"]),
                    "emotions": safe_parse_list(row["emotions"]),
                    "actors": safe_parse_list(row["actors"]),
                    "keywords": row["keywords"].split(", ") if row["keywords"] else []
                }
                batch.append(movie)

                if len(batch) >= BATCH_SIZE:
                    _write_with_retries(session, batch)
                    batch = []

            except Exception as e:
                print(f"Skipping movie due to error: {e}\n→ {row.get('title', '')}")

        # Final leftover batch
        if batch:
            _write_with_retries(session, batch)

    driver.close()
    print("✅ Neo4j setup complete for", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the movie CSV chunk to upload")
    parser.add_argument("--uri", default="bolt+s://4732744a.databases.neo4j.io", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    args = parser.parse_args()

    upload_to_neo4j(args.csv, args.uri, args.user, args.password)