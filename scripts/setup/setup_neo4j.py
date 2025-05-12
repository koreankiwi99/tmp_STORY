import csv
import argparse
from neo4j import GraphDatabase
from tqdm import tqdm
import ast

CSV_PATH = "./data/movie_metadata.csv"

def safe_parse_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def create_movie_node(tx, movie):
    tx.run("""
        MERGE (m:Movie {id: $id})
        SET m.title = $title,
            m.year = $year

        WITH m
        UNWIND $actors AS actor
        MERGE (a:Actor {name: actor})
        MERGE (m)-[:HAS_ACTOR]->(a)

        WITH m
        UNWIND $genres AS genre
        MERGE (g:Genre {name: genre})
        MERGE (m)-[:HAS_GENRE]->(g)

        WITH m
        UNWIND $emotions AS emotion
        MERGE (e:Emotion {name: emotion})
        MERGE (m)-[:HAS_EMOTION]->(e)

        WITH m
        UNWIND $keywords AS keyword
        MERGE (k:Keyword {name: keyword})
        MERGE (m)-[:HAS_KEYWORD]->(k)
    """, movie)

def upload_to_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # First count total rows for tqdm
    with open(CSV_PATH, encoding="utf-8") as f:
        total_rows = sum(1 for _ in f) - 1  # subtract header

    with driver.session() as session, open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_rows, desc="Uploading movies"):
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
                session.write_transaction(create_movie_node, movie)
            except Exception as e:
                print(f"Skipping movie due to error: {e}\n→ {row.get('title', '')}")

    driver.close()
    print("✅ Neo4j setup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="bolt+s://4732744a.databases.neo4j.io", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    args = parser.parse_args()

    upload_to_neo4j(args.uri, args.user, args.password)