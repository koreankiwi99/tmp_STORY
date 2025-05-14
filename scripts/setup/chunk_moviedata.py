import csv
import os

INPUT_CSV = "data/movie_metadata.csv"
CHUNK_SIZE = 100
OUTPUT_DIR = "data/movie_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_CSV, encoding="utf-8") as f:
    reader = list(csv.reader(f))
    header = reader[0]
    rows = reader[1:]

    total_chunks = (len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(total_chunks):
        chunk_rows = rows[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        output_path = os.path.join(OUTPUT_DIR, f"chunk_{i+1}.csv")
        with open(output_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            writer.writerows(chunk_rows)

print(f"✅ Split into {total_chunks} files in {OUTPUT_DIR}")