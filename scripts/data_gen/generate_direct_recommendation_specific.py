import argparse
import json
import os
import time
import pandas as pd
from openai import OpenAI
import ast

# Final system prompt
SYSTEM_PROMPT = """You are generating training data for a movie recommendation chatbot.

Given a movie's metadata, do the following:

1. Create a **realistic user query** that is detailed and specific, combining multiple clear preferences. The query should feel natural and personal, but include 2–3 specific elements such as genre, mood, themes, decade, actors, or plot features.

   ❌ Avoid vague or open-ended queries like “something good” or “anything intense.”

   ✅ Use queries like:
      - “a feel-good road trip movie from the 90s with great music”
      - “a thrilling mystery set in a snowy small town”
      - “an animated film about growing up that’ll make me cry”

2. Generate a chatbot-style recommendation response that recommends **only the given movie**. The tone should be friendly, casual, and include 1–3 appropriate emojis (e.g., 🎬, 🦸, 😢, ❤️, 😂).

Avoid being too formal. Make it sound like a fun, helpful chatbot giving a personal suggestion.
"""

# User prompt template
USER_PROMPT_TEMPLATE = """Movie metadata:
Title: {title}

Genres: {genres}

Tags: {tags}

Actors: {actors}

Director: {director}

Emotion Tags: {emotions}

Overview: {overview}

Now write 3 JSON entries like this:

{{
"instruction": "a short, casual user query someone might type when they want a movie like this",
"output": "a fun, emoji-rich response that recommends only this movie in a friendly, personal tone"
}}
"""

# Call OpenAI API
def call_gpt(client, model, system_prompt, user_prompt, temperature, top_p):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message.content

# Parse GPT output as JSON lines
def parse_gpt_json_response(text):
    try:
        # Try loading the full response as a JSON array
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: line-by-line parsing if not a valid JSON array
    results = []
    for line in text.strip().split('\n'):
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results

# Generate examples from one movie row
def generate_samples_for_row(row, client, model, temperature, top_p):
    # Parse actors (string → list), cut to 5, fallback to ['Unknown'] if empty
    actors = ast.literal_eval(row["actors"]) if pd.notna(row["actors"]) else []
    actors = actors[:5] if actors else ["Unknown"]

    # Parse emotions (string → list), fallback to ['Unknown'] if empty
    emotions = ast.literal_eval(row["emotions"]) if pd.notna(row["emotions"]) else []
    emotions = emotions if emotions else ["Unknown"]

    # Use director or "Unknown"
    director = row["director"] if pd.notna(row["director"]) else "Unknown"

    # Format prompt
    prompt = USER_PROMPT_TEMPLATE.format(
        title=row["title"],
        genres=row["genres"],
        tags=row["keywords"],
        actors=", ".join(actors),
        director=director,
        emotions=", ".join(emotions),
        overview=row["overview"]
    )

    response_text = call_gpt(client, model, SYSTEM_PROMPT, prompt, temperature, top_p)
    return parse_gpt_json_response(response_text)

# Main script logic
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default="data/movie_metadata.csv")
    parser.add_argument("--per_movie_dir", type=str, default="data/recommendation_batches")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_rows", type=int, default=100)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    df = pd.read_csv(args.csv_path).head(args.max_rows)
    os.makedirs(args.per_movie_dir, exist_ok=True)

    for i, row in df.iterrows():
        movie_id = str(row["id"])
        out_path = os.path.join(args.per_movie_dir, f"{movie_id}.jsonl")

        if os.path.exists(out_path):
            print(f"Skipping {row['title']} (ID: {movie_id}) — already exists.")
            continue

        print(f"Generating for {row['title']} (ID: {movie_id})")
        try:
            samples = generate_samples_for_row(row, client, args.model, args.temperature, args.top_p)
            with open(out_path, "w", encoding="utf-8") as f:
                for s in samples:
                    json.dump({
                        "instruction": s.get("instruction", ""),
                        "output": s.get("output", "")
                    }, f, ensure_ascii=False)
                    f.write("\n")
            print(f"Saved to {out_path}")
        except Exception as e:
            print(f"Error generating for {row['title']} (ID: {movie_id}): {e}")

        time.sleep(1)

if __name__ == "__main__":
    main()