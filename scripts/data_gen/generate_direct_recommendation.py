import argparse
import json
import os
import time
import random
import pandas as pd
from openai import OpenAI

# System message to guide GPT-4.1 to generate accurate genre-based recommendations
SYSTEM_PROMPT = """You are generating training data for a movie recommendation chatbot.

Given a movie's metadata, do the following:

1. Create a realistic user query requesting a movie of a specific genre. The query should be brief and natural — avoid technical terms.
2. Generate a chatbot-style response where the bot:
    - **Matches** the user's requested genre to the movie’s genres (exact match).
    - **Suggests similar genres** if the movie's genre matches the user's query, or if not, suggests a different genre that’s close.

The response should be casual, friendly, and include 1-3 fitting emojis. The movie's metadata should guide the recommendation — make it sound personal and engaging.
"""

# User prompt template to plug in the movie metadata
USER_PROMPT_TEMPLATE = """Movie metadata:
Title: {title}

Genres: {genres}  
Tags: {tags}  
Actors: {actors}  
Director: {director}  
Emotion Tags: {emotions}  
Overview: {overview}

Respond with 3 different JSON entries. Each should include:
- "instruction": a unique user query requesting a genre
- "output": a fun, emoji-rich recommendation response matching the genre and possibly suggesting similar ones

Example Queries:
1. "Can you recommend a fun action movie?"
2. "Looking for an emotional sci-fi movie."
3. "I want an adventure movie that's high-energy."
"""

# Function to make GPT-4.1 call
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

# Function to parse the GPT-4.1 response (JSON)
def parse_gpt_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"⚠️ Failed to parse GPT response:", text)
        return []

# Function to generate samples for each row (movie)
def generate_samples_for_row(row, client, model, temperature, top_p):
    prompt = USER_PROMPT_TEMPLATE.format(
        title=row['title'],
        genres=row['genres'],
        tags=row['keywords'],
        actors=", ".join(eval(row['actors'])),
        director=row.get('director', 'Unknown'),  # Optional: extract from another field if you have it
        emotions=", ".join(eval(row['emotions'])),
        overview=row['overview']
    )
    response_text = call_gpt(client, model, SYSTEM_PROMPT, prompt, temperature, top_p)
    return parse_gpt_json_response(response_text)

# Main function to handle the script flow
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True, help="OpenAI API key")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to your movie metadata CSV")
    parser.add_argument('--out_path', type=str, default='data/direct_recommendation_sft.json', help="Path to save the output file")
    parser.add_argument('--per_movie_dir', type=str, default='data/direct_recommendation_by_movie', help="Folder to save per-movie files")
    parser.add_argument('--model', type=str, default='gpt-4.1', help="GPT model name (default is gpt-4.1)")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for GPT-4.1 generation")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p value for GPT-4.1 generation")
    parser.add_argument('--max_rows', type=int, default=100, help="Limit number of rows to process from the CSV")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    df = pd.read_csv(args.csv_path).head(args.max_rows)
    os.makedirs(args.per_movie_dir, exist_ok=True)

    all_examples = []
    for i, row in df.iterrows():
        print(f"Generating for: {row['title']}")
        samples = generate_samples_for_row(row, client, args.model, args.temperature, args.top_p)
        movie_examples = []
        for s in samples:
            example = {
                "instruction": s.get("instruction", ""),
                "input": "",
                "output": s.get("output", "")
            }
            all_examples.append(example)
            movie_examples.append(example)

        # Save per-movie file
        movie_file = os.path.join(args.per_movie_dir, f"{row['title'].strip().replace(' ', '_').replace('/', '-')}.json")
        with open(movie_file, "w", encoding="utf-8") as f:
            json.dump(movie_examples, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    # Save the combined examples
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(all_examples)} examples to {args.out_path}")

if __name__ == "__main__":
    main()