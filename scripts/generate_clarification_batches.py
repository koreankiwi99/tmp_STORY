import argparse
import json
import os
import time
import random
from openai import OpenAI

# Prompt modifiers to rotate constraints per batch
PROMPT_VARIANTS = [
    "Focus more on light-hearted comedies or quirky suggestions.",
    "Avoid nostalgic or retro themes like 80s or 90s references.",
    "Keep the examples modern and grounded in recent trends.",
    "No era-based queries this time — focus on moods and genres.",
    "Lean into emotional or thought-provoking requests.",
    "Favor unusual genres or tones (e.g., surreal, experimental, feel-good).",
    "Focus more on animated films, shorts, or offbeat categories.",
    "Use phrasing that sounds more like casual texting or slang.",
    "Include more questions based on tone (e.g., relaxing, exciting, weird).",
]

SYSTEM_MSG = """You are helping create high-quality and diverse clarification examples for a movie recommendation chatbot.

The user queries (\"instruction\") should be vague but clearly related to movies — something a person might casually type when they’re not sure what they want to watch.

Your task is to:
1. Write a short, natural-sounding movie-related query.
2. Write a friendly follow-up message from the chatbot that asks for clarification to better recommend a movie.

Your clarification questions should refer to movie-specific aspects, such as:
- \ud83c\udfae genre (e.g., horror, rom-com, sci-fi)
- \ud83c\udfad tone or mood (e.g., emotional, chill, fast-paced)
- \ud83c\udf9d\ufe0f format (e.g., animated, classic, short film)
- \ud83d\udd70\ufe0f era or style (e.g., 80s vibe, vintage noir, old-school)
- \ud83d\udc64 actors/directors (without overusing names)

\ud83c\udfa8 Tone: Make responses warm, curious, concise, or playful — vary them across examples. Use emojis naturally where appropriate.

\u274c Avoid:
- Repeating the same sentence structures, phrases, or emoji combinations
- Using templates like \"Something <adjective>\" more than once per batch
- Overusing the word \"movie\" — get creative with how people ask
"""

BASE_USER_MSG = """Generate 10 diverse JSON-formatted clarification examples for training a movie recommendation chatbot.

Each example must:
- Include a unique and varied movie-related query in the \"instruction\" field  
- Include a friendly, casual clarification message in the \"output\" field
- Use different phrasing styles across examples (fragments, questions, casual language)
- Vary the tone of the follow-up (warm, curious, humorous, concise)
- Include emojis where appropriate, and avoid reusing the same combinations

Use this format:
{
  \"instruction\": \"...\",
  \"output\": \"...\"
}
"""

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

def parse_json_lines(text):
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    results = []
    current_block = ""
    for line in text.splitlines():
        if line.strip().startswith("{"):
            current_block = line.strip()
        elif current_block:
            current_block += line.strip()
            if current_block.endswith("}"):
                try:
                    obj = json.loads(current_block)
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                current_block = ""
    return results

def save_batch(batch_data, batch_number, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"clarification_batch_{batch_number:02d}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved batch {batch_number:02d} to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True, help="OpenAI API key")
    parser.add_argument('--model', type=str, default='gpt-4.1-mini-2025-04-14', help="Model name (e.g., gpt-4.1-mini)")
    parser.add_argument('--batches', type=int, default=30, help="Number of 10-sample batches to generate")
    parser.add_argument('--out_dir', type=str, default='data/clarification_batches', help="Directory to save each batch")
    parser.add_argument('--temperature', type=float, default=1.0, help="Default sampling temperature")
    parser.add_argument('--top_p', type=float, default=0.95, help="Default top-p value")
    parser.add_argument('--sleep', type=float, default=1.0, help="Delay between requests (sec)")
    parser.add_argument('--randomize', action='store_true', help="Randomize temperature/top_p and constraints per batch")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    os.makedirs(args.out_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(args.out_dir)
        if f.startswith("clarification_batch_") and f.endswith(".json")
    ]
    existing_numbers = [
        int(f.split("_")[-1].replace(".json", "")) for f in existing if f.split("_")[-1].replace(".json", "").isdigit()
    ]
    start_index = max(existing_numbers, default=0) + 1

    for i in range(start_index, start_index + args.batches):
        temp = round(random.uniform(1.0, 1.3), 2) if args.randomize else args.temperature
        top_p = round(random.uniform(0.9, 0.98), 2) if args.randomize else args.top_p
        constraint = random.choice(PROMPT_VARIANTS) if args.randomize else ""
        user_msg = BASE_USER_MSG + ("\n\nAdditional constraint for this batch: " + constraint if constraint else "")

        print(f"Generating batch {i} | temperature={temp}, top_p={top_p}")
        if constraint:
            print(f"Constraint: {constraint}")

        try:
            text = call_gpt(client, args.model, SYSTEM_MSG, user_msg, temp, top_p)
            batch = parse_json_lines(text)
            save_batch(batch, i, args.out_dir)
        except Exception as e:
            print(f"Error in batch {i}: {e}")
        time.sleep(args.sleep)

if __name__ == "__main__":
    main()