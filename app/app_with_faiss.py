import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
import torch
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

#config
FAISS_INDEX_PATH = "data/overview_faiss_full.index"
FAISS_META_PATH = "data/overview_metadata_full.pkl"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
RERANK_K = 20

#streamlit
st.set_page_config(page_title="STORY: Your Movie Recommender", page_icon="🎬")

#Load embedding model, FAISS index, metadata, and rerankers
@st.cache_resource
def load_faiss_and_models():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        metadata = pickle.load(f)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    return index, metadata, embed_model, reranker

faiss_index, metadata, embed_model, reranker = load_faiss_and_models()

# === Load LLaMA Lora for chatbot ===
@st.cache_resource
def load_llama_lora():
    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_model_id = "koreankiwi99/llama3.1-8b-moviebot-lora"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_model_id, device_map="auto")
    model.eval()
    return tokenizer, model

tokenizer, chatbot_model = load_llama_lora()

# === RAG pipeline ===
def retrieve_and_rerank(query):
    emb = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = faiss_index.search(np.array(emb).astype("float32"), RERANK_K)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx]
        candidates.append({
            **item,
            "score": float(score)
        })

    pairs = [[query, f"{c['title']} {c.get('overview', '')}"] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for i, s in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(s)

    sorted_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_results[:TOP_K]

# === Streamlit UI ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "text": "Hi! 🎬 I'm STORY, your Movie Recommender Friend. Ask me for a movie recommendation!"}
    ]

st.title("🎥 STORY: Your Movie Recommender Friend")

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["text"])

if prompt := st.chat_input("You:"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "text": prompt})

    # === Retrieve results ===
    with st.spinner("Finding the best movie for you..."):
        top_movies = retrieve_and_rerank(prompt)

    with st.chat_message("assistant"):
        with st.spinner("STORY is replying..."):
            context_str = "\n".join([f"- {m['title']} ({', '.join(m['genres'])})" for m in top_movies])
            user_query = f"User: {prompt}"
            
            # After top-1 reranked movie
            selected = top_movies[0]
            movie_text = (
                f"Title: {selected['title']}\n"
                f"Genres: {', '.join(selected['genres'])}\n"
                f"Tags: {', '.join(selected['keywords'])}\n"
                f"Actors: {', '.join(selected['actors'])}\n"
                f"Emotion Tags: {', '.join(selected['emotions'])}\n"
                f"Overview: {selected['overview']}"
            )

            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}\n\n{movie_text}"}
            ]
            full_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(full_prompt, return_tensors="pt").to(chatbot_model.device)
            output = chatbot_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            input_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "text": response})