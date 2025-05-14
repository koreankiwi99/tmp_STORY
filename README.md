<p align="center">
  <img src="./logo.png" alt="STORY Logo" width="180"/>
</p>

# 🎬 STORY: Your Movie Recommender Friend

**STORY** is a conversational movie recommendation assistant powered by:
- 🔍 Semantic search using **FAISS** + `BAAI/bge-m3`
- 🔁 Reranking via **CrossEncoder**
- 💬 Chat-style recommendation with **LLaMA 3.1 8B + LoRA**
- 🎥 Metadata and posters from **TMDB**
- 🧠 Modular graph-based retrieval with **Neo4j** (in progress)

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/abcd1234/0.jpg)](https://www.youtube.com/watch?v=abcd1234)

---

## ✨ Features

- Natural language queries like “something fun with DiCaprio”
- Emotion-, actor-, and genre-aware movie retrieval
- TMDB poster previews
- Streamlit chatbot interface
- Easy-to-extend RAG pipeline

---

## 🚧 In Progress

- 🧠 **Neo4j + MovieQuerySegmenter** module for structured search  
  → Automatically extract filters (actor, genre, emotion) from user input  
  → Generate Cypher queries from plain text

---

## 🛠️ Quickstart

```bash
# Clone and install
git clone https://github.com/yourusername/STORY.git
cd STORY
pip install -r requirements.txt

# Preprocess movie data
python scripts/setup/chunk_moviedata.py

# Build FAISS index
python scripts/setup/build_faiss.py

# Run the Streamlit app
streamlit run app/app_with_faiss.py
