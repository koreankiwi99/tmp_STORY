import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

# Page configuration
st.set_page_config(
    page_title="Chat with your Movie Recommender Friend!",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cache model and tokenizer loading
@st.cache_resource
def load_blenderbot():
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    return tokenizer, model

tokenizer, model = load_blenderbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "text": "Hi! 🎬 I'm your Movie Recommender Friend. Ask me for a movie recommendation or tell me what you like!"
        }
    ]

st.title("Your Movie Recommender Friend!")

# Display chat messages
for msg in st.session_state.messages:
    role = msg["role"]
    text = msg["text"]
    st.chat_message(role).write(text)

# Chat input (Streamlit's Chat-like UI)
if prompt := st.chat_input("You:"):

    # 1) Immediately show the user’s message
    st.chat_message("user").write(prompt)

    # 2) Save to history so it persists
    st.session_state.messages.append({"role": "user", "text": prompt})

    # 3) Now show the assistant bubble and stream or spinner
    with st.chat_message("assistant"):
        with st.spinner("BlenderBot is thinking…"):
            inputs = tokenizer(prompt, return_tensors="pt")
            reply_ids = model.generate(
                **inputs,
                max_length=100,
                num_beams=3,
                early_stopping=True
            )
            reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Write the final text
        st.write(reply)

    # 4) Save the assistant reply too
    st.session_state.messages.append({"role": "assistant", "text": reply})