import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Page configuration
st.set_page_config(
    page_title="Chat with your Movie Recommender Friend!",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cache model and tokenizer loading
@st.cache_resource
def load_llama_lora():
    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_model_id = "kiwi1229/llama3.1-8b-moviebot-lora"

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )

    # Merge LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_model_id)
    model.eval()
    return tokenizer, model
    
#def load_blenderbot():
#    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
#    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
#    return tokenizer, model

tokenizer, model = load_llama_lora()

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

    # Show user message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "text": prompt})

    # Format the prompt (you can customize this part based on how your model was fine-tuned)
    chat_history = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in st.session_state.messages])
    full_prompt = f"{chat_history}\nAssistant:"

    # Generate reply
    with st.chat_message("assistant"):
        with st.spinner("MovieBot is thinking..."):
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            reply = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract only the final assistant message
            reply = reply.split("Assistant:")[-1].strip().split("User:")[0].strip()
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "text": reply})
