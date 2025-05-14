import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from accelerate import init_empty_weights, infer_auto_device_map

st.set_page_config(page_title="Chat with your Movie Recommender Friend!", page_icon="🤖")

@st.cache_resource
def load_llama_lora():

    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_model_id = "koreankiwi99/llama3.1-8b-moviebot-lora"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    device_map = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, lora_model_id, device_map=device_map)
    model.eval()
    return tokenizer, model


tokenizer, model = load_llama_lora()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": "Hi! 🎬 I'm STORY, your Movie Recommender Friend. Ask me for a movie recommendation!"}]

st.title("🎥 STORY: Your Movie Recommender Friend")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["text"])

if prompt := st.chat_input("You:"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "text": prompt})

    chat_history = "\\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in st.session_state.messages])
    full_prompt = f"{chat_history}\\nAssistant:"

    with st.chat_message("assistant"):
        with st.spinner("STORY is thinking... 🎞️"):
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            reply = tokenizer.decode(output[0], skip_special_tokens=True)
            reply = reply.split("Assistant:")[-1].strip().split("User:")[0].strip()
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "text": reply})
