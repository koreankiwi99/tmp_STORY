import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(page_title="Chat with your Movie Recommender Friend!", page_icon="🤖")

@st.cache_resource
def load_llama_lora():
    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_model_id = "koreankiwi99/llama3.1-8b-moviebot-lora"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
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

tokenizer, model = load_llama_lora()

# Initialize chat history with greeting (not used for prompting)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "text": "Hi! 🎬 I'm STORY, your Movie Recommender Friend. Ask me for a movie recommendation!"}
    ]

st.title("🎥 STORY: Your Movie Recommender Friend")

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["text"])

# User input box
if prompt := st.chat_input("You:"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "text": prompt})

    # Reconstruct messages (excluding the first assistant greeting)
    messages = [
        {"role": m["role"], "content": m["text"]}
        for i, m in enumerate(st.session_state.messages)
        if not (i == 0 and m["role"] == "assistant")
    ]

    # Optional: add a system message (only if you trained with it)
    # messages.insert(0, {"role": "system", "content": "You are STORY, a fun movie recommender chatbot."})

    # Format chat prompt using LLaMA 3.1 chat template
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    with st.chat_message("assistant"):
        with st.spinner("STORY is thinking... 🎞️"):
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.9,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # Trim off prompt portion from output
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_len:]
            reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "text": reply})