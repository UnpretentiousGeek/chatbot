import streamlit as st
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import gdown
import torch


def generate_text(prompt, my_model, myTokenizer, max_length=50, temperature=0.3, top_k=50, top_p=0.9):
    # Encode the input prompt
    input_ids = myTokenizer.encode(prompt, return_tensors='pt').to(my_model.device)
    attention_mask = torch.ones_like(input_ids)
    # Generate text
    output = my_model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=myTokenizer.eos_token_id
    )

    # Decode the generated text
    generated_text = myTokenizer.decode(output[0], skip_special_tokens=False)
    return generated_text

# Show title and description.
st.title("Panini Parser")
st.write("Ask anything to the chatbot!")

if "model" not in st.session_state:
    url = "https://drive.google.com/file/d/1-kU1fY8LdiNa-osQwpaGO43vB3LN88-7/view?usp=drive_link"
    output = "model.pth"
    gdown.download(url, output, quiet=False)
    config = GPT2Config(vocab_size=65536,n_head=16)
    st.session_state.model = GPT2LMHeadModel(config)
    model_save_path = output
    st.session_state.model.load_state_dict(torch.load(model_save_path))

if "token" not in st.session_state:

    tokenizer_path = 'fine_tuned_tokenizer.json'
    st.session_state.tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_path)

    if st.session_state.tokenizer.pad_token is None:
        if st.session_state.tokenizer.eos_token is not None:
            st.session_state.tokenizer.pad_token = st.session_state.tokenizer.eos_token
        else:
            st.session_state.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

user_input = st.text_input("You:", placeholder="Type your message here...")

if user_input:
    with st.spinner("Generating response..."):
        response = generate_text(user_input, st.session_state.model, st.session_state.tokenizer, max_length=50, temperature=0.3, top_k=50, top_p=0.9 )
        st.write(f"Chatbot: {response}")
