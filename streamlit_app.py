import streamlit as st
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import gdown
import torch

# Show title and description.
st.title("Panini Parser")

if "model" not in st.session_state:
    url = "https://drive.google.com/file/d/1-kU1fY8LdiNa-osQwpaGO43vB3LN88-7/view?usp=drive_link"
    output = "model.bin"
    gdown.download(url, output, quiet=False)
    st.session_state.model = output

else: 
    st.write("hello")
