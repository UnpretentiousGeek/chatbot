import streamlit as st
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import gdown
import torch

# Show title and description.
st.title("Panini Parser")

if "model" not in st.session_state:
    url = "https://drive.google.com/file/d/1-kU1fY8LdiNa-osQwpaGO43vB3LN88-7/view?usp=drive_link"
    gdown.download(url, st.session_state.model, quiet=False)

else: 
    st.write("hello")
