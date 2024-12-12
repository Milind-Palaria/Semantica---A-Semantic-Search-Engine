import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd

st.title("Vector-Fusion Hub")


es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "12345678"),
    verify_certs=False,
)

if es.ping():
    st.success("Successfully connected to Elasticsearch!")
else:
    st.error("Cannot connect to Elasticsearch!")


selected_model = st.selectbox(
    "Choose a Sentence Transformer model",
    ['paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2']
)


uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())