import streamlit as st
from elasticsearch import Elasticsearch

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