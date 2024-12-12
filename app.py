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
    st.subheader("Dataset Preview:")
    st.dataframe(df.head())

    st.header("Customize Search Results")
    text_column = st.selectbox("Select the text column (e.g., description)", df.columns)
    id_column = st.selectbox("Select the unique ID column", df.columns)
    display_columns = st.multiselect("Choose columns to display in results", df.columns.tolist(), default=[text_column, id_column])

    if st.button("Process and Index Dataset"):
        st.write("Processing the dataset...")
        model = SentenceTransformer(selected_model)
        df['DescriptionVector'] = df[text_column].apply(
            lambda x: model.encode(str(x), clean_up_tokenization_spaces=False) if isinstance(x, str) else model.encode("")
        )
        record_list = df.to_dict("records")
        index_name = "user_uploaded_data"
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name)

        for record in record_list:
            try:
                es.index(index=index_name, document=record, id=record[id_column])
            except Exception as e:
                st.error(f"Error indexing data: {e}")
        st.success("Data indexed successfully!")

st.header("Search Indexed Data")
search_query = st.text_input("Enter your search query")
if st.button("Search"):
    model = SentenceTransformer(selected_model)
    query_vector = model.encode(search_query)
    query = {
        "field": "DescriptionVector",
        "query_vector": query_vector,
        "k": 10,
        "num_candidates": 500
    }
    try:
        res = es.knn_search(index=index_name, knn=query, source=display_columns)
        results = res["hits"]["hits"]
        search_results = pd.DataFrame([result['_source'] for result in results])
        st.subheader("Search Results:")
        st.dataframe(search_results)
    except Exception as e:
        st.error(f"Search failed: {e}")
