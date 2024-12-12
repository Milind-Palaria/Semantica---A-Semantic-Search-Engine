import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd

indexName = "user_uploaded_data"
try:
    es = Elasticsearch(
        "http://localhost:9200",
        basic_auth=("elastic", "12345678"),  
        verify_certs=False 
    )
except Exception as e:
    st.error(f"Elasticsearch Connection Error: {e}")

if es.ping():
    st.success("Successfully connected to Elasticsearch!", icon="✅")
else:
    st.error("Cannot connect to Elasticsearch!")

st.header("1. Select a Model")
with st.container():
    selected_model = st.selectbox("Choose a Sentence Transformer model", ['paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2'])

st.header("2. Upload Your CSV Dataset")
with st.container():
    uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview:")
    st.dataframe(df.head(), use_container_width=True)

    st.write("Preprocessing dataset...")
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)  # Replace NaN in numeric columns with 0
    df = df.where(pd.notnull(df), None)  # Replace other NaN values with None

    st.header("3. Customize Search Results")
    text_column = st.selectbox("Select the text column (e.g., description)", df.columns)
    id_column = st.selectbox("Select the unique ID column", df.columns)
    display_columns = st.multiselect("Choose columns to display in results", df.columns.tolist(), default=[text_column, id_column])

    if st.button("Process and Index Dataset"):
        st.write("Processing the dataset...")
        model = SentenceTransformer(selected_model)

        try:
            df['DescriptionVector'] = df[text_column].apply(
                lambda x: model.encode(str(x), clean_up_tokenization_spaces=False) if isinstance(x, str) else model.encode("")
            )
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")

        record_list = df.to_dict("records")
        if not es.indices.exists(index=indexName):
            es.indices.create(index=indexName)

        for record in record_list:
            record = {key: (None if pd.isna(value) else value) for key, value in record.items()}
            try:
                es.index(index=indexName, document=record, id=record[id_column])
            except Exception as e:
                st.error(f"Error indexing data: {e}")
        st.success("Data indexed successfully!", icon="✅")

st.header("4. Search the Indexed Data")
search_query = st.text_input("Enter your search query")

if st.button("Search"):
    if search_query.strip():
        model = SentenceTransformer(selected_model)
        vector_of_input_keyword = model.encode(search_query)

        query = {
            "field": "DescriptionVector",
            "query_vector": vector_of_input_keyword,
            "k": 10,
            "num_candidates": 500
        }

        try:
            res = es.knn_search(index=indexName, knn=query, source=display_columns)
            results = res["hits"]["hits"]

            # Convert search results to a DataFrame for easier handling
            search_results = pd.DataFrame([result['_source'] for result in results])

            st.subheader("Search Results:")
            st.dataframe(search_results)

        except Exception as e:
            st.error(f"Search failed: {e}")
    else:
        st.warning("Please enter a valid search query.")
