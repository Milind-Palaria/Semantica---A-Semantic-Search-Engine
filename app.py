import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd
from report_generator import generate_csv, generate_excel

indexName = "user_uploaded_data"
try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "12345678"),
        verify_certs=False,
    )
except ConnectionError as e:
    st.error(f"Connection Error: {e}")

if es.ping():
    st.success("Successfully connected to Elasticsearch!", icon="✅")
else:
    st.error("Cannot connect to Elasticsearch!")

st.header("1. Select a Model")
selected_model = st.selectbox("Choose a Sentence Transformer model", ['paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2'])

st.header("2. Upload Your CSV Dataset")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.fillna("none", inplace=True)

    st.subheader("Dataset Preview:")
    st.dataframe(df.head(), use_container_width=True)

    st.header("3. Customize Search Results")
    text_column = st.selectbox("Select the text column (e.g., description)", df.columns)
    id_column = st.selectbox("Select the unique ID column", df.columns)
    display_columns = st.multiselect("Choose columns to display in results", df.columns.tolist(), default=[text_column, id_column])

    if st.button("Process and Index Dataset"):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text.text("Progress: 0%")

        st.write("Starting to process the dataset...")
        model = SentenceTransformer(selected_model)
        total_rows = len(df)

        df['DescriptionVector'] = df[text_column].apply(lambda x: model.encode(str(x), clean_up_tokenization_spaces=False) if isinstance(x, str) else model.encode("", clean_up_tokenization_spaces=False))
        record_list = df.to_dict("records")
        if not es.indices.exists(index=indexName):
            es.indices.create(index=indexName)

        for i, record in enumerate(record_list):
            try:
                es.index(index=indexName, document=record, id=record[id_column])
            except Exception as e:
                st.error(f"Error: {e}")
            
            progress = int((i + 1) / total_rows * 100)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {progress}%")

        st.success("Data indexed successfully!", icon="✅")
        progress_text.text("Progress: 100%")

st.header("4. Search the Indexed Data")
search_query = st.text_input("Enter your search query")

if st.button("Search"):
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Progress: 0%")

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

        progress_bar.progress(100)
        progress_text.text("Progress: 100%")

        search_results = pd.DataFrame([result['_source'] for result in results])

        st.subheader("Search Results:")
        st.dataframe(search_results)

        st.header("5. Export Search Results")

        csv_data = generate_csv(search_results)
        st.download_button(
            label="Download Search Results as CSV",
            data=csv_data,
            file_name='search_results.csv',
            mime='text/csv',
        )

        excel_data = generate_excel(search_results)
        st.download_button(
            label="Download Search Results as Excel",
            data=excel_data,
            file_name='search_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

    except Exception as e:
        st.error(f"Search failed: {e}")
