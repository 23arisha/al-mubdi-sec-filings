import streamlit as st
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
# Define prompt templates
from prompt_templates import prompt_template_10K, prompt_template_4
# Import utility functions for extracting sections from filings
from utils import extract_sections_from_10k_html, extract_form4_text

logging.basicConfig(level=logging.INFO)

page_styles = """
<style>
/* Page background */
[data-testid="stAppViewContainer"] {
    background-color: #036803;
}

[data-testid="stHeader"] {
    background-color: #036803;
}

/* Button styling */
.stButton > button {
    background-color: #ffc107;
    color: black;
    border: none;
    padding: 0.5em 1em;
    border-radius: 8px;
    font-weight: bold;
}

/* Button hover */
.stButton > button:hover {
    background-color: #e0a800;
    color: black;
    border: none;
}

input[type="text"] {
    background-color: white;
    color: black;
    border: 1px solid #ccc;
    padding: 0.5em;
    border-radius: 8px;
}

.summary-box {
    background-color: white;
    color: black;
    padding: 1.5em;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
    margin-top: 1em;
}
</style>

"""

# Load models
chat_model = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile"
)

# Streamlit UI
st.markdown(page_styles, unsafe_allow_html=True)
st.title("AL-Mubdi SEC Filing Summarizer")


url = st.text_input("Enter the filing URL:",)

form_type = st.selectbox("Select filing type:", options=["10-K", "4"], index=0)

if st.button("Summarize"):
    if not url:
        st.error("Please enter a URL.")
    else:
        try:
            status_placeholder = st.empty()
            status_placeholder.info("Fetching and summarizing...")

            if form_type == "10-K":
                text = extract_sections_from_10k_html(url)
                if not text.strip():
                    st.warning("❗ No recognizable 10-K sections found in the URL. Please check the URL and try again.")
                    status_placeholder.empty()
                    st.stop()

                prompt = prompt_template_10K
            elif form_type == "4":
                text = extract_form4_text(url)
                if not text.strip():
                    st.warning("❗ No recognizable Form 4 content found. Please check the URL and try again.")
                    status_placeholder.empty()
                    st.stop()

                prompt = prompt_template_4
            else:
                st.error(f"Unsupported form type: {form_type}")
                status_placeholder.empty()
                st.stop()

            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder="./hf_model_cache",
            )
            documents = [Document(page_content=text)]
            vectordb = VectorstoreIndexCreator(
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100),
                embedding=embedding_model
            ).from_documents(documents)
            qa_chain = RetrievalQA.from_chain_type(
                llm=chat_model,
                retriever=vectordb.vectorstore.as_retriever(),
                chain_type="stuff"
            )

            query = prompt + "\n\nPlease analyze the filing in detail."
            summary = qa_chain.run(query)

            status_placeholder.empty()
            st.success("Summary Ready!")
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            status_placeholder.empty()