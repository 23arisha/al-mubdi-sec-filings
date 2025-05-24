import streamlit as st
import re
import logging
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document

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
    white-space: pre-wrap;
}
</style>


"""


# Load models
chat_model = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama3-8b-8192"
)

# Prompt template
prompt_template = """
You are a financial analyst tasked with producing a detailed and actionable summary of a company’s 10-K filing. Do not simply list or paraphrase sections—instead, deeply analyze the content to extract insights that traders or investors can use to make real decisions. Your goal is to rank, interpret, and connect the dots to form a forward-looking picture of the company’s performance and potential stock implications. Begin by briefly explaining the company’s core business model, its key revenue streams, and how current macroeconomic or industry conditions may affect it. When covering risk factors, do not just list them—analyze how each one could realistically impact the company’s performance or valuation. Highlight whether each risk is new, worsening, or improving compared to previous years, assess the tone used by management when discussing it, note any mitigation strategies, and rank them by both likelihood and material impact. Only include risks that are truly relevant; avoid generic or boilerplate ones. For legal proceedings, summarize any lawsuits or regulatory actions and estimate their potential financial or reputational cost, noting whether the company is transparent or downplaying their importance. In the MD&A section, look for signs of strategic shifts, operational struggles, or subtle red flags like terms such as “optimization,” “cost efficiencies,” or “headcount reduction,” and identify disconnects between positive narrative and weak numbers. Evaluate the accounting section for anomalies like restatements, policy changes, or hidden liabilities, and assess whether the financial controls are strong or weak. Analyze trends in key financials, including revenue, margins, cash flow, and debt, and flag any signs of financial engineering such as stock buybacks funded by debt or one-time gains. Finally, conclude with a clear recommendation whether based on analysis Stock seem good or bad or what is condition, backed by specific observations from your analysis. Mention that Recommendation of stock based on this 10-k report only so user should not rely on it only. If applicable, mention any recent news or guidance that may impact the company’s outlook. Avoid using headings like "Risk Factors" or "MD&A"—instead, present your findings in a clear, cohesive narrative format that is concise yet analytical, without filler or paraphrasing of obvious statements.
"""
# Helper Functions
def clean_text(text):
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)
    text = re.sub(r'(?<=\S)\n(?=\S)', ' ', text)
    text = re.sub(r'[•#]', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def extract_section(text, start_pattern, end_pattern):
    start = re.search(start_pattern, text, re.IGNORECASE)
    end = re.search(end_pattern, text, re.IGNORECASE)
    if start and end:
        return text[start.end():end.start()].strip()
    elif start:
        return text[start.end():].strip()
    return ""

def extract_sections_from_10k_html(url):
    headers = {'User-Agent': 'your-email@example.com'}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup(['script', 'style', 'table']):
        tag.decompose()
    text = clean_text(soup.get_text(separator='\n'))
    sections = {
        "Business Overview": extract_section(text, r"item\s+1\.*\s*business", r"item\s+1a\.*"),
        "Risk Factors": extract_section(text, r"item\s+1a\.*\s*risk factors", r"item\s+1b\.*"),
        "Legal Proceedings": extract_section(text, r"item\s+3\.*\s*legal proceedings", r"item\s+4\.*"),
        "MD&A": extract_section(text, r"item\s+7\.*\s+management.*?operations", r"item\s+7a\.*"),
        "Market Risk": extract_section(text, r"item\s+7a\.*.*?market risk", r"item\s+8\.*"),
        "Financial Statements": extract_section(text, r"item\s+8\.*\s*financial statements", r"item\s+9\.*"),
        "Controls and Procedures": extract_section(text, r"item\s+9a\.*\s*controls and procedures", r"item\s+9b\.*"),
        "Executive Compensation": extract_section(text, r"item\s+11\.*\s*executive compensation", r"item\s+12\.*"),
    }
    return " ".join([f"{k}\n{v}" for k, v in sections.items() if v])

# Streamlit UI
st.markdown(page_styles, unsafe_allow_html=True)
st.title("AL-Mubdi SEC 10-K Filing Summarizer")


url = st.text_input("Enter the 10-K filing URL:",)

if st.button("Summarize"):
    if not url:
        st.error("Please enter a URL.")
    else:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder="./hf_model_cache",  # safer default
            )
            status_placeholder = st.empty()

            # Show the loading message
            status_placeholder.info("Fetching and summarizing...")
            text = extract_sections_from_10k_html(url)
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
            query = prompt_template + "\n\nPlease evaluate the 10-K section in detail."
            summary = qa_chain.run(query)
            # Clear the loading message
            status_placeholder.empty()

            # Show success message and summary
            st.success("Summary Ready!")
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

