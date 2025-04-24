# app.py
import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langdetect import detect
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Set your OpenAI API key here or through secrets/environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# UI setup
st.set_page_config(page_title="Persian Research Assistant", layout="wide")
st.title("🧠 دستیار تحقیقاتی با پشتیبانی از زبان فارسی")

uploaded_file = st.file_uploader("📄 لطفاً فایل PDF خود را آپلود کنید", type=["pdf"])
user_query = st.text_input("❓ سوال خود را وارد نمایید:")

# Helper function to extract text
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"

# Main logic
if uploaded_file and user_query:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    text = extract_text_from_pdf(tmp_file_path)
    lang = detect_language(text)
    st.info(f"زبان شناسایی‌شده: {'فارسی' if lang == 'fa' else lang}")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma.from_texts(texts, embedding=embeddings)

    prompt = user_query if lang != 'fa' else f"پاسخ را به زبان فارسی، دقیق و علمی ارائه بده: {user_query}"

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    response = qa.run(prompt)

    st.subheader("✍️ پاسخ دستیار:")
    st.write(response)

