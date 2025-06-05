
import os
import streamlit as st
from huggingface_hub import login
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import fitz  # PyMuPDF

# Secure login to Hugging Face using secret token
try:
    hf_token = st.secrets["HF_TOKEN"]
    login(token=hf_token)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
except KeyError:
    st.error("Hugging Face token not found in secrets. Please configure it.")
    st.stop()

# UI styling
st.title("📘 GHA SpecBot")
st.caption("Ask questions from the Ghana Highway Authority Standard Specification (2007)")

# PDF Upload
pdf_file = st.file_uploader("📄 Upload GHA Specification PDF", type="pdf")

if pdf_file:
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    st.success("✅ PDF successfully loaded!")

    # Chunk and embed text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    with st.spinner("🔍 Embedding document..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(chunks, embedding=embeddings)

    st.success("✅ Document indexed and ready!")

    # User input
    user_input = st.text_input("💬 Ask a question:")
    if user_input:
        with st.spinner("🧠 Thinking..."):
            retriever = vectordb.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
                ),
                retriever=retriever
            )
            response = qa_chain.run(user_input)
        st.markdown("### 📌 Answer")
        st.success(response)
