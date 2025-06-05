import os
import streamlit as st
from huggingface_hub import login
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import fitz  # PyMuPDF
import sqlite3
import sys

# Display package versions for debugging
st.sidebar.markdown("### Environment Info")
st.sidebar.text(f"Python: {sys.version.split()[0]}")
st.sidebar.text(f"SQLite: {sqlite3.sqlite_version}")

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
    try:
        with st.spinner("📖 Reading PDF..."):
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
            # Use persistent storage
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            vectordb.persist()

        st.success("✅ Document indexed and ready!")

        # User input
        user_input = st.text_input("💬 Ask a question:")
        if user_input:
            with st.spinner("🧠 Thinking..."):
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=HuggingFaceHub(
                        repo_id="google/flan-t5-large",
                        model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
                    ),
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                result = qa_chain({"query": user_input})
                response = result["result"]
                
            st.markdown("### 📌 Answer")
            st.success(response)
            
            # Show source documents
            with st.expander("🔍 See source documents"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)
                    st.write("---")
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please ensure you have the correct dependencies installed.")

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: #6c757d;
    text-align: center;
    padding: 10px;
    margin-top: 30px;
}
</style>
<div class="footer">
    <p>GHA SpecBot v1.1 | © 2007 Ghana Highway Authority</p>
    <p>Powered by Automation_Hub | Python {python_version}</p>
    <p>For support contact: specs@ghanahighways.gov.gh</p>
</div>
""".format(python_version=sys.version.split()[0]), unsafe_allow_html=True)
