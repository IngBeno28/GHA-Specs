import os
import streamlit as st
from huggingface_hub import login, HfFolder
import chromadb
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

# Hugging Face authentication
def setup_huggingface():
    try:
        hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        
        if not hf_token:
            st.error("Hugging Face token not found. Please configure it in secrets.toml or environment variables.")
            st.stop()
            
        if not hf_token.startswith("hf_"):
            st.error("Invalid token format. Hugging Face tokens should start with 'hf_'")
            st.stop()
            
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        login(token=hf_token, add_to_git_credential=False)
        HfFolder.save_token(hf_token)
        return True
            
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.stop()

setup_huggingface()

# ChromaDB client initialization
def get_chroma_client():
    try:
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {str(e)}")
        st.error("Please ensure SQLite >= 3.35.0 is installed or try Chroma's cloud version.")
        st.stop()

# UI
st.title("📘 GHA SpecBot")
st.caption("Ask questions from the Ghana Highway Authority Standard Specification (2007)")

# PDF Processing
pdf_file = st.file_uploader("📄 Upload GHA Specification PDF", type="pdf")

if pdf_file:
    try:
        with st.spinner("📖 Reading PDF..."):
            text = ""
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
        st.success("✅ PDF successfully loaded!")

        # Document processing
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])

        with st.spinner("🔍 Embedding document..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            chroma_client = get_chroma_client()
            
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                client=chroma_client,
                collection_name="gha_specs",
                persist_directory="./chroma_db"
            )
            vectordb.persist()

        st.success("✅ Document indexed and ready!")

        # QA System
        user_input = st.text_input("💬 Ask a question:")
        if user_input:
            with st.spinner("🧠 Thinking..."):
                try:
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
                    
                    st.markdown("### 📌 Answer")
                    st.success(result.get("result", "I couldn't find an answer to your question."))
                    
                    with st.expander("🔍 See source documents"):
                        for doc in result.get("source_documents", []):
                            st.write(doc.page_content)
                            st.write("---")
                            
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")
                    
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")

# Enhanced Footer with error handling
try:
    footer = """
    <div style="
        font-size: 0.8rem;
        color: #6c757d;
        text-align: center;
        padding: 10px;
        margin-top: 30px;
    ">
        <p>GHA SpecBot v1.1 | © 2007 Ghana Highway Authority</p>
        <p>Powered by LangChain, ChromaDB and HuggingFace | Python {version}</p>
        <p>For support contact: wiafe1713@gmail.com</p>
    </div>
    """.format(version=sys.version.split()[0])
    st.markdown(footer, unsafe_allow_html=True)
except Exception as e:
    st.warning("Footer could not be displayed properly")
