import os
import streamlit as st
from huggingface_hub import login, HfFolder
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import fitz  # PyMuPDF
import hashlib
from typing import Optional, Tuple, List, Dict, Any

# Display package versions for debugging
st.sidebar.markdown("### Environment Info")
st.sidebar.text(f"Python: {sys.version.split()[0]}")
st.sidebar.text(f"SQLite: {pysqlite3.sqlite_version}")

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_KWARGS = {"k": 3}
LLM_TEMPERATURE = 0.3
MAX_NEW_TOKENS = 512
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "gha_specs"

# Hugging Face authentication with enhanced validation
def setup_huggingface() -> bool:
    """Initialize Hugging Face authentication with proper error handling."""
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

# ChromaDB client initialization with persistence
def get_chroma_client() -> from chromadb import PersistentClient:
    """Initialize and return a ChromaDB client with error handling."""
    try:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        return chromadb.PersistentClient(path=PERSIST_DIR)
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {str(e)}")
        st.error("Please ensure SQLite >= 3.35.0 is installed or try Chroma's cloud version.")
        st.stop()

def get_file_hash(file_bytes: bytes) -> str:
    """Generate a hash for the uploaded file to detect changes."""
    return hashlib.md5(file_bytes).hexdigest()

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF with proper error handling."""
    try:
        with st.spinner("📖 Reading PDF..."):
            text = ""
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        st.stop()

def process_document(text: str) -> Tuple[Chroma, str]:
    """Process and index the document text."""
    try:
        # Document splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.create_documents([text])
        
        # Embedding and vector store creation
        with st.spinner("🔍 Embedding document..."):
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            chroma_client = get_chroma_client()
            
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR
            )
            vectordb.persist()
            
        return vectordb, "✅ Document indexed and ready!"
        
    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        st.stop()

def initialize_qa_chain(vectordb: Chroma) -> RetrievalQA:
    """Initialize the QA chain with proper configuration."""
    return RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(
            repo_id=LLM_MODEL,
            model_kwargs={
                "temperature": LLM_TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS
            }
        ),
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs=SEARCH_KWARGS),
        return_source_documents=True
    )

def display_results(result: Dict[str, Any]) -> None:
    """Display the QA results in a user-friendly format."""
    st.markdown("### 📌 Answer")
    st.success(result.get("result", "I couldn't find an answer to your question."))
    
    with st.expander("🔍 See source documents"):
        for i, doc in enumerate(result.get("source_documents", []), 1):
            st.markdown(f"**Document {i}**")
            st.write(doc.page_content)
            st.write("---")

def main():
    """Main application function."""
    # Initialize Hugging Face
    setup_huggingface()
    
    # UI Setup
    st.title("📘 GHA SpecBot")
    st.caption("Ask questions from the Ghana Highway Authority Standard Specification (2007)")
    
    # Session state initialization
    if 'file_hash' not in st.session_state:
        st.session_state.file_hash = None
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    
    # PDF Processing
    pdf_file = st.file_uploader("📄 Upload GHA Specification PDF", type="pdf")
    
    if pdf_file:
        current_hash = get_file_hash(pdf_file.getvalue())
        
        # Only reprocess if the file has changed
        if current_hash != st.session_state.file_hash or st.session_state.vectordb is None:
            st.session_state.file_hash = current_hash
            
            text = extract_text_from_pdf(pdf_file)
            st.success("✅ PDF successfully loaded!")
            
            vectordb, message = process_document(text)
            st.session_state.vectordb = vectordb
            st.success(message)
    
    # QA System
    if st.session_state.get('vectordb'):
        user_input = st.text_input("💬 Ask a question:")
        if user_input:
            with st.spinner("🧠 Thinking..."):
                try:
                    qa_chain = initialize_qa_chain(st.session_state.vectordb)
                    result = qa_chain({"query": user_input})
                    display_results(result)
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")

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

if __name__ == "__main__":
    main()
