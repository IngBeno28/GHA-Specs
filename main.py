import os
import sys
import logging
import hashlib
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import streamlit as st
from huggingface_hub import login, HfFolder
import pysqlite3
sys.modules["pysqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for all application settings."""
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "google/flan-t5-large"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEARCH_KWARGS: dict = None
    LLM_TEMPERATURE: float = 0.3
    MAX_NEW_TOKENS: int = 512
    PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "gha_specs"
    MAX_FILE_SIZE_MB: int = 50
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    def __post_init__(self):
        if self.SEARCH_KWARGS is None:
            self.SEARCH_KWARGS = {"k": 3}

# Initialize configuration
config = Config()

# Display package versions for debugging
st.sidebar.markdown("### Environment Info")
st.sidebar.text(f"Python: {sys.version.split()[0]}")
st.sidebar.text(f"SQLite: {pysqlite3.sqlite_version}")

def retry_on_failure(max_retries: int = config.MAX_RETRIES, delay: float = config.RETRY_DELAY):
    """Decorator to retry functions on failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def setup_huggingface() -> bool:
    """Initialize Hugging Face authentication with enhanced error handling and retry logic."""
    try:
        hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        
        if not hf_token:
            st.error("🔑 Hugging Face token not found. Please configure it in secrets.toml or environment variables.")
            logger.error("HF token not found in secrets or environment variables")
            st.stop()
            
        if not hf_token.startswith("hf_"):
            st.error("❌ Invalid token format. Hugging Face tokens should start with 'hf_'")
            logger.error("Invalid HF token format")
            st.stop()
            
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        
        # Enhanced error handling for specific exceptions
        try:
            login(token=hf_token, add_to_git_credential=False)
            HfFolder.save_token(hf_token)
            logger.info("Successfully authenticated with Hugging Face")
            return True
        except ValueError as e:
            st.error(f"❌ Invalid token: {str(e)}")
            logger.error(f"Invalid HF token: {str(e)}")
        except ConnectionError as e:
            st.error(f"🌐 Network error during authentication: {str(e)}")
            logger.error(f"Network error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Authentication error: {str(e)}")
            logger.error(f"Authentication error: {str(e)}")
            
        st.stop()
            
    except Exception as e:
        st.error(f"❌ Unexpected error during setup: {str(e)}")
        logger.error(f"Unexpected setup error: {str(e)}")
        st.stop()

@retry_on_failure()
def get_chroma_client() -> chromadb.PersistentClient:
    """Initialize and return a ChromaDB client with error handling and retry logic."""
    try:
        os.makedirs(config.PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=config.PERSIST_DIR)
        logger.info("Successfully initialized ChromaDB client")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}")
        st.error(f"❌ Failed to initialize ChromaDB: {str(e)}")
        st.error("💡 Please ensure SQLite >= 3.35.0 is installed or try Chroma's cloud version.")
        raise e

def get_file_hash(file_bytes: bytes) -> str:
    """Generate a hash for the uploaded file to detect changes."""
    return hashlib.md5(file_bytes).hexdigest()

def validate_pdf_file(pdf_file) -> bool:
    """Validate PDF file size and format."""
    if pdf_file is None:
        return False
        
    # Check file size
    file_size_mb = len(pdf_file.getvalue()) / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB, your file: {file_size_mb:.1f}MB")
        return False
        
    # Check if it's actually a PDF
    try:
        pdf_header = pdf_file.getvalue()[:5]
        if pdf_header != b'%PDF-':
            st.error("❌ Invalid PDF file format")
            return False
    except Exception:
        st.error("❌ Could not validate PDF file")
        return False
        
    return True

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes, file_hash: str) -> str:
    """Extract text from PDF with proper error handling and resource cleanup."""
    doc = None
    try:
        with st.spinner("📖 Reading PDF..."):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_chunks = []
            
            # Add progress bar for large documents
            total_pages = len(doc)
            progress_bar = st.progress(0)
            
            for i, page in enumerate(doc):
                text_chunks.append(page.get_text())
                # Update progress
                progress_bar.progress((i + 1) / total_pages)
            
            progress_bar.empty()
            full_text = "\n".join(text_chunks)
            
            # Basic content validation
            if len(full_text.strip()) < 100:
                st.warning("⚠️ PDF appears to contain very little text. It might be image-based.")
                
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        st.error(f"❌ PDF processing error: {str(e)}")
        st.error("💡 Please ensure the PDF is not corrupted and contains readable text.")
        st.stop()
    finally:
        # Ensure document is properly closed
        if doc:
            doc.close()

@st.cache_resource
def get_embeddings_model():
    """Get embeddings model with caching."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        logger.info(f"Successfully loaded embeddings model: {config.EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {str(e)}")
        st.error(f"❌ Failed to load embeddings model: {str(e)}")
        st.stop()

def process_document(text: str) -> Tuple[Chroma, str]:
    """Process and index the document text with enhanced progress tracking."""
    try:
        # Document splitting
        with st.spinner("✂️ Splitting document into chunks..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            chunks = splitter.create_documents([text])
            st.info(f"📄 Created {len(chunks)} text chunks")
            logger.info(f"Document split into {len(chunks)} chunks")
        
        # Embedding and vector store creation
        with st.spinner("🔍 Creating embeddings and building vector store..."):
            embeddings = get_embeddings_model()
            chroma_client = get_chroma_client()
            
            # Progress tracking for embedding
            progress_bar = st.progress(0)
            st.text("Processing chunks...")
            
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                client=chroma_client,
                collection_name=config.COLLECTION_NAME,
                persist_directory=config.PERSIST_DIR
            )
            vectordb.persist()
            progress_bar.progress(1.0)
            progress_bar.empty()
            
        logger.info("Successfully created and persisted vector database")
        return vectordb, f"✅ Document indexed successfully! {len(chunks)} chunks processed."
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        st.error(f"❌ Document processing error: {str(e)}")
        st.error("💡 Please try with a smaller document or check your internet connection.")
        st.stop()

@st.cache_resource
def get_llm_model():
    """Get LLM model with caching."""
    try:
        llm = HuggingFaceHub(
            repo_id=config.LLM_MODEL,
            model_kwargs={
                "temperature": config.LLM_TEMPERATURE,
                "max_new_tokens": config.MAX_NEW_TOKENS
            }
        )
        logger.info(f"Successfully loaded LLM model: {config.LLM_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLM model: {str(e)}")
        st.error(f"❌ Failed to load LLM model: {str(e)}")
        st.stop()

def initialize_qa_chain(vectordb: Chroma) -> RetrievalQA:
    """Initialize the QA chain with proper configuration and caching."""
    try:
        llm = get_llm_model()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs=config.SEARCH_KWARGS),
            return_source_documents=True
        )
        logger.info("Successfully initialized QA chain")
        return qa_chain
    except Exception as e:
        logger.error(f"Failed to initialize QA chain: {str(e)}")
        st.error(f"❌ Failed to initialize QA system: {str(e)}")
        st.stop()

def display_results(result: Dict[str, Any]) -> None:
    """Display the QA results in a user-friendly format with enhanced formatting."""
    st.markdown("### 📌 Answer")
    
    answer = result.get("result", "I couldn't find an answer to your question.")
    if answer and answer.strip():
        st.success(answer)
    else:
        st.warning("⚠️ No relevant answer found in the document.")
    
    # Enhanced source document display
    source_docs = result.get("source_documents", [])
    if source_docs:
        with st.expander(f"🔍 View {len(source_docs)} source document(s)"):
            for i, doc in enumerate(source_docs, 1):
                st.markdown(f"**📄 Source Document {i}**")
                
                # Truncate very long content
                content = doc.page_content
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                    
                st.text_area(
                    f"Content {i}:",
                    content,
                    height=100,
                    key=f"source_{i}",
                    disabled=True
                )
                
                if i < len(source_docs):
                    st.divider()
    else:
        st.info("ℹ️ No source documents available.")

def display_safe_footer() -> None:
    """Display footer using safe HTML rendering."""
    try:
        # Using Streamlit's native components for safety
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("🤖 GHA SpecBot v2.0")
            
        with col2:
            st.caption("📧 wiafe1713@gmail.com")
            
        with col3:
            st.caption(f"🐍 Python {sys.version.split()[0]}")
            
        st.caption("Powered by LangChain, ChromaDB and HuggingFace | © 2007 Ghana Highway Authority")
        
    except Exception as e:
        logger.warning(f"Footer display error: {str(e)}")
        st.caption("GHA SpecBot - For technical support, please contact the administrator.")

def main() -> None:
    """Main application function with enhanced error handling."""
    try:
        # Initialize Hugging Face
        setup_huggingface()
        
        # UI Setup
        st.title("📘 GHA SpecBot")
        st.caption("Ask questions from the Ghana Highway Authority Standard Specification (2007)")
        
        # Add usage instructions
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Upload a PDF**: Click 'Browse files' and select your GHA specification PDF
            2. **Wait for processing**: The app will extract and index the document content
            3. **Ask questions**: Type your question in the text box and press Enter
            4. **Review results**: Get answers with source document references
            
            **Tips:**
            - Use specific questions for better results
            - File size limit: 50MB
            - Processing time depends on document size
            """)
        
        # Session state initialization
        if 'file_hash' not in st.session_state:
            st.session_state.file_hash = None
        if 'vectordb' not in st.session_state:
            st.session_state.vectordb = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        
        # PDF Processing
        st.markdown("### 📄 Document Upload")
        pdf_file = st.file_uploader(
            "Upload GHA Specification PDF", 
            type="pdf",
            help=f"Maximum file size: {config.MAX_FILE_SIZE_MB}MB"
        )
        
        if pdf_file and validate_pdf_file(pdf_file):
            current_hash = get_file_hash(pdf_file.getvalue())
            
            # Only reprocess if the file has changed
            if (current_hash != st.session_state.file_hash or 
                st.session_state.vectordb is None or
                not st.session_state.processing_complete):
                
                st.session_state.file_hash = current_hash
                st.session_state.processing_complete = False
                
                # Extract text with progress tracking
                text = extract_text_from_pdf(pdf_file.getvalue(), current_hash)
                st.success("✅ PDF successfully loaded!")
                
                # Process document
                vectordb, message = process_document(text)
                st.session_state.vectordb = vectordb
                st.session_state.processing_complete = True
                st.success(message)
                
                # Display document stats
                st.info(f"📊 Document contains {len(text):,} characters")
        
        # QA System
        if st.session_state.get('vectordb') and st.session_state.get('processing_complete'):
            st.markdown("### 💬 Ask Questions")
            
            # Question input with example
            user_input = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the requirements for concrete mixing?",
                help="Ask specific questions about the uploaded document for best results"
            )
            
            if user_input.strip():
                with st.spinner("🧠 Analyzing document and generating answer..."):
                    try:
                        qa_chain = initialize_qa_chain(st.session_state.vectordb)
                        result = qa_chain({"query": user_input})
                        display_results(result)
                        logger.info(f"Successfully processed query: {user_input[:50]}...")
                    except Exception as e:
                        logger.error(f"Error processing query: {str(e)}")
                        st.error(f"❌ Error processing your question: {str(e)}")
                        st.error("💡 Please try rephrasing your question or check your internet connection.")
        
        elif pdf_file:
            st.info("⏳ Please wait for document processing to complete before asking questions.")
        else:
            st.info("📤 Please upload a PDF document to get started.")
        
        # Enhanced Footer
        display_safe_footer()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"❌ Application error: {str(e)}")
        st.error("💡 Please refresh the page and try again.")

if __name__ == "__main__":
    main()
