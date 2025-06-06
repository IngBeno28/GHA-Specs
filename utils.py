# utils.py
import os
import logging
import hashlib
import time
from typing import Optional, Tuple, List, Dict, Any, Callable
from functools import wraps
import fitz  # PyMuPDF
from huggingface_hub import login, HfFolder
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles all document processing operations."""
    
    @staticmethod
    def get_file_hash(file_bytes: bytes) -> str:
        """Generate MD5 hash for file content."""
        return hashlib.md5(file_bytes).hexdigest()

    @staticmethod
    def validate_pdf(file_bytes: bytes, max_size_mb: int = 50) -> bool:
        """Validate PDF file size and format."""
        if not file_bytes:
            return False
            
        if len(file_bytes) / (1024 * 1024) > max_size_mb:
            return False
            
        return file_bytes[:5] == b'%PDF-'

    @staticmethod
    def extract_text(file_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise

class VectorDBManager:
    """Manages ChromaDB vector database operations."""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        
    def get_client(self) -> chromadb.PersistentClient:
        """Get ChromaDB client with retry logic."""
        os.makedirs(self.persist_dir, exist_ok=True)
        return chromadb.PersistentClient(path=self.persist_dir)
        
    def create_vector_store(self, 
                          text: str, 
                          collection_name: str,
                          chunk_size: int = 1000,
                          chunk_overlap: int = 200,
                          embedding_model: str = "all-MiniLM-L6-v2") -> Chroma:
        """Process text and create vector store."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.create_documents([text])
        
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        client = self.get_client()
        
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name=collection_name,
            persist_directory=self.persist_dir
        )
        vectordb.persist()
        return vectordb

class HuggingFaceHelper:
    """Handles Hugging Face operations."""
    
    @staticmethod
    def validate_token(token: str) -> bool:
        """Validate HF token format."""
        return token.startswith("hf_") if token else False
        
    @staticmethod
    def authenticate(token: str) -> bool:
        """Authenticate with Hugging Face Hub."""
        if not HuggingFaceHelper.validate_token(token):
            raise ValueError("Invalid token format")
            
        try:
            login(token=token, add_to_git_credential=False)
            HfFolder.save_token(token)
            return True
        except Exception as e:
            logger.error(f"HF authentication failed: {str(e)}")
            raise

def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """Decorator factory for retry logic."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt+1} failed, retrying...")
                    time.sleep(delay)
            raise RuntimeError("Max retries exceeded")
        return wrapper
    return decorator
