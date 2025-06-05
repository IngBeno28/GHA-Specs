import os
import streamlit as st
from huggingface_hub import login, HfFolder
import Chromadb
client = chromadb.HttpClient(host="localhost", port="8000")
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

# Improved Hugging Face authentication
def setup_huggingface():
    try:
        # Try getting token from secrets first
        hf_token = st.secrets.get("HF_TOKEN")
        
        if not hf_token:
            # Fallback to environment variable
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            
        if not hf_token:
            st.error("Hugging Face token not found. Please configure it in secrets.toml or environment variables.")
            st.stop()
            
        # Validate token format
        if not hf_token.startswith("hf_"):
            st.error("Invalid token format. Hugging Face tokens should start with 'hf_'")
            st.stop()
            
        # Set environment variable
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        
        # Try login with validation
        try:
            login(token=hf_token, add_to_git_credential=False)
            HfFolder.save_token(hf_token)
            return True
        except Exception as e:
            st.error(f"Failed to authenticate with Hugging Face: {str(e)}")
            st.error("Please check your token is valid and has the correct permissions.")
            st.stop()
            
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.stop()

# Initialize Hugging Face
setup_huggingface()

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
                    
                    # Safely access result dictionary
                    response = result.get("result", "I couldn't find an answer to your question.")
                    
                    st.markdown("### 📌 Answer")
                    st.success(response)
                    
                    # Show source documents if available
                    with st.expander("🔍 See source documents"):
                        for doc in result.get("source_documents", []):
                            st.write(doc.page_content)
                            st.write("---")
                            
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")
                    st.error("Please try a different question or check your PDF content.")
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please ensure you have the correct dependencies installed.")

# Footer with simplified styling and better error handling
st.markdown("---")
try:
    footer = """
    <div style="
        font-size: 0.8rem;
        color: #6c757d;
        text-align: center;
        padding: 10px;
        margin-top: 30px;
    ">
        <p>GHA SpecBot v1.0 | © 2007 Ghana Highway Authority</p>
        <p>Powered by LangChain and HuggingFace | Python {version}</p>
        <p>For support contact: wiafe1713@gmail.com</p>
    </div>
    """.format(version=sys.version.split()[0])
    st.markdown(footer, unsafe_allow_html=True)
except Exception as e:
    st.warning("Footer could not be displayed properly")
