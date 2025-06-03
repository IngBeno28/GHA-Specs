import streamlit as st
from huggingface_hub import login
# Access the token from secrets.toml
hf_token = st.secrets["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"]
# Authenticate with Hugging Face
login(token=hf_token)
# Now you can use HF models (e.g., download or inference)
st.success("✅ Successfully authenticated with Hugging Face!")
import fitz  # PyMuPDF
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from utils import export_answer_to_pdf
import os

# Set HuggingFace API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Styling
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("GHA Specification Chatbot (SpecBot)")
st.caption("Ask questions from the Ghana Highway Authority Specifications (2007 Edition)")

pdf_file = st.file_uploader("📄 Upload the GHA Spec PDF", type="pdf")
if pdf_file:
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    st.success("✅ PDF loaded successfully!")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    with st.spinner("⚙️ Creating document embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(chunks, embeddings)  # Removed persist_directory
        # vectordb.persist()  # Remove this line for Streamlit Cloud

    st.success("✅ Document indexed!")

    user_input = st.text_input("💬 Ask your question:")
    if user_input:
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=HuggingFaceHub(
                repo_id="google/flan-t5-large",  # Smaller alternative
                model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
            ),
            retriever=retriever
        )
        with st.spinner("🧠 Thinking..."):
            response = qa_chain.run(user_input)

        st.markdown("### ✅ Answer")
        st.success(response)

        if st.button("📤 Export Answer to PDF"):
            filepath = export_answer_to_pdf(user_input, response)
            with open(filepath, "rb") as f:
                st.download_button(
                    label="📥 Download PDF",
                    data=f,
                    file_name="SpecBot_QA.pdf",
                    mime="application/pdf"
                )
