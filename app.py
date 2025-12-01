import streamlit as st
import os
from src.qa_system import QASystem
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Document Q&A System", page_icon="📄")

st.title("Document Q&A System")
st.write("Upload a PDF to ask questions about it")

with st.expander("Example questions : "):
    st.write("- What is the document about?")
    st.write("- What are the main findings?")
    st.write("- Summarize the key points")

if "qa_system" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    st.session_state.qa_system = QASystem(api_key)
    st.session_state.document_loaded = False

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and st.button("Process document"):
    with st.spinner("Processing the document, this may take a minute ... "):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        api_key = os.getenv("GROQ_API_KEY")
        st.session_state.qa_system = QASystem(api_key)
        
        st.session_state.qa_system.load_document("temp.pdf")
        st.session_state.document_loaded = True
        
    st.success("Document loaded ✅")

if st.session_state.document_loaded:
    question = st.text_input("Ask a question about the document : ")

    if question:
        with st.spinner("Thinking ... "):
            result = st.session_state.qa_system.ask(question)
    
        st.write("Answer : ")
        st.write(result["answer"])

        st.write("Sources : ")
        for i, source in enumerate(result["sources"], 1):
            with st.expander(f"Source {i}"):
                st.write(source)

else:
    st.info("Please upload and process a document first")