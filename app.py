import sys
import os
# adding project root to python path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.qa_system import QASystem
from dotenv import load_dotenv
 
# loading environment variables
load_dotenv()

# configure page settings
st.set_page_config(page_title="Document Q&A System", page_icon="📄")

# main title
st.title("Document Q&A System")
st.write("Upload a PDF to ask questions about it")

# sidebar options
with st.sidebar:
    st.header("Options")

    # show current document info; if loaded 
    if st.session_state.get("document_loaded", False):
        st.info(f"Current document : {st.session_state.get("document_name", "Unknown")}")

        # clear conversation 
        if st.button("Clear conversations"):
            st.session_state.chat_history = []
            st.session_state.last_question = ""
            st.success("Conversation cleared")
            st.rerun()

# example questions
with st.expander("Example questions : "):
    st.write("- What is the document about?")
    st.write("- What are the main findings?")
    st.write("- Summarize the key points")

# session state initialization
if "qa_system" not in st.session_state:
    # getting API key
    try:
        api_key = st.secrets("GROQ_API_KEY")
    except:
        api_key = os.getenv("GROQ_API_KEY")

    # initializing qa sytsem
    st.session_state.qa_system = QASystem(api_key)
    st.session_state.document_loaded = False
    st.session_state.chat_history = []

# file upload
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# document processing
if uploaded_file and st.button("Process document"):
    try:
        with st.spinner("Processing the document, this may take a minute ... "):
            # save uploaded file to temp location
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # get api key again -> for  new qa system instance
            api_key = os.getenv("GROQ_API_KEY")
            st.session_state.qa_system = QASystem(api_key)

            # creating fresh qa system -> clears previous document
            st.session_state.chat_history = []
            st.session_state.document_name = uploaded_file.name
            
            # processing doc (extract -> chunks -> embed -> store)
            st.session_state.qa_system.load_document("temp.pdf")
            st.session_state.document_loaded = True
    
        st.success(f"Document loaded ✅ : {uploaded_file.name}")

    except Exception as e:
        st.error(f"❌ Error processing doc : {str(e)}")
        st.info("Pls make sure to upload a valid PDF file")
        st.session_state.document_loaded = False

# chat interface
if st.session_state.document_loaded:
    # display conversation history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            # user message bubble
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            # assistant message bubble
            with st.chat_message("assistant"):
                st.write(message["content"])
                # showing sources in expandable section
                if "sources" in message:
                    with st.expander("📄 View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"Source {i}")
                            # truncate long sentences
                            st.write(source[:300] + "..." if len(source) > 300 else source)
        st.write("---")
    
    # question input
    # st.chat_input auto clears after submit
    question = st.chat_input("Ask a question about the document")

    if question:
        try:
            with st.spinner("Thinking ... "):
                # get result + conversation history from qa system 
                result = st.session_state.qa_system.ask(question, chat_history=st.session_state.chat_history)

            # add user question to history
            st.session_state.chat_history.append(
                {
                    "role": "user", 
                    "content": question
                }
            )
            # add assistant answer to history
            st.session_state.chat_history.append(
                {
                    "role": "assistant", 
                    "content": result["answer"], 
                    "sources": result["sources"]
                }
            )     
            # force rerun to display new messages       
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error getting answer : {str(e)}")
else:
    st.info("Please upload and process a document first")