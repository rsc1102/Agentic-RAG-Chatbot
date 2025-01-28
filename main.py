import streamlit as st
from chat import stream_graph_updates
import uuid
from document_handler import document_parser, vectorize_document
from dotenv import load_dotenv

load_dotenv()


if "client_id" not in st.session_state:
    st.session_state.client_id = uuid.uuid4().hex    

st.title("🤖 Agentic RAG Chatbot 💬")
uploaded_file = st.file_uploader("Upload your files", type=("txt","pdf"))
if uploaded_file is not None:
    docs = document_parser(uploaded_file,st.session_state.client_id)
    vectorize_document(docs)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": 
            "How can I assist you today?"
            }
    ]
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Accept user input
if prompt := st.chat_input("Say something..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = stream_graph_updates(prompt,thread_id=st.session_state.client_id)
        response = st.write_stream(stream)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    
