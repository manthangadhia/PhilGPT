import sys
import pathlib

# Add the project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and ensure all dependencies are loaded
try:
    from scripts.run_rag_pipeline import main as run_rag_pipeline
    from utils.retriever import Retriever
    from utils.io_utils import load_system_prompt
    from dotenv import load_dotenv
    
    # Load environment variables early
    load_dotenv()
    
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

import streamlit as st
import os

def main():
    st.title("PhilGPT")
    
    # Initialize components (with caching for efficiency)
    @st.cache_resource
    def load_retriever():
        return Retriever()
    
    @st.cache_data
    def load_prompt():
        return load_system_prompt()
    
    def get_previous_exchange():
        if len(st.session_state.messages) >= 2:
            last_user = st.session_state.messages[-2]
            last_assistant = st.session_state.messages[-1]
            
            if last_user["role"] == "user" and last_assistant["role"] == "assistant":
                last_exchange = f"""
                {last_user["content"]}
                {last_assistant["content"]}
                """
                return last_exchange
            
        return None

    # UI for the RAG pipeline
    st.write("This is a question-answer machine based on transcripts from "
    "['Philosophize This!'](https://www.philosophizethis.org/)")
    st.write("The answers generated here are based directly on information presented in the podcast. "
    "(In some of the answers, the model refers to this as its *context*.)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask PhilGPT your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # generate response through rag pipeline    
        with st.spinner("Thinking..."):
            try:
                retriever = load_retriever()
                system_prompt = load_prompt()

                # call rag pipeline
                response = run_rag_pipeline(prompt, retriever, system_prompt, 
                                            previous_query=get_previous_exchange(), 
                                            return_response=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
