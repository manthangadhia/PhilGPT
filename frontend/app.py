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
    st.title("PhilGPT - Philosophy Q&A")
    
    # Initialize components (with caching for efficiency)
    @st.cache_resource
    def load_retriever():
        return Retriever()
    
    @st.cache_data
    def load_prompt():
        return load_system_prompt()
    
    # UI for the RAG pipeline
    st.write("Ask questions about philosophy and get informed responses!")
    
    user_query = st.text_input("Enter your philosophy question:")
    
    if user_query:
        with st.spinner("Thinking..."):
            try:
                retriever = load_retriever()
                system_prompt = load_prompt()
                
                # Call the RAG pipeline
                response = run_rag_pipeline(user_query, retriever, system_prompt, return_response=True)
                
                st.write("**PhilGPT's Response:**")
                st.write(response)
                
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
