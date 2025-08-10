import sys
import pathlib

# Add the project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.retriever import Retriever
from utils.io_utils import load_system_prompt
from google import genai

from dotenv import load_dotenv
import os

# load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def main(user_query, retriever=None, SYSTEM_PROMPT=None):
    if retriever is None:
        retriever = Retriever()
    if SYSTEM_PROMPT is None:
        SYSTEM_PROMPT = load_system_prompt()

    context = retriever.retrieve(user_query, k=8)

    # first try with gemini api
    if GEMINI_API_KEY:
        print("Using Gemini API for response generation...")

        prompt = f"""
        {SYSTEM_PROMPT}

        **CONTEXT**:
        {context}

        **USER QUERY**:
        {user_query}
        """

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        output = response.text

    else: # if no gemini api key is available, default to loading and using local ollama model
        import ollama    
        messages=[
            {'role': "system", 'content': SYSTEM_PROMPT},
            {'role': "assistant", 'content': context},
            {'role': "user", 'content': user_query}
        ]
        response = ollama.chat(
            model="gemma3:4b",
            messages=messages
        )
        output = response['message']['content']

    # print(f"Your question: {user_query}")
    print("PhilGPT's response:")
    print(output)

if __name__ == "__main__":
    # Initialize retriever
    import time
    start = time.time()
    retriever = Retriever()
    end = time.time()
    print(f"Retriever total initialization time: {end - start} seconds")

    # Load system prompt
    SYSTEM_PROMPT = load_system_prompt()

    # Start interactive session
    print("Welcome to PhilGPT! You can ask questions about philosophy.")
    quit_phrases = ["exit", "quit", "stop", "bye", "goodbye"]

    user_query = input("Ask your PhilGPT your question: ")
    while user_query.lower() not in quit_phrases:
        main(user_query, retriever, SYSTEM_PROMPT)
        user_query = input("Ask your PhilGPT your question: ")
    print("Goodbye! Thanks for using PhilGPT.")