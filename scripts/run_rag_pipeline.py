import sys
import pathlib

# Add the project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ollama
from utils.retriever import Retriever
from utils.io_utils import load_system_prompt

def main(user_query, retriever=None, SYSTEM_PROMPT=None):
    if retriever is None:
        retriever = Retriever()

    if SYSTEM_PROMPT is None:
        SYSTEM_PROMPT = load_system_prompt()

    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {'role': "system", 'content': SYSTEM_PROMPT},
            {'role': "assistant", 'content': Retriever().retrieve(user_query)},
            {'role': "user", 'content': user_query}
        ])

    print(f"Your question: {user_query}")
    print("PhilGPT's response:")
    print(response['message']['content'])

if __name__ == "__main__":
    # Initialize retriever
    retriever = Retriever()
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