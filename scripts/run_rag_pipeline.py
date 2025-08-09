import sys
import pathlib

# Add the project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ollama
from utils.retriever import Retriever

SYSTEM_PROMPT = """
You are a knowledgeable assistant who is interested in philosophy and about teaching 
others about philosophy. When asked questions, you answer them in a concise and informative manner,
drawing from the provided context. If the context does not contain relevant information, you will say
"I don't know" or "I don't have enough information to answer that question."
"""

def main(user_query, retriever=None):
    if retriever is None:
        retriever = Retriever()

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

    # Start interactive session
    print("Welcome to PhilGPT! You can ask questions about philosophy.")
    quit_phrases = ["exit", "quit", "stop", "bye", "goodbye"]

    user_query = input("Ask your PhilGPT your question: ")
    while user_query.lower() not in quit_phrases:
        main(user_query, retriever)
        user_query = input("Ask your PhilGPT your question: ")
    print("Goodbye! Thanks for using PhilGPT.")