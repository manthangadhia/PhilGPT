"""
The entire RAG pipeline goes as follows, divided into offline and online tasks done at different stages and frequencies:

** Offline **
1. Scraping (done periodically using something like cron) to gather transcript files and store their (meta)data.
2. Passing the scraped text data through `chunk_generator` to get hashable, vectorisable chunks.
3. Embed chunks and build vector index with hashing to avoid duplication.
4. A BM25 keyword index (built in memory from the chunks file at load time) complements the vector
   index, surfacing names/places/terms the embeddings might otherwise miss.

** Online **
5. Acquire user query.
6. Retrieve candidates with both vector search and BM25, fuse them with Reciprocal Rank Fusion,
   and reserve slots for any episodes the user explicitly names.
7. Format user query and retrieved information into system prompt and generate a response.
"""

import os
import sys
import pathlib

# Add the project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
CHROMA_PERSIST_DIR = project_root / "data" / "chroma"

from utils.retriever import Retriever
from utils.llm_utils import load_system_prompt, load_gemini_api_key
from utils.episode_embeddings import find_similar_episodes
from utils.query_parser import extract_episode_numbers
from google import genai

import argparse

def format_context(items):
    if not items:
        return ""

    sections = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not text:
            continue
        episode_number = item.get("episode_number")
        title = item.get("title")

        if episode_number is None:
            header = "Episode Unknown"
        else:
            header = f"Episode {episode_number}"
        if title:
            header = f"{header} - {title}"

        sections.append(f"{header}\n{text}")

    return "\n\n".join(sections)

def rerank_enabled_from_env():
    """True if the PHILGPT_RERANK environment variable asks for cross-encoder reranking."""
    return os.getenv("PHILGPT_RERANK", "").strip().lower() in {"1", "true", "yes", "on"}

def main(user_query, 
         retriever=None, 
         base_k=10,
         SYSTEM_PROMPT=None, 
         previous_query=None, 
         return_response=False,
         debug=False,
         rerank=None,
         ):
    if retriever is None:
        retriever = Retriever(persist_directory=CHROMA_PERSIST_DIR)
    if SYSTEM_PROMPT is None:
        SYSTEM_PROMPT = load_system_prompt()
    GEMINI_API_KEY = load_gemini_api_key()

    if rerank is None:
        # Off by default (the hosted app uses plain hybrid search for latency);
        # opt in locally with --rerank or PHILGPT_RERANK=1.
        rerank = rerank_enabled_from_env()

    epnums = extract_episode_numbers(user_query)
    k = base_k
    episode_filter = None
    if epnums:
        k = min(max(base_k, 8 * len(epnums)), 30)
        candidates = find_similar_episodes(epnums, top_k=20, include_original=True, exclude_self=True)
        if candidates:
            episode_filter = sorted(candidates)

    context_items = retriever.retrieve(
        user_query,
        k=k,
        episode_numbers=episode_filter,
        required_episodes=epnums or None,
        rerank=rerank,
    )
    context = format_context(context_items)

    if debug:
        print(f"[debug] episodes named: {epnums or '-'}; retrieved {len(context_items)} chunks:")
        for item in context_items:
            print(
                f"[debug]   ep {item.get('episode_number')!s:>4}  {item.get('chunk_id')}  "
                f"{item.get('sources'):<7}  {item.get('score', 0):.4f}  {item.get('title')}"
            )

    output = None
    if GEMINI_API_KEY:

        prompt = f"""
        {SYSTEM_PROMPT}

        **PREVIOUS QUERY AND RESPONSE**:
        {previous_query if previous_query else "N/A"}

        **USER QUERY**:
        {user_query}

        **CONTEXT**:
        {context if context else "N/A"}

        """

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        output = response.text
    else:
        output = "No LLM access is available (GEMINI_API_KEY is not set), so no answer could be generated."
    if return_response:
        return output
    else:
        # print(f"Your question: {user_query}")
        print("PhilGPT's response:")
        print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive PhilGPT session.")
    parser.add_argument("--debug", action="store_true", help="print retrieved chunks before each answer")
    parser.add_argument("--rerank", action="store_true",
                        help="rerank hybrid results with a cross-encoder (slower, ~+250 ms/query on CPU)")
    args = parser.parse_args()

    # Initialize retriever
    retriever = Retriever(persist_directory=CHROMA_PERSIST_DIR)
    base_k = 10  # number of chunks to retrieve
    # Load system prompt
    SYSTEM_PROMPT = load_system_prompt()

    # Start interactive session
    print("Welcome to PhilGPT! You can ask questions about philosophy.")
    quit_phrases = ["exit", "quit", "stop", "bye", "goodbye"]
    user_query = input("Ask your PhilGPT your question: ")

    while user_query.lower() not in quit_phrases:
        main(user_query, retriever, base_k, SYSTEM_PROMPT, debug=args.debug,
             rerank=args.rerank or rerank_enabled_from_env())
        user_query = input("Ask your PhilGPT your question: ")
    print("Goodbye! Thanks for using PhilGPT.")