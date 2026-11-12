# Welcome to PhilGPT
A question-answer machine to answer all your philosophy questions (as long as they are) based on the material of the 'PhilosophizeThis!' podcast hosted by Stephen West. 

This is a [RAG (Retrieval Augmented Generator)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) which relies on processed chunks of the text available in on the [philosophizethis.org](https://www.philosophizethis.org) website. **Disclaimer:** Stephen West provided consent for me to scrape the podcast transcripts.

## How to PhilGPT?

You have two options for interacting with this tool:
1. You can access a very [barebones version of PhilGPT, hosted for free on Streamlit](https://philgpt.streamlit.app/) for a browser-based chat interface! (Easiest way to go about this, but beware that things might break if there are multiple users at once, this is not a scalable software-engineering project.)
2. You can clone the project [GitHub repository](https://github.com/manthangadhia/PhilGPT), set your own Gemini-API key following `.env.example`, install the python environment either in a *venv* using the `requirements.txt`, or you can use *Pixi* as your package manager as I typically do and setup your minimal dependencies using the available `pixi.toml.example` into your local `pixi.toml` file. And finally, run `scripts/run_rag_pipeline.py` (or `streamlit run frontend/app.py` for the chat interface). Running locally also lets you switch on an optional reranking step (see *Reranking* below). ([For a quick guide to *Pixi*, I recommend these notes by GitHub user *willdumm*.](https://gist.github.com/willdumm/6b5063877f70157536111a92974535d1)).

## How did I PhilGPT?

The overall RAG pipeline implemented here consists of the following pieces:
1. **Scraping** -- I scrape all the transcript texts off the website while tracking relevant metadata: `scripts/scrape.py`
2. **Chunking** -- All the text is chunked at the sentence level: `utils/chunk_generator.py`
    * Each chunk contains multiple neighbouring sentences as long as the total number of characters is *maximally* 1500. If adding a new sentence will go over this limit, that sentence is added to the new chunk.
    * Each chunk has one overlapping sentence with the previous chunk for continuity of context when chunks are retrieved later.
3. **Vectorisation** -- All these chunks of text are vectorised using the [all-MiniLM-L6-v1](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model running locally. This step is not carried out independently, but rather texts are vectorised *on the fly* while being added to the database.
4. **Indexing** -- I build a ChromaDB index storing all the embedding vectors for each chunk to make it searchable, and make sure each vector is linked with relevant metadata (episode number, url, episode title): `scripts/build_chroma_index.py`
5. **Querying (hybrid retrieval)** -- Given a user input, I retrieve relevant chunks in two ways and combine them: `utils/retriever.py`
    * *Dense search*: the query is encoded with the same embedding model and matched against the ChromaDB index (good for paraphrased, conceptual questions).
    * *Keyword search*: a BM25 index, built in memory from `data/transcript_chunks.jsonl` when the app starts, scores chunks by exact (stemmed, accent-insensitive) term overlap (good for names and specific terms like "Žižek" or "categorical imperative"): `utils/bm25_index.py`
    * The two rankings are merged with *Reciprocal Rank Fusion*. If the query names episodes ("ep 120", "episodes 98-99"), those references are stripped from the search text and a majority of the result slots are reserved for the named episodes; a query like "summarise episode 120" gets chunks spread evenly across that episode.
    * *Reranking (optional, off by default)*: the top 30 fused candidates can be re-sorted by a cross-encoder ([ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)), which reads the query and each chunk together to judge relevance: `utils/reranker.py`. The hosted Streamlit app uses plain hybrid search to keep answers fast. When running locally you can turn reranking on in either of two ways:
        * pass `--rerank` to the terminal version: `python scripts/run_rag_pipeline.py --rerank`
        * set `PHILGPT_RERANK=1` in your `.env` (or environment); this also applies to `streamlit run frontend/app.py`
        * the model (~90 MB) is downloaded on first use into `data/rerankers/` and adds roughly 250 ms per question on a CPU
    * The text of the retrieved chunks is provided to a Gemini model as context along with instructions on the task to generate a coherent output via API: `scripts/run_rag_pipeline.py` (add `--debug` to see which chunks were retrieved).
6. (Bonus) **Scheduled updates** -- I setup a biweekly (once every two weeks) GitHub action to check for any new transcripts on the website, and to scrape, chunk, vectorise, and index them if so. The database hosted on GitHub will either be up-to-date, or maximally 2 weeks behind the actual latest released episode. 

#### Evaluation
`python scripts/eval_retrieval.py` compares three retrieval stages on 100 labelled queries (`data/eval_queries.jsonl`): **dense** only, **hybrid** (dense + BM25 fused with RRF), and **hybrid + rerank**. Relevance is episode-level: a retrieved chunk counts if it comes from an episode labelled for that query. The query groups are names/terms (45), concepts (11), paraphrases that avoid any names (34), and queries naming episodes (10).

| stage | MRR (all, n=90) | Hit@5 | P@10 | MRR (paraphrase) | MRR (names) | named-episode share | latency |
|---|---|---|---|---|---|---|---|
| dense | 0.807 | 0.911 | 0.413 | 0.663 | **0.913** | 0.829 | 13 ms |
| hybrid (RRF) | 0.837 | 0.956 | 0.438 | 0.750 | 0.903 | 0.855 | 10 ms |
| hybrid + rerank | **0.842** | **0.967** | **0.443** | **0.826** | 0.860 | **0.874** | 254 ms |

What each stage adds: BM25 fusion mostly helps paraphrased and conceptual questions (paraphrase MRR +0.09) while staying level on name queries. The reranker helps paraphrases further (+0.08), but on name queries it sometimes pushes a passing mention from another episode above the main episode (MRR -0.04). Net, it is a small overall gain at about a quarter of a second per query, so it is offered as a local opt-in rather than used in the hosted app. The larger `BAAI/bge-reranker-base` scored the same overall (MRR 0.841) at about 3 s per query, so the small model stays the default (`--reranker` to compare others). P@10 is capped at 0.4 for single-episode queries, because at most 4 chunks per episode are returned. The query set is small and hand-written, so treat the numbers as indicative.

### PhilGPT use cases
I have to say I worked on this project primarily as an opportunity to learn about and implement some of the information processing techniques that LLMs can be used for. I have found it fun (but unreliable) when asking questions about specific episodes (*mentioning "ep xx" or "episodes xx-xx"*) to draw connections or find insights. I'm also happy about the reliable citation behaviour (implemented through tracking metadata during retrieval) as this can let me know specific topics that were discussed in specific episodes I may want to go (back) to!


### Limitations
There are many limitations to this system when compared to large-scale, consumer-grade chatbots:
1. *No memory, oops.* I mean there is a little memory, since I pass the previous prompt-response pair with each new prompt, so the LLM knows what the very previous question and answer was. 
2. This a question-answer machine *rather than* a philosophical conversationalist--this was my original aim to stay within scope. 

**Note:** For a more personal reflection on the usage of a chat-like interface for open-ended search read: #TODO: add personal essay