# Welcome to PhilGPT
A question-answer machine to answer all your philosophy questions (as long as they are) based on the material of the 'PhilosophizeThis!' podcast hosted by Stephen West. 

This is a [RAG (Retrieval Augmented Generator)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) which relies on processed chunks of the text available in on the [philosophizethis.org](https://www.philosophizethis.org) website. **Disclaimer:** Stephen West provided consent for me to scrape the podcast transcripts.

## How to PhilGPT?

You have two options for interacting with this tool:
1. You can clone this repository, set your own Gemini-API key following `.env.example`, install the python environment either in a *venv* using the `requirements.txt`, or you can use *Pixi* as your package manager as I typically do and setup your minimal dependencies using the available `pixi.toml.example` into your local `pixi.toml` file. And finally, run `scripts/run_rag_pipeline.py`. ([For a quick guide to *Pixi*, I recommend these notes by GitHub user *willdumm*.](https://gist.github.com/willdumm/6b5063877f70157536111a92974535d1))
2. You can access a very [barebones version of PhilGPT, hosted for free on Streamlit](https://philgpt.streamlit.app/)

## How did I PhilGPT?

The overall RAG pipeline implemented here consists of the following pieces:
1. **Scraping** -- I scrape all the transcript texts off the website while tracking relevant metadata: `scripts/scrape.py`
2. **Chunking** -- All the text is chunked at the sentence level: `utils/chunk_generator.py`
    * Each chunk contains multiple neighbouring sentences as long as the total number of characters is *maximally* 1500. If adding a new sentence will go over this limit, that sentence is added to the new chunk.
    * Each chunk has one overlapping sentence with the previous chunk for continuity of context when chunks are retrieved later.
3. **Vectorisation** -- All these chunks of text are vectorised using the [all-MiniLM-L6-v1](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model running locally. This step is not carried out independently, but rather texts are vectorised *on the fly* while being added to the database.
4. **Indexing** -- I build a ChromaDB index storing all the embedding vectors for each chunk to make it searchable, and make sure each vector is linked with relevant metadata (episode number, url, episode title): `scripts/build_chroma_index.py`
5. **Querying** -- Given a user input, this input is encoded using the same embedding model and I conduct a search against the database to retrieve the 5 most similar (relevant) chunks. The text of these chunks is provided to a Gemini model as context along with intructions on the task to generate a coherent output via API: `scripts/run_rag_pipeline.py`
    * Functionality coming soon: Model cites the sources; Ask questions about specific episodes by mentioning them in your query;
6. (Bonus) **Scheduled updates** -- I setup a biweekly (once every two weeks) GitHub action to check for any new transcripts on the website, and to scrape, chunk, vectorise, and index them if so. The database hosted on GitHub will either be up-to-date, or maximally 2 weeks behind the actual latest released episode. 