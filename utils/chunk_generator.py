"""
In the first iteration, this chunking script was generating chunks in terms of character counts,
and this did not make a lot of sense to me in hindsight since words would sometimes get split
across chunks and even the overlap was not sufficient to capture the context.

I have now adapted the script to chunk sentences up to a maximum character length, and for the
overlap to be 1-sentence.
"""

import json
from tqdm import tqdm

import pathlib
root_dir = pathlib.Path(__file__).parent.parent
data_dir = root_dir / 'data'

import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

def chunk_text(text, chunk_size=1500, overlap=1):
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap:]  # keep last N sentences as overlap
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks

def load_existing_chunk_urls(chunks_file: pathlib.Path) -> set:
    if not chunks_file.exists():
        return set()
    existing_urls = set()
    with chunks_file.open('r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = data.get("url")
            if url:
                existing_urls.add(url)
    return existing_urls

if __name__ == "__main__":
    input_file = data_dir / 'transcripts.jsonl'
    output_file = data_dir / 'transcript_chunks.jsonl'

    existing_urls = load_existing_chunk_urls(output_file)
    total_transcripts = 0
    skipped_transcripts = 0
    new_transcripts = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'a', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Processing transcripts"):
            data = json.loads(line)
            total_transcripts += 1
            url = data.get("url")
            if url and url in existing_urls:
                skipped_transcripts += 1
                continue
            chunks = chunk_text(data["transcript"])
            if not chunks:
                continue
            new_transcripts += 1
            for i, chunk in enumerate(chunks):
                chunked_item = {
                    "episode_number": data["episode_number"],
                    "chunk_id": f"ep{data['episode_number']}_chunk{i:03}",
                    "text": chunk,
                    "title": data["title"],
                    "url": data["url"]
                }
                json.dump(chunked_item, outfile)
                outfile.write("\n")

    print(f"Total transcripts: {total_transcripts}")
    print(f"Skipped (already chunked): {skipped_transcripts}")
    print(f"New transcripts chunked: {new_transcripts}")