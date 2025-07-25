import json
from tqdm import tqdm

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits the input text into chunks of specified size.
    
    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum size of each chunk in characters.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        text_chunk = text[i:i + chunk_size]
        if text_chunk:
            chunks.append(text_chunk)
    return chunks

if __name__ == "__main__":
    input_file = 'data/transcripts.jsonl'
    output_file = 'data/transcript_chunks.jsonl'

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'a', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Processing transcripts"):
            data = json.loads(line)
            chunks = chunk_text(data["transcript"])
            if not chunks:
                continue
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