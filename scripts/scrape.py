import requests
import time
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT_DIR))
CHROMA_PERSIST_DIR = DATA_DIR / "chroma"

from utils.chroma_store import get_existing_episode_urls

ROOT_URL = "https://www.philosophizethis.org"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 5  # seconds
REQUEST_DELAY = 2  # seconds

def fetch_response_with_retry(url, max_retries, timeout):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # raise an error for HTTP errors
            return response
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed to fetch {url}: {e}")
            if attempt == (max_retries - 1):
                print(f"❌ Giving up on {url}")
                # Log failed url in a txt
                with open("failed_urls.txt", "a") as f:
                    f.write(url + "\n")
                return None
        time.sleep(REQUEST_DELAY)

def extract_text_and_metadata(full_transcript_links,
                              attempts=MAX_RETRIES,
                              timeout=REQUEST_TIMEOUT,
                              delay=REQUEST_DELAY,
                              output_file=DATA_DIR / "transcripts.jsonl",
                              append_mode: bool = False,
                              retry_failed: bool = True):

    def get_episode_number(title):
        # Extract the episode number from the title using regex
        import re
        match = re.findall(r'\d+', title)
        return int(match[0]) if match else None

    mode = "a" if append_mode else "w"
    failed_list = []

    with open(output_file, mode, encoding="utf-8") as f:
        for url in tqdm(full_transcript_links, desc="Scraping transcripts", unit="transcript"):

            # Add delay before each request to avoid overwhelming the server
            time.sleep(delay)

            response = fetch_response_with_retry(url, attempts, timeout)
            if response is None:
                failed_list.append(url)
                print("Failed to fetch response, stored in failed_urls for retry.")
                continue
            soup = BeautifulSoup(response.text, 'lxml')
            # Narrow down the div type with all the text and specifically get the paragraphs
            title_container = soup.find("div", class_="blog-item-title")
            title = "NONE" if not title_container else title_container.find_all("h1")[0].get_text()

            html_container = soup.find("div", class_="sqs-html-content")
            paragraphs = html_container.find_all("p") if html_container else []

            # Strip all paragraphs for only the text in them and none of the other addons
            transcript_lines = [p.get_text() for p in paragraphs]  
            transcript = " ".join(transcript_lines)

            # Write the data to a JSONL file
            json.dump({
                "url": url,
                "title": title,
                "episode_number": get_episode_number(title),
                "transcript": transcript
            }, f, ensure_ascii=False)
            f.write("\n")

    # If some URLs failed and retry is enabled, re-run on just those URLs once (append to same file)
    if retry_failed and failed_list:
        print(f"Retrying {len(failed_list)} failed URLs...")
        # call function again in append mode and disable further retries to avoid loops
        extract_text_and_metadata(failed_list, attempts=attempts, timeout=timeout, delay=delay,
                                  output_file=output_file, append_mode=True, retry_failed=False)

def get_links_to_transcripts(url=ROOT_URL, max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT):
    # First access page with all the links to individual transcripts
    TRANSCRIPT_URL = url + "/transcripts" # direct request to transcript page specifically

    response = fetch_response_with_retry(TRANSCRIPT_URL, max_retries, timeout)
    soup = BeautifulSoup(response.text, 'lxml')
            
    # Next get all links to pages w/ 'transcript' in the href and avoid all other links
    links_to_transcripts = [tag["href"] for tag in soup.find_all("a", href=True) if ('/transcript/' in tag["href"])]
    full_links = [ROOT_URL + link for link in links_to_transcripts]

    return full_links

if __name__ == "__main__":
    # Get all links to transcripts
    full_links = get_links_to_transcripts()
    try:
        existing_urls = get_existing_episode_urls(persist_directory=CHROMA_PERSIST_DIR)
    except Exception as e:
        print(f"Warning: failed to read existing Chroma URLs ({e}). Proceeding with all links.")
        existing_urls = set()

    new_links = [link for link in full_links if link not in existing_urls]
    print(f"Total transcript links: {len(full_links)}")
    print(f"Already indexed links: {len(existing_urls)}")
    print(f"New links to scrape: {len(new_links)}")

    if not new_links:
        print("No new transcripts found. Nothing to scrape.")
    else:
        extract_text_and_metadata(new_links, append_mode=True)