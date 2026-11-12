"""
In-memory BM25 (Okapi) keyword index over the transcript chunks.

The index is built from `data/transcript_chunks.jsonl` (the same file the Chroma
index is built from, and whose `chunk_id`s are the Chroma document ids), so the
sparse and dense results can be fused by id. With ~6k chunks the build takes a
couple of seconds, so nothing is persisted; the index is cached per process.

Implementation uses a scipy sparse term-document matrix with precomputed BM25
weights, so a query is a sum of a few sparse rows.
"""
import json
import pathlib
import re
import threading
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse

from .path_config import DATA_DIR

CHUNKS_PATH = DATA_DIR / "transcript_chunks.jsonl"
# Title tokens are counted this many times to boost title matches. The scraped titles are
# currently just "Episode #N - Transcript" (no topic), so they're left out for now; raise
# this if the scraper starts capturing real episode titles.
TITLE_REPEAT = 0

# Compact English stopword list (kept inline so no corpus download is needed).
# Note: "will" is deliberately NOT a stopword ("free will", "will to power").
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no nor
    not now of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where
    which while who whom why with would you your yours yourself yourselves
    also like really get got go going gonna know thing things think say said says
    one two us let well yeah okay ok kind sort lot way even much many
    episode episodes ep eps podcast philosophize today
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CHUNK_INDEX_RE = re.compile(r"_chunk(\d+)$")

try:
    from nltk.stem.snowball import SnowballStemmer

    _stem = SnowballStemmer("english").stem
except Exception:  # pragma: no cover - nltk is a dependency, but degrade gracefully
    def _stem(token: str) -> str:
        return token

_STEM_CACHE: Dict[str, str] = {}


def _fold(text: str) -> str:
    """Lowercase and strip diacritics (so 'Zizek' matches 'Žižek')."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().replace("’", "'")


def tokenize(text: str) -> List[str]:
    tokens = []
    for tok in _TOKEN_RE.findall(_fold(text or "")):
        if tok in STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        stemmed = _STEM_CACHE.get(tok)
        if stemmed is None:
            stemmed = _stem(tok)
            _STEM_CACHE[tok] = stemmed
        tokens.append(stemmed)
    return tokens


def _chunk_index(chunk_id: str) -> int:
    match = _CHUNK_INDEX_RE.search(chunk_id or "")
    return int(match.group(1)) if match else 0


class BM25Index:
    def __init__(self, chunks: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict] = []
        self.id_to_row: Dict[str, int] = {}
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            if not chunk_id or not isinstance(text, str) or chunk_id in self.id_to_row:
                continue
            self.id_to_row[chunk_id] = len(self.chunks)
            self.chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "title": chunk.get("title"),
                    "episode_number": chunk.get("episode_number"),
                    "url": chunk.get("url"),
                    "chunk_index": _chunk_index(chunk_id),
                }
            )
        self.episode_numbers = np.array(
            [c["episode_number"] if c["episode_number"] is not None else -1 for c in self.chunks],
            dtype=np.int64,
        )
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        vocab: Dict[str, int] = {}
        rows, cols, vals = [], [], []
        doc_lens = np.zeros(len(self.chunks), dtype=np.float32)

        for row, chunk in enumerate(self.chunks):
            title_tokens = tokenize(chunk["title"] or "") * TITLE_REPEAT
            tokens = title_tokens + tokenize(chunk["text"])
            doc_lens[row] = len(tokens)
            counts: Dict[int, int] = {}
            for tok in tokens:
                col = vocab.setdefault(tok, len(vocab))
                counts[col] = counts.get(col, 0) + 1
            for col, tf in counts.items():
                rows.append(row)
                cols.append(col)
                vals.append(tf)

        n_docs = len(self.chunks)
        tf = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(n_docs, len(vocab)),
        )
        df = np.bincount(tf.indices, minlength=len(vocab)).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))

        avg_len = float(doc_lens.mean()) if n_docs else 1.0
        norm = self.k1 * (1 - self.b + self.b * doc_lens / max(avg_len, 1e-9))
        # Per-entry BM25 weight: idf * tf * (k1 + 1) / (tf + norm_doc)
        tf_coo = tf.tocoo()
        weights = (
            idf[tf_coo.col]
            * tf_coo.data
            * (self.k1 + 1)
            / (tf_coo.data + norm[tf_coo.row])
        ).astype(np.float32)
        # Term-major (CSR over terms) so a query is a sum of a few rows.
        self.term_doc = sparse.csr_matrix(
            (weights, (tf_coo.col, tf_coo.row)), shape=(len(vocab), n_docs)
        )
        self.vocab = vocab

    @classmethod
    def from_jsonl(cls, path: pathlib.Path = CHUNKS_PATH) -> "BM25Index":
        chunks = []
        with pathlib.Path(path).open("r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return cls(chunks)

    # ----------------------------------------------------------------- query
    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Optional[Dict]:
        row = self.id_to_row.get(chunk_id)
        return None if row is None else self.chunks[row]

    def _episode_mask(self, allowed_episodes: Optional[Iterable[int]]) -> Optional[np.ndarray]:
        if not allowed_episodes:
            return None
        allowed = np.fromiter((int(e) for e in allowed_episodes), dtype=np.int64)
        return np.isin(self.episode_numbers, allowed)

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(len(self.chunks), dtype=np.float32)
        # Repeated query terms count multiple times, as in standard BM25.
        return np.asarray(self.term_doc[cols].sum(axis=0)).ravel()

    def search(
        self,
        query: str,
        top_n: int = 50,
        allowed_episodes: Optional[Iterable[int]] = None,
    ) -> List[Tuple[str, float]]:
        """Return up to `top_n` (chunk_id, score) pairs with score > 0, best first."""
        scores = self.scores(query)
        mask = self._episode_mask(allowed_episodes)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return []
        if candidates.size > top_n:
            part = np.argpartition(-scores[candidates], top_n - 1)[:top_n]
            candidates = candidates[part]
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [(self.chunks[i]["chunk_id"], float(scores[i])) for i in order]

    def episode_chunks(self, episode_number: int) -> List[Dict]:
        """All chunks of an episode in transcript order."""
        rows = np.flatnonzero(self.episode_numbers == int(episode_number))
        return sorted((self.chunks[i] for i in rows), key=lambda c: c["chunk_index"])


_INDEX_CACHE: Dict[str, BM25Index] = {}
_INDEX_LOCK = threading.Lock()


def get_bm25_index(path: pathlib.Path = CHUNKS_PATH) -> BM25Index:
    """Return a process-wide cached BM25 index for the given chunks file."""
    key = str(pathlib.Path(path).resolve())
    index = _INDEX_CACHE.get(key)
    if index is not None:
        return index
    with _INDEX_LOCK:
        index = _INDEX_CACHE.get(key)
        if index is None:
            index = BM25Index.from_jsonl(path)
            _INDEX_CACHE[key] = index
    return index
