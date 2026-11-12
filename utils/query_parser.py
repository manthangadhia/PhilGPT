"""Parse user queries: pull out explicit episode references ("ep 12", "episodes 3-5")."""
import re
from dataclasses import dataclass, field
from typing import List

EPISODE_BLOCK = re.compile(
    r"\b(?:ep|eps|episode|episodes)\b\.?[\s#]*([0-9][0-9,\s-]*(?:\s*(?:and|&|to)\s*[0-9]+)*)",
    re.IGNORECASE,
)
RANGE = re.compile(r"(\d+)(?:\s*(?:-|to)\s*(\d+))?")
MAX_RANGE = 50  # guard against "episodes 1-1000" exploding the filter

# Words that carry no topic once the episode reference is removed.
_FILLER = re.compile(
    r"\b(?:summari[sz]e|summary|of|the|a|an|in|on|from|about|what|whats|what's|is|are|was|"
    r"tell|me|give|explain|describe|discussed|discuss|covered|cover|happens|happened|"
    r"talk|talked|talks|say|says|said|and|&|please|can|could|you|do|does|did|overview|"
    r"recap|main|key|points|ideas|idea|topics|topic|go|over|through|compare|between)\b",
    re.IGNORECASE,
)


@dataclass
class ParsedQuery:
    original: str
    text: str  # query with episode references removed
    episodes: List[int] = field(default_factory=list)

    @property
    def has_topic(self) -> bool:
        """False for queries like 'summarise episode 120' that carry no searchable topic."""
        leftover = _FILLER.sub(" ", self.text)
        return bool(re.search(r"[A-Za-z]{3,}", leftover))


def extract_episode_numbers(text: str) -> List[int]:
    episodes = []
    for block_match in EPISODE_BLOCK.finditer(text or ""):
        block = block_match.group(1)
        for start, end in RANGE.findall(block):
            start_n = int(start)
            end_n = int(end) if end else start_n
            if end_n < start_n:
                start_n, end_n = end_n, start_n
            if end_n - start_n > MAX_RANGE:
                end_n = start_n + MAX_RANGE
            episodes.extend(range(start_n, end_n + 1))
    return sorted(set(episodes))


def strip_episode_refs(text: str) -> str:
    stripped = EPISODE_BLOCK.sub(" ", text or "")
    return re.sub(r"\s+", " ", stripped).strip(" ,.;:-")


def parse_query(query: str) -> ParsedQuery:
    return ParsedQuery(
        original=query,
        text=strip_episode_refs(query),
        episodes=extract_episode_numbers(query),
    )
