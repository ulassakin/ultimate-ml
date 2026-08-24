import json
import re
from pathlib import Path

from .schemas import TranscriptDocument, TranscriptSegment, TranscriptSource
from .transcript_provider import TranscriptUnavailableError, canonical_youtube_url


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


class YoutubeService:
    def __init__(self, cache_directory: Path):
        self.cache_directory = cache_directory
        self.cache_directory.mkdir(parents=True, exist_ok=True)

    def save_transcript(self, import_id: str, document: TranscriptDocument) -> Path:
        path = self.cache_directory / f"{import_id}.json"
        path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_transcript(self, path: str | Path) -> TranscriptDocument:
        return TranscriptDocument.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def pasted_transcript(self, text: str, *, title: str = "", video_url: str = "", channel: str = "") -> TranscriptDocument:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Paste a non-empty transcript.")
        # A pasted transcript may not have timestamps. Approximate positions only for navigation.
        position = 0.0
        segments = []
        for line in lines:
            segments.append(TranscriptSegment(text=line, start_seconds=position))
            position += max(3.0, len(line.split()) / 2.5)
        try:
            canonical_url = canonical_youtube_url(video_url) if video_url else None
        except TranscriptUnavailableError:
            canonical_url = video_url or None
        return TranscriptDocument(source=TranscriptSource(kind="pasted", video_url=canonical_url,
            title=title or "Pasted transcript", channel=channel or None,
            transcript_attribution="Transcript pasted locally by the user"), segments=segments)

    @staticmethod
    def chunks(document: TranscriptDocument, max_characters: int = 18000) -> list[str]:
        chunks, current = [], []
        length = 0
        for segment in document.segments:
            prefix = f"[{segment.start_seconds:.0f}s] " if segment.start_seconds is not None else ""
            part = prefix + segment.text
            if current and length + len(part) > max_characters:
                chunks.append("\n".join(current)); current=[]; length=0
            current.append(part); length += len(part) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    @staticmethod
    def preview(document: TranscriptDocument, limit: int = 900) -> str:
        return " ".join(segment.text for segment in document.segments)[:limit]

    @staticmethod
    def match_existing_topic(name: str, topics: dict) -> dict | None:
        normalized = _normal(name)
        for item in topics.values():
            if normalized in {_normal(item["id"]), _normal(item["title"])}:
                return {"id": item["id"], "title": item["title"], "match_type": "exact"}
        return None
