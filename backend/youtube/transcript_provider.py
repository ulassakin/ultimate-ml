from urllib.parse import parse_qs, urlparse

from .schemas import TranscriptDocument, TranscriptSegment, TranscriptSource


class TranscriptUnavailableError(RuntimeError):
    """A transcript is unavailable without attempting to bypass access controls."""


def youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.casefold().split(":")[0]
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        candidate = parse_qs(parsed.query).get("v", [""])[0]
        if not candidate and parsed.path.startswith("/shorts/"):
            candidate = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else ""
    else:
        raise TranscriptUnavailableError("Enter a standard YouTube watch, short, or youtu.be URL.")
    if not candidate or not all(char.isalnum() or char in "-_" for char in candidate):
        raise TranscriptUnavailableError("The YouTube URL does not contain a valid video ID.")
    return candidate


def canonical_youtube_url(url: str) -> str:
    """Discard player state such as arbitrary timestamps while preserving the video identity."""
    return f"https://www.youtube.com/watch?v={youtube_video_id(url)}"


class YouTubeTranscriptProvider:
    """Thin official-access-respecting adapter; it never authenticates or circumvents restrictions."""

    def retrieve(self, url: str, *, title: str = "", channel: str = "") -> TranscriptDocument:
        video_id = youtube_video_id(url)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            client = YouTubeTranscriptApi()
            fetched = client.fetch(video_id, languages=["en"])
        except Exception as exc:
            raise TranscriptUnavailableError("No accessible transcript was available. Paste a transcript manually instead.") from exc
        segments = []
        for item in fetched:
            text = getattr(item, "text", None) or item.get("text", "")
            start = getattr(item, "start", None)
            if start is None and isinstance(item, dict):
                start = item.get("start")
            duration = getattr(item, "duration", None)
            if duration is None and isinstance(item, dict):
                duration = item.get("duration")
            if text.strip():
                segments.append(TranscriptSegment(text=text.strip(), start_seconds=start, duration_seconds=duration))
        if not segments:
            raise TranscriptUnavailableError("The available transcript was empty. Paste a transcript manually instead.")
        return TranscriptDocument(source=TranscriptSource(kind="youtube", video_url=canonical_youtube_url(url), video_id=video_id,
            title=title or f"YouTube video {video_id}", channel=channel or None, transcript_attribution="YouTube transcript retrieved locally"), segments=segments)
