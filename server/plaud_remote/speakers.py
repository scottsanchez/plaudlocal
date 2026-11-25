"""Speaker-attributed transcripts (issue #12, format version 1).

The transcription provider `nemo_speech_chat` sends a recording to NVIDIA NeMo-Speech.cpp
running Nemotron 3.5 Transcribe, which returns a punctuated transcript with `<spk:N>` speaker
tags plus word timestamps. It uses the chat route: NeMo-Speech.cpp's /v1/audio/transcriptions
returns lowercase text without punctuation when speakers are requested. NeMo-Speech.cpp can't
decode Ogg/Opus (what the app uploads), so each recording is converted to 16 kHz mono FLAC
with ffmpeg first.

Each job then has two forms of the transcript:
- `transcript_json`, the document: speakers [{id: "S1"}, ...] and segments [{start, end,
  speaker, text}], one segment per speaker turn;
- `transcript`, the text the app stores and the summary reads, rendered from the document by
  `render`. The app side of issue #12 follows in a later PR and must render the same way; the
  fixture in tests/test_speakers.py pins the rules.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from typing import Any

from .providers import ProviderError, _json, _request, chat

log = logging.getLogger("plaud_remote.speakers")

PROVIDER = "nemo_speech_chat"
MAX_SECONDS = 3600  # NeMo-Speech.cpp's limit per request; longer recordings fail clearly

INSTRUCTION = (
    "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal "
    "fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> "
    "<spk:3> <spk:4> <spk:5> <spk:6> <spk:7> on the transcription."
)
_TAG = re.compile(r"<spk:(\d+)>")
_CONVERT_TIMEOUT = 180  # seconds; a stuck ffmpeg must not hold the only worker for long


# ---- rendering -------------------------------------------------------------

def render(document: dict[str, Any]) -> str:
    """The transcript text for a document (issue #12 rendering rules).

    One paragraph per segment, separated by a blank line, labelled with the speaker's confirmed
    name or "Speaker N". A segment without a speaker is unlabelled. With a single speaker who has
    no confirmed name, the text is plain: "Speaker 1:" on every line of a voice memo is noise.
    """
    speakers: dict[Any, dict[str, Any]] = {}
    for speaker in document.get("speakers") or []:
        if isinstance(speaker, dict):
            speakers.setdefault(speaker.get("id"), speaker)  # first wins, as in the app
    single_unnamed = len(speakers) == 1 and not next(iter(speakers.values())).get("name")
    paragraphs = []
    for segment in document.get("segments") or []:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        speaker = speakers.get(segment.get("speaker"))
        if speaker is None or single_unnamed:
            paragraphs.append(text)
        else:
            label = speaker.get("name") or f"Speaker {str(speaker['id'])[1:]}"
            paragraphs.append(f"{label}: {text}")
    return "\n\n".join(paragraphs)


# ---- audio -----------------------------------------------------------------

def _duration_seconds(audio_path: str) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
            check=True, capture_output=True, timeout=30,
        )
        return float(out.stdout.decode().strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None  # unknown: let the transcription server decide


def to_flac(audio_path: str) -> bytes:
    """The recording as 16 kHz mono FLAC. Bad or over-long audio fails permanently."""
    seconds = _duration_seconds(audio_path)
    if seconds is not None and seconds > MAX_SECONDS:
        minutes, rest = divmod(int(seconds + 0.999), 60)
        raise ProviderError(
            f"The recording is {minutes} min {rest:02d} s long; speaker transcription handles up to "
            f"{MAX_SECONDS // 60} minutes per recording"
        )
    with tempfile.TemporaryDirectory(prefix="plaud-flac-") as tmp:
        out = os.path.join(tmp, "audio.flac")
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", out],
                check=True, capture_output=True, timeout=_CONVERT_TIMEOUT,
            )
        except FileNotFoundError:
            raise ProviderError(
                "ffmpeg is not installed on the server; it is needed to convert recordings "
                "(Docker: rebuild with PLAUD_REMOTE_WITH_FFMPEG=1)"
            ) from None
        except subprocess.TimeoutExpired:
            # Probably load, not a bad file: worth another attempt.
            raise ProviderError(f"Converting the recording took longer than {_CONVERT_TIMEOUT} s", retryable=True) from None
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", "replace").strip()[:300]
            raise ProviderError(f"Could not convert the recording: {detail or 'ffmpeg failed'}") from None
        with open(out, "rb") as handle:
            return handle.read()


# ---- request ---------------------------------------------------------------

def transcribe(config: dict[str, Any], audio_path: str) -> tuple[str, dict[str, Any]]:
    """Transcribe one recording; returns (transcript text, document)."""
    url = f"{config['base_url']}/chat/completions"
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(to_flac(audio_path)).decode(), "format": "flac"}},
        ]}],
        "timestamp_granularities": ["word"],
        "temperature": 0,
    }
    raw = _request(
        url, api_key=config.get("api_key", ""), body=json.dumps(payload).encode(),
        content_type="application/json", timeout=config["timeout_seconds"],
    )
    data = _json(raw, url)
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"{url} returned an unexpected response shape") from None
    transcription = data.get("transcription") if isinstance(data.get("transcription"), dict) else {}
    document = build(content, transcription, model=config["model"])
    text = render(document)
    if not text:
        raise ProviderError("The transcription model returned no text")
    return text, document


# ---- parsing ---------------------------------------------------------------

def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word).lower()


def _turns(content: str) -> list[tuple[int | None, str]]:
    """[(raw speaker number or None, text)]: empty pieces dropped, same-speaker runs merged."""
    parts = _TAG.split(content)
    pieces: list[tuple[int | None, str]] = []
    lead = " ".join(parts[0].split())
    if lead:  # text before the first tag, or a transcript without tags
        pieces.append((None, lead))
    for number, text in zip(parts[1::2], parts[2::2]):
        text = " ".join(text.split())
        if text:
            pieces.append((int(number), text))
    merged: list[tuple[int | None, str]] = []
    for speaker, text in pieces:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, f"{merged[-1][1]} {text}")
        else:
            merged.append((speaker, text))
    return merged


def _turn_times(turns: list[tuple[int | None, str]], words: list[dict[str, Any]]) -> list[tuple[float, float] | None]:
    """(start, end) per turn from word timestamps, or None for every turn if they don't line up.

    Words come in time order, but each has `word_index`, its position in the transcript text:
    sorted by it, word i is the transcript's i-th whitespace-separated word. A turn runs from its
    first word's start to its last word's end. The per-word `speaker` is not used: it comes from
    a separate acoustic diarizer and does not follow the text's tags.
    """
    if not all(isinstance(w, dict) and isinstance(w.get("word_index", 0), int) for w in words):
        return [None] * len(turns)
    ordered = sorted(words, key=lambda w: w.get("word_index", 0))
    tokens = [token for _, text in turns for token in text.split()]
    if len(ordered) != len(tokens) or any(
        _norm(token) != _norm(str(word.get("word", ""))) for token, word in zip(tokens, ordered)
    ):
        return [None] * len(turns)
    spans: list[tuple[float, float] | None] = []
    position = 0
    for _, text in turns:
        count = len(text.split())
        start, end = ordered[position].get("start"), ordered[position + count - 1].get("end")
        position += count
        ok = isinstance(start, (int, float)) and isinstance(end, (int, float))
        spans.append((float(start), float(end)) if ok else None)
    return spans


def build(content: str, transcription: dict[str, Any], model: str = "") -> dict[str, Any]:
    """The issue #12 document for a NeMo-Speech.cpp chat response."""
    turns = _turns(content)
    aligned = (transcription.get("word_timestamp_alignment") or {}).get("status") == "aligned"
    times = _turn_times(turns, transcription.get("words") or []) if aligned else [None] * len(turns)

    order: dict[int, int] = {}  # raw tag -> S-number, by first appearance
    for speaker, _ in turns:
        if speaker is not None and speaker not in order:
            order[speaker] = len(order) + 1
    segments = [
        {
            "start": span[0] if span else None,
            "end": span[1] if span else None,
            "speaker": f"S{order[speaker]}" if speaker is not None else None,
            "text": text,
        }
        for (speaker, text), span in zip(turns, times)
    ]
    return {
        "version": 1,
        "source": {"provider": "nemo-speech.cpp", "model": model, "diarized": bool(order)},
        "speakers": [{"id": f"S{n}"} for n in range(1, len(order) + 1)],
        "segments": segments,
    }


def valid_document(value: Any) -> bool:
    """Light shape check for a client-supplied document (text jobs)."""
    return (
        isinstance(value, dict) and value.get("version") == 1
        and isinstance(value.get("speakers"), list) and isinstance(value.get("segments"), list)
        and all(isinstance(s, dict) and isinstance(s.get("id"), str) for s in value["speakers"])
        and all(isinstance(s, dict) and isinstance(s.get("text"), str) for s in value["segments"])
    )


# ---- speaker name suggestions (issue #12) -----------------------------------

NAMES_SYSTEM = "You identify speakers in meeting transcripts. Speakers are labelled Speaker 1, Speaker 2, and so on."
NAMES_REQUEST = (
    "Suggest a real name for each speaker only where the conversation supports it: an introduction, "
    "or being addressed by name and answering. Reply with JSON only, keyed S1, S2, ...: "
    '{"S1": {"name": "...", "confidence": 0.0-1.0, "evidence": "a short quote or reason"}}. '
    "Use null for the name when unsure. Sometimes one label is really two or more people: the same "
    "label says \"I'm A\" in one place and \"I'm B\" in another. Use null for such a label. "
    "Mentioning or introducing someone else (\"this is B\") doesn't count."
)
# The last three sentences are there because the transcriber can merge similar voices into one
# label. Without them, Qwen3.8-27B named such labels after one of the people at full confidence
# (NVIDIA's 8-voice demo and an AMI meeting); with them it left them blank, kept the other names,
# and still named both people when one introduces the other.
MIN_CONFIDENCE = 0.5
_NAMES_TIMEOUT = 300  # seconds; names are a nicety, so a slow model is cut off rather than holding the job


def _one_line(value: Any, limit: int) -> str:
    return " ".join(str(value).split())[:limit] if isinstance(value, str) else ""


def _confidence(value: Any) -> float | None:
    """A confidence in 0..1, or None if it isn't a usable number (bad types, inf, huge ints)."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return min(max(number, 0.0), 1.0) if math.isfinite(number) else None


_NOT_A_NAME = re.compile(r"(null|none|unknown|n/?a|speaker\s*\d*|s\d+)", re.IGNORECASE)


def suggest_names(config: dict[str, Any], transcript: str, document: dict[str, Any]) -> dict[str, Any]:
    """The document with LLM name suggestions added. Never raises: on any failure the document
    comes back unchanged and the job carries on to its summary.

    Suggestions below MIN_CONFIDENCE are dropped; names are cleaned to one short line; speakers
    with a confirmed name are left alone.
    """
    speakers_list = [s for s in document.get("speakers") or [] if isinstance(s, dict)]
    if len(speakers_list) < 2:
        return document
    limited = dict(config, timeout_seconds=min(config["timeout_seconds"], _NAMES_TIMEOUT))
    try:
        reply = chat(limited, NAMES_SYSTEM,
                     f"{NAMES_REQUEST}\n\nTranscript:\n\n{transcript[: config['max_chunk_chars']]}")
        start = reply.find("{")
        names, _ = json.JSONDecoder().raw_decode(reply[start:]) if start >= 0 else ({}, 0)
    except Exception as error:  # noqa: BLE001 - any failure just means no suggestions
        log.warning("Speaker name suggestion failed: %s", error)
        return document
    if not isinstance(names, dict):
        return document
    updated = json.loads(json.dumps(document))
    for speaker in updated.get("speakers") or []:
        if not isinstance(speaker, dict) or speaker.get("name"):
            continue  # a confirmed name is never second-guessed
        entry = names.get(speaker.get("id"))
        if not isinstance(entry, dict):
            continue
        name = _one_line(entry.get("name"), 60)
        confidence = _confidence(entry.get("confidence", 0))
        if name and not _NOT_A_NAME.fullmatch(name) and confidence is not None and confidence >= MIN_CONFIDENCE:
            speaker["suggestion"] = {
                "name": name, "source": "llm", "confidence": round(confidence, 2),
                "evidence": _one_line(entry.get("evidence"), 300),
            }
    return updated


def names_hint(document: dict[str, Any] | None, suggestions: bool = True) -> str:
    """One line for the summary prompt: confirmed names as such, then suggestions (if wanted)
    marked as guesses. Never includes the evidence."""
    confirmed, suggested = [], []
    for speaker in (document or {}).get("speakers") or []:
        if not isinstance(speaker, dict) or not str(speaker.get("id", "")).startswith("S"):
            continue
        label = f"Speaker {str(speaker['id'])[1:]}"
        name = _one_line(speaker.get("name"), 60)
        suggestion = speaker.get("suggestion") if isinstance(speaker.get("suggestion"), dict) else {}
        confidence = _confidence(suggestion.get("confidence", 0))
        if name:
            confirmed.append(f"{label} = {name}")
        elif suggestions and confidence is not None and confidence >= MIN_CONFIDENCE:
            guess = _one_line(suggestion.get("name"), 60)
            if guess:
                suggested.append(f"{label} = {guess}")
    parts = []
    if confirmed:
        parts.append(f"Speaker names: {', '.join(confirmed)}.")
    if suggested:
        parts.append(f"Speaker names suggested from context (unconfirmed): {', '.join(suggested)}.")
    return " ".join(parts)
