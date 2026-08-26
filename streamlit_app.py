
"""
Ask Across Videos — v3
Multi-video YouTube research assistant.

Highlights:
- 1–10 YouTube videos (kept from v2)
- Transcript extraction with timestamps
- No arbitrary 30k-character transcript cutoff
- Local transcript chunking + question-focused retrieval
- Follow-up questions with conversation context
- Synthesize / Compare / Consensus / Contradictions / Video-by-video modes
- Source labels with clickable YouTube links
- Duplicate-video detection
- Graceful failures and per-video retry
- Same warm-stone visual direction as v2
"""

import html
import re
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
from openai import OpenAI
import openai as openai_module
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript, NoTranscriptFound

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ---------- PROVIDERS ----------
PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "signup_url": "https://openrouter.ai/keys",
        "models": [
            # Speed-first: GPT-OSS 20B is optimized for lower-latency inference.
            # Use the larger model only as a fallback when quality/reliability needs it.
            "openai/gpt-oss-20b:free",
            "openai/gpt-oss-120b:free",
            "openrouter/free",
        ],
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "signup_url": "https://console.groq.com/keys",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
    },
    "Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "signup_url": "https://aistudio.google.com/apikey",
        "models": ["gemini-2.5-flash", "gemini-2.0-flash"],
    },
}

MAX_VIDEOS = 10
MAX_QUESTIONS = 10
CHUNK_CHARS = 5000
CHUNK_OVERLAP_CHARS = 500
MAX_CHUNKS_PER_VIDEO = 6
MIN_CHUNKS_PER_VIDEO = 1
MAX_EVIDENCE_CHARS = 90000
MAX_EVIDENCE_CHARS_PER_VIDEO = 60000
MAX_EVIDENCE_CHARS_PER_CHUNK = 5000
AI_TIMEOUT_SECONDS = 10
STREAM_ENABLED = True
WHOLE_VIDEO_CONTEXT_CHARS = 90000
SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
PLACEHOLDER_THUMB = "https://placehold.co/480x360/D9D8D5/896A58?text=%F0%9F%8E%A5"

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "what", "which", "who", "why", "how", "when", "where",
    "about", "from", "with", "into", "for", "are", "was", "were", "is", "be",
    "to", "of", "in", "on", "at", "by", "as", "it", "its", "they", "them",
    "their", "do", "does", "did", "can", "could", "would", "should", "will",
    "you", "your", "we", "our", "i", "me", "my", "say", "says", "according",
    "video", "videos", "tell", "explain", "discuss", "discusses",
}


# ---------- HELPERS ----------
def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    patterns = [
        r"(?:v=|youtu\.be/|youtube\.com/(?:shorts/|embed/|live/))([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    return value


def extract_video_urls_from_text(text: str) -> List[str]:
    """Extract YouTube URLs from whitespace/comma/newline-separated text."""
    if not text:
        return []
    pattern = re.compile(
        r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?[^\s,]+|shorts/[A-Za-z0-9_-]{11}[^\s,]*|embed/[A-Za-z0-9_-]{11}[^\s,]*|live/[A-Za-z0-9_-]{11}[^\s,]*)|youtu\.be/[A-Za-z0-9_-]{11}[^\s,]*)",
        re.I,
    )
    return pattern.findall(text)


def youtube_url(video_id: str, start_seconds: int = 0) -> str:
    base = f"https://www.youtube.com/watch?v={video_id}"
    return f"{base}&t={max(0, int(start_seconds))}s" if start_seconds else base


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def tokenize(text: str) -> List[str]:
    return [
        t for t in re.findall(r"[\w][\w'-]{1,}", text.lower(), flags=re.UNICODE)
        if t not in STOPWORDS
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def get_video_info(video_url: str) -> dict:
    video_id = extract_video_id(video_url)
    try:
        oembed_url = (
            "https://www.youtube.com/oembed?"
            f"url=https://www.youtube.com/watch?v={video_id}&format=json"
        )
        resp = requests.get(oembed_url, timeout=10)
        if resp.ok:
            data = resp.json()
            return {
                "title": data.get("title", video_id),
                "author": data.get("author_name", ""),
                "thumbnail": data.get("thumbnail_url", "") or PLACEHOLDER_THUMB,
                "video_id": video_id,
            }
    except Exception:
        pass
    return {
        "title": video_id,
        "author": "",
        "thumbnail": PLACEHOLDER_THUMB,
        "video_id": video_id,
    }


def _snippet_to_dict(snippet) -> dict:
    return {
        "text": str(getattr(snippet, "text", "")).strip(),
        "start": float(getattr(snippet, "start", 0) or 0),
        "duration": float(getattr(snippet, "duration", 0) or 0),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_transcript(video_id: str) -> Tuple[Optional[List[dict]], Optional[str], Optional[str]]:
    try:
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(
                video_id,
                languages=["en", "en-US", "en-IN", "hi", "kn", "te", "ta", "mr", "bn", "gu"],
            )
            lang = getattr(fetched, "language", None) or "preferred"
            snippets = [_snippet_to_dict(x) for x in fetched]
            return snippets, lang, None
        except NoTranscriptFound:
            transcript_list = api.list(video_id)
            first_available = next(iter(transcript_list))
            fetched = first_available.fetch()
            lang = getattr(first_available, "language", None) or getattr(
                first_available, "language_code", "unknown"
            )
            snippets = [_snippet_to_dict(x) for x in fetched]
            return snippets, lang, None
    except CouldNotRetrieveTranscript:
        return None, None, "YouTube could not provide a transcript for this video."
    except Exception as exc:
        return None, None, f"Transcript retrieval failed: {type(exc).__name__}"


def build_chunks(snippets: List[dict]) -> List[dict]:
    """Create timestamp-preserving chunks without a hard whole-transcript cutoff."""
    if not snippets:
        return []

    chunks = []
    current = []
    current_chars = 0

    for snip in snippets:
        text = snip["text"]
        if not text:
            continue

        if current and current_chars + len(text) + 1 > CHUNK_CHARS:
            chunks.append({
                "text": " ".join(x["text"] for x in current),
                "start": current[0]["start"],
                "end": current[-1]["start"] + current[-1]["duration"],
                "index": len(chunks) + 1,
            })

            # Keep a small overlap using recent snippets.
            overlap = []
            overlap_chars = 0
            for x in reversed(current):
                if overlap_chars + len(x["text"]) > CHUNK_OVERLAP_CHARS:
                    break
                overlap.append(x)
                overlap_chars += len(x["text"]) + 1
            current = list(reversed(overlap))
            current_chars = sum(len(x["text"]) + 1 for x in current)

        current.append(snip)
        current_chars += len(text) + 1

    if current:
        chunks.append({
            "text": " ".join(x["text"] for x in current),
            "start": current[0]["start"],
            "end": current[-1]["start"] + current[-1]["duration"],
            "index": len(chunks) + 1,
        })

    return chunks


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load a multilingual embedding model and reuse it across the session."""
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(SEMANTIC_MODEL_NAME)


@st.cache_data(ttl=86400, show_spinner=False)
def embed_text_batch(texts: Tuple[str, ...]) -> Optional[np.ndarray]:
    """Cache multilingual embeddings so re-runs do not re-index the same transcript."""
    model = get_embedding_model()
    if model is None or not texts:
        return None
    vectors = model.encode(
        list(texts), batch_size=32, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def add_semantic_index(processed_results: List[dict]) -> None:
    """Embed transcript chunks in one multilingual batch and cache the result."""
    refs = []
    texts = []
    for video in processed_results:
        if video.get("status") != "ok":
            continue
        for chunk in video.get("chunks", []):
            text = chunk.get("text", "")
            if text:
                texts.append(text)
                refs.append(chunk)
    if not texts:
        return
    vectors = embed_text_batch(tuple(texts))
    if vectors is None:
        return
    for ref, vector in zip(refs, vectors):
        ref["embedding"] = vector

def process_one_video(url: str) -> dict:
    info = get_video_info(url)
    snippets, lang, error = get_transcript(info["video_id"])
    if not snippets:
        return {
            "status": "skipped",
            **info,
            "reason": error or "No captions were available.",
        }

    chunks = build_chunks(snippets)
    full_text = " ".join(x["text"] for x in snippets)
    duration = snippets[-1]["start"] + snippets[-1]["duration"] if snippets else 0

    # Precompute lightweight retrieval metadata once. Semantic vectors are added
    # after all videos are processed so the embedding model can batch the work.
    for chunk in chunks:
        terms = tokenize(chunk["text"])
        counts = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        chunk["term_counts"] = counts
        chunk["term_set"] = set(counts)
        chunk["text_lower"] = chunk["text"].lower()

    return {
        "status": "ok",
        **info,
        "lang": lang,
        "snippets": snippets,
        "chunks": chunks,
        "chars": len(full_text),
        "word_count": len(full_text.split()),
        "duration": duration,
    }


def score_chunk(question: str, chunk: dict, title: str = "") -> float:
    """Fast hybrid lexical retrieval using precomputed chunk statistics."""
    q_terms = tokenize(question)
    if not q_terms:
        return 0.0
    counts = chunk.get("term_counts") or {}
    term_set = chunk.get("term_set") or set(counts)
    title_terms = set(tokenize(title))

    score = 0.0
    unique_hits = 0
    for term in set(q_terms):
        if term in term_set:
            unique_hits += 1
            score += 1.0 + min(counts.get(term, 1), 4) * 0.15
            if term in title_terms:
                score += 0.35

    # Phrase bonus for names / technical phrases.
    text_lower = chunk.get("text_lower", chunk.get("text", "").lower())
    q_clean = " ".join(q_terms)
    if len(q_terms) >= 2 and q_clean in text_lower:
        score += 2.0
    elif len(q_terms) >= 2:
        for n in (3, 2):
            for i in range(max(0, len(q_terms) - n + 1)):
                if " ".join(q_terms[i:i+n]) in text_lower:
                    score += 0.8
                    break

    return score + (unique_hits / max(1, len(set(q_terms)))) * 2.0

def question_is_broad(question: str) -> bool:
    q = question.lower()
    broad_markers = (
        "main topics", "main ideas", "key ideas", "key points", "what is discussed",
        "what do they discuss", "summarize", "summary", "overview", "each video",
        "in each video", "what does each video", "what are the videos about",
    )
    return any(marker in q for marker in broad_markers)


def semantic_scores(question: str, chunks: List[dict]) -> Dict[int, float]:
    """Return cosine similarity scores using cached local embeddings."""
    model = get_embedding_model()
    if model is None:
        return {}
    vectors = [(i, c.get("embedding")) for i, c in enumerate(chunks) if c.get("embedding") is not None]
    if not vectors:
        return {}
    q = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
    return {i: float(np.dot(vec, q)) for i, vec in vectors}


def retrieve_evidence(question: str, processed: Dict[str, dict], retrieval_context: str = "") -> List[dict]:
    """Retrieve evidence with whole-video context when feasible, otherwise multilingual semantic coverage.

    Crucially, retrieval never uses English-only keyword matching as the primary mechanism.
    For a small enough set of videos we pass the full transcript for every video. For larger
    transcripts we select several semantically related and temporally distributed passages.
    """
    evidence = []
    broad = question_is_broad(question)
    retrieval_question = retrieval_context or question
    ok_videos = [v for v in processed.values() if v.get("status") == "ok"]
    total_chars = sum(v.get("chars", 0) for v in ok_videos)
    use_full_transcripts = total_chars <= WHOLE_VIDEO_CONTEXT_CHARS

    for source_index, (url, video) in enumerate(processed.items(), 1):
        if video.get("status") != "ok":
            continue
        chunks = video.get("chunks", [])
        if not chunks:
            continue

        if use_full_transcripts:
            selected = list(chunks)
        elif broad:
            # Broad questions need temporal coverage across the entire video.
            if len(chunks) <= MAX_CHUNKS_PER_VIDEO:
                selected = list(chunks)
            else:
                positions = np.linspace(0, len(chunks) - 1, MAX_CHUNKS_PER_VIDEO, dtype=int).tolist()
                selected = [chunks[i] for i in sorted(set(positions))]
        else:
            # Multilingual semantic retrieval. No English-only lexical gate.
            sem = semantic_scores(retrieval_question, chunks)
            if sem:
                ranked = sorted(sem.items(), key=lambda x: x[1], reverse=True)
                selected = []
                # Strong semantic matches.
                for idx, sim in ranked[:MAX_CHUNKS_PER_VIDEO * 3]:
                    if sim >= 0.18:
                        selected.append(chunks[idx])
                    if len(selected) >= MAX_CHUNKS_PER_VIDEO - 2:
                        break
                # Add temporal coverage so the answer isn't anchored to one isolated moment.
                for idx in (0, len(chunks) // 2, len(chunks) - 1):
                    if chunks[idx] not in selected:
                        selected.append(chunks[idx])
                selected = selected[:MAX_CHUNKS_PER_VIDEO]
                if not selected:
                    selected = [chunks[idx] for idx, _ in ranked[:MAX_CHUNKS_PER_VIDEO]]
            else:
                # If embeddings cannot load, prefer coverage over a broken English-only matcher.
                if len(chunks) <= MAX_CHUNKS_PER_VIDEO:
                    selected = list(chunks)
                else:
                    positions = np.linspace(0, len(chunks) - 1, MAX_CHUNKS_PER_VIDEO, dtype=int).tolist()
                    selected = [chunks[i] for i in sorted(set(positions))]

        unique_selected = []
        seen = set()
        for c in selected:
            idx = c.get("index")
            if idx not in seen:
                seen.add(idx)
                unique_selected.append(c)

        # Full transcript mode gets a larger per-video budget; retrieval mode stays compact.
        video_budget = MAX_EVIDENCE_CHARS_PER_VIDEO if use_full_transcripts else MAX_EVIDENCE_CHARS_PER_VIDEO
        chunk_limit = 5000 if use_full_transcripts else MAX_EVIDENCE_CHARS_PER_CHUNK
        chunk_records = []
        remaining_video_budget = video_budget
        for c in unique_selected:
            text = c["text"].strip()
            if len(text) > chunk_limit:
                text = text[:chunk_limit].rsplit(" ", 1)[0] + " …"
            text = text[:remaining_video_budget]
            if not text:
                continue
            chunk_records.append({
                "start": c["start"], "end": c["end"],
                "url": youtube_url(video["video_id"], int(c["start"])),
                "text": text, "chunk_index": c["index"],
            })
            remaining_video_budget -= len(text)
            if remaining_video_budget <= 0:
                break

        if chunk_records:
            evidence.append({
                "id": f"V{source_index}", "source_index": source_index,
                "video_id": video["video_id"], "title": video["title"],
                "author": video.get("author", ""), "url": youtube_url(video["video_id"]),
                "chunks": chunk_records,
                "whole_video_context": use_full_transcripts,
            })
    return evidence

def flatten_evidence_text(evidence: List[dict]) -> str:
    """Build a compact prompt while preserving evidence from every video."""
    parts = []
    total = 0
    for source in evidence:
        header = f"[{source['id']}] {source['title']} (Video {source['source_index']})"
        video_parts = []
        video_total = 0
        for chunk in source["chunks"]:
            block = (
                f"{header} | transcript segment "
                f"{format_time(chunk['start'])}–{format_time(chunk['end'])}\n"
                f"{chunk['text']}"
            )
            if video_total + len(block) > MAX_EVIDENCE_CHARS_PER_VIDEO:
                break
            video_parts.append(block)
            video_total += len(block)
        joined = "\n\n".join(video_parts)
        if total + len(joined) > MAX_EVIDENCE_CHARS:
            break
        if joined:
            parts.append(joined)
            total += len(joined)
    return "\n\n".join(parts)

def call_ai_with_retry(client, model_list, prompt: str, max_retries: int = 0,
                       max_tokens: Optional[int] = None, system_prompt: Optional[str] = None,
                       stream_placeholder=None) -> str:
    """Fast model fallback with optional safe streaming.

    A failed/deprecated/rate-limited model is skipped immediately. When streaming is
    enabled, only user-facing content is shown; reasoning/meta output is filtered by
    clean_model_output before it reaches the UI.
    """
    preferred = st.session_state.get("working_model")
    models_to_try = ([preferred] + [m for m in model_list if m != preferred]) if preferred else list(model_list)
    errors = []

    for model_name in models_to_try:
        try:
            st.session_state.request_count = st.session_state.get("request_count", 0) + 1
            kwargs = {
                "model": model_name,
                "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                    + [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "timeout": AI_TIMEOUT_SECONDS,
            }
            if st.session_state.get("active_provider") == "OpenRouter":
                kwargs["extra_body"] = {
                    "provider": {"sort": "throughput"},
                    "reasoning": {"effort": "none"},
                }

            if stream_placeholder is not None and STREAM_ENABLED:
                kwargs["stream"] = True
                stream = client.chat.completions.create(**kwargs)
                pieces = []
                last_render = 0.0
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    piece = getattr(delta, "content", None) if delta is not None else None
                    if not piece:
                        continue
                    pieces.append(piece)
                    now = time.monotonic()
                    # Avoid excessive Streamlit rerenders while still feeling live.
                    if now - last_render >= 0.06:
                        partial = clean_model_output("".join(pieces))
                        if partial:
                            stream_placeholder.markdown(partial)
                        last_render = now
                content = "".join(pieces)
                if not content:
                    raise RuntimeError("Provider returned an empty streamed response.")
                stream_placeholder.markdown(clean_model_output(content) or "Generating…")
            else:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Provider returned an empty response.")

            normalized = re.sub(r"\s+", " ", content).strip().lower()
            safety_only = bool(re.fullmatch(
                r"(?:user\s+)?safety\s*:\s*(?:safe|unsafe|allowed|blocked|unknown)",
                normalized,
                flags=re.I,
            ))
            cleaned_preview = clean_model_output(content)
            reasoning_leak = bool(re.search(
                r"(?:here(?:'s| is) (?:a )?(?:thinking|reasoning|analysis)|the user (?:is asking|wants)|"
                r"let me (?:analyze|compare|examine)|i need to (?:analyze|find|organize))",
                cleaned_preview[:500], flags=re.I,
            ))
            if safety_only or not cleaned_preview or reasoning_leak:
                errors.append(f"{model_name}: invalid/meta output")
                if st.session_state.get("working_model") == model_name:
                    st.session_state.working_model = None
                if stream_placeholder is not None:
                    stream_placeholder.empty()
                continue

            st.session_state.working_model = model_name
            return content

        except openai_module.BadRequestError as exc:
            errors.append(f"{model_name}: BadRequestError: {exc}")
        except openai_module.NotFoundError as exc:
            errors.append(f"{model_name}: NotFoundError: {exc}")
        except openai_module.RateLimitError as exc:
            errors.append(f"{model_name}: RateLimitError: {exc}")
        except openai_module.APITimeoutError as exc:
            errors.append(f"{model_name}: APITimeoutError: {exc}")
        except openai_module.APIConnectionError as exc:
            errors.append(f"{model_name}: APIConnectionError: {exc}")
        except openai_module.InternalServerError as exc:
            errors.append(f"{model_name}: InternalServerError: {exc}")
        except Exception as exc:
            errors.append(f"{model_name}: {type(exc).__name__}: {exc}")

        if st.session_state.get("working_model") == model_name:
            st.session_state.working_model = None
        if stream_placeholder is not None:
            stream_placeholder.empty()

    detail = "\n".join(errors[-5:])
    raise RuntimeError(
        "No configured AI model responded.\n\n"
        f"Provider: {st.session_state.get('active_provider', 'unknown')}\n"
        f"Details:\n{detail or 'Unknown provider error.'}"
    )


def build_conversation_context(history: List[dict], max_items: int = 2) -> str:
    if not history:
        return "No previous questions in this conversation."
    recent = history[-max_items:]
    return "\n\n".join(
        f"Previous Q{i}: {item['question']}\nPrevious A{i}: "
        f"{re.sub(r'\s+', ' ', item['answer']).strip()[:700]}"
        for i, item in enumerate(recent, 1)
    )

def build_retrieval_query(question: str, history: List[dict], max_chars: int = 2500) -> str:
    """Expand a follow-up question with prior context for retrieval only.

    Previous answers are NEVER treated as evidence. They are used only to resolve
    references such as "that point", "the second example", or "what about him?"
    The final answer is still generated only from newly retrieved transcript chunks.
    """
    if not history:
        return question
    recent = history[-2:]
    context_bits = []
    for item in recent:
        context_bits.append(f"Earlier question: {item.get('question','')}")
        # Keep context compact so retrieval remains cheap.
        previous_answer = re.sub(r"\s+", " ", item.get("answer", "")).strip()
        context_bits.append(f"Earlier answer context: {previous_answer[:900]}")
    expanded = question + "\n" + "\n".join(context_bits)
    return expanded[:max_chars]


def clean_model_output(text: str) -> str:
    """Return only clean, user-facing answer text.

    Models sometimes emit output wrappers such as <FINAL> without the matching
    closing tag, or leak planning/meta commentary. None of that should ever be
    shown in the UI.
    """
    if not text:
        return ""

    text = text.strip()

    # Extract a complete FINAL wrapper when present.
    tagged = re.search(
        r"<FINAL(?:\s+ANSWER)?\s*>(.*?)</FINAL(?:\s+ANSWER)?\s*>",
        text,
        flags=re.I | re.S,
    )
    if tagged:
        text = tagged.group(1).strip()
    else:
        # Never allow an unmatched wrapper tag to leak into the answer.
        text = re.sub(r"</?FINAL(?:\s+ANSWER)?\s*>", "", text, flags=re.I)

    # Remove common explicit answer/reasoning markers.
    text = re.sub(r"^\s*(?:FINAL ANSWER|ANSWER)\s*:\s*", "", text, flags=re.I)

    lines = text.splitlines()
    bad_starts = (
        "the user is asking", "the user wants", "i need to", "let me analyze",
        "let me compare", "let me examine", "i will analyze", "i'll analyze",
        "here is my analysis", "analysis:", "reasoning:", "thinking:",
        "let me find", "i need to find", "i need to organize",
    )
    cleaned = []
    skipping_meta = False
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if not stripped:
            # Preserve normal paragraph spacing, but don't preserve leading meta blanks.
            if cleaned:
                cleaned.append("")
            continue
        if low.startswith(bad_starts):
            skipping_meta = True
            continue
        # Remove a few common model-planning fragments that appear as standalone lines.
        if low in {
            "let me analyze the transcripts:",
            "let me analyze both videos:",
            "let me analyze the transcript:",
        }:
            skipping_meta = True
            continue
        # Once a normal answer line appears, normal output resumes.
        if skipping_meta and not low.startswith(bad_starts):
            skipping_meta = False
        cleaned.append(line)

    text = "\n".join(cleaned).strip()

    # Remove any stray XML-like final tags that survived in the middle/end.
    text = re.sub(r"</?FINAL(?:\s+ANSWER)?\s*>", "", text, flags=re.I)
    return text.strip()

def normalize_points(text: str) -> str:
    """Normalize Points mode into meaningful, line-separated bullets.

    This is a presentation safeguard: the model must already produce distinct ideas,
    but if it returns multiple '- ' markers on one line, split them so Streamlit
    renders genuine point-wise output instead of one long paragraph.
    """
    if not text:
        return text
    # Convert common bullet glyphs to a consistent marker.
    text = re.sub(r"[•●▪◦]\s*", "- ", text)
    # Split inline bullets when the model flattened them into one paragraph.
    text = re.sub(r"\s+(?=-\s+)", "\n", text)
    # Keep headings separated from bullets.
    text = re.sub(r"\s+(?=\*\*Video\s+\d+[^*]*\*\*)", "\n\n", text, flags=re.I)
    return text.strip()


def hard_limit_words(text: str, target_words: int) -> str:
    """Enforce a hard maximum on the model's answer body."""
    words = text.split()
    if len(words) <= target_words:
        return text.strip()
    clipped = " ".join(words[:target_words]).strip()
    # Prefer a clean sentence ending when one is available near the limit.
    candidates = [clipped.rfind(p) for p in (". ", "! ", "? ", ".", "!", "?")]
    cut = max(candidates)
    if cut >= max(40, int(target_words * 0.72)):
        clipped = clipped[:cut + 1]
    return clipped.strip()


def answer_question(
    client,
    model_list,
    question: str,
    evidence: List[dict],
    answer_format: str,
    target_words: int,
    mode: str,
    history: List[dict],
    stream_placeholder=None,
) -> str:
    if not evidence:
        return "I couldn't find relevant transcript evidence in the processed videos for this question."

    evidence_text = flatten_evidence_text(evidence)
    source_list = "\n".join(
        f"[{e['id']}] Video {e['source_index']}: {e['title']}" for e in evidence
    )

    if mode == "Synthesize":
        mode_instruction = (
            "Synthesize the answer across the provided videos. Explain the overall answer, "
            "but preserve meaningful source differences. Do not split a single video's "
            "chunks into separate videos."
        )
    elif mode == "Compare":
        mode_instruction = (
            "Compare the videos as whole sources. Organize the answer by Video 1, Video 2, "
            "etc. when useful, then summarize similarities and differences. Never treat a "
            "transcript chunk as its own video. Only report differences supported by evidence."
        )
    elif mode == "Consensus":
        mode_instruction = (
            "Find genuine common ground across whole videos. State what multiple videos "
            "support, then identify source-specific points. Do not call two chunks from the "
            "same video a consensus between videos."
        )
    elif mode == "Contradictions":
        mode_instruction = (
            "Look for genuine conflicts between different videos. Compare claims source-by-source. "
            "If there is no meaningful contradiction, explicitly say so rather than inventing one."
        )
    else:
        mode_instruction = (
            "Answer separately for EVERY input video. Use exactly one section per video, "
            "named 'Video 1 — <title>', 'Video 2 — <title>', etc. Combine all relevant chunks "
            "belonging to the same video into that video's single answer. Never create separate "
            "answers for transcript chunks."
        )

    format_instruction = (
        "POINTS MODE: Organize the answer as meaningful, self-contained bullet points. "
        "Each bullet must express ONE distinct idea and briefly explain it. Do not merely "
        "split a paragraph into lines. Use a bold idea label followed by a short explanation, "
        "for example: '- **Creativity matters:** Robinson argues that creativity is as important as literacy.' "
        "For multiple videos, clearly group the meaningful points under each video's title. "
        "Do not output one giant paragraph with hyphens."
        if answer_format == "Points"
        else "PARAGRAPH MODE: Use coherent paragraphs. Keep the required video headings if the mode asks for them."
    )

    prompt = f"""You are a careful multi-source research assistant.

IMPORTANT OUTPUT RULE:
Return ONLY the final user-facing answer. Do NOT reveal your analysis, planning, reasoning process, instructions, prompt interpretation, or phrases such as "the user wants", "I need to", "let me analyze", or "based on the prompt". Do not describe what you are about to do. Start directly with the answer.

QUESTION:
{question}

ANALYSIS MODE:
{mode}
{mode_instruction}

OUTPUT FORMAT:
{format_instruction}

LENGTH RULE — STRICT:
The answer body must contain NO MORE THAN {target_words} words. Target approximately {target_words} words, but NEVER exceed {target_words}. Be concise and prioritize the most useful supported information. Do not use the word budget for meta-commentary.

SOURCE LIST:
{source_list}

FOLLOW-UP CONTEXT:
{build_conversation_context(history)}

SOURCE RULES:
1. Use ONLY the transcript evidence supplied below for factual claims.
2. The evidence may be in English, Hindi, Kannada, Tamil, Telugu, Marathi, Bengali, Gujarati, or another language. Understand the transcript semantically; do not rely on literal English word overlap.
2. Never invent a fact, citation, timestamp, similarity, difference, consensus, contradiction, or source.
3. Cite substantive claims with [V1], [V2], etc.
4. Each [V#] represents ONE WHOLE INPUT VIDEO. Multiple transcript segments under the same [V#] are parts of that same video.
5. Never rename or renumber sources.
6. A previous answer is context only; it is NOT evidence.
7. If the current question refers to "that", "they", "it", etc., use the follow-up context only to understand the reference, then verify it against current transcript evidence.
8. For a follow-up question, answer the new question fully. Do not return only a heading, source name, or fragment merely because the question is short. Resolve the reference from prior context, then retrieve and use fresh transcript evidence.
9. If evidence is insufficient, say so clearly. Do not guess or fill gaps from general knowledge.
10. Treat each input URL as one whole video. The transcript is the source of truth. Answer from the video's overall content and development, not from one matching word or isolated sentence. Retrieval supplies representative evidence; it must not be treated as the video's entire meaning.
11. For broad questions such as "main topics" or "what is discussed in each video", cover each whole video using evidence distributed across the video rather than focusing on one matching phrase.
12. For Compare mode, compare the actual videos as wholes. If the videos have little or no meaningful overlap, explicitly say so. Do NOT invent a similarity just because comparison was requested.
13. For Consensus mode, only call something common ground when it is genuinely supported by at least two DIFFERENT input videos.
14. For Contradictions mode, only report a contradiction when two DIFFERENT input videos make materially conflicting claims. If there is no real contradiction, say that clearly.
15. For Video-by-video mode, produce exactly one section for each actual input video and combine all relevant transcript chunks belonging to that video.
16. Do not output internal labels such as "analysis", "reasoning", "transcript analysis", or "the user wants".

TRANSCRIPT EVIDENCE:
{evidence_text}
"""

    system_prompt = (
        "You are a strict user-facing answer generator. Return ONLY the final answer text. "
        "Never reveal chain-of-thought, analysis, planning, prompt interpretation, or hidden reasoning. "
        "Never say what the user wants, what you need to do, or that you are analyzing. "
        "Do not preface the answer with meta-commentary. Use only the supplied transcript evidence."
    )
    prompt = prompt.replace(
        "Return ONLY the final user-facing answer. Do NOT reveal your analysis, planning, reasoning process, instructions, prompt interpretation, or phrases such as \"the user wants\", \"I need to\", \"let me analyze\", or \"based on the prompt\". Do not describe what you are about to do. Start directly with the answer.",
        "Return only the final user-facing answer between <FINAL> and </FINAL>. Never reveal analysis, planning, reasoning, prompt interpretation, or meta-commentary. Start directly with the answer."
    )
    raw = call_ai_with_retry(
        client,
        model_list,
        prompt,
        max_retries=1,
        max_tokens=max(160, int(target_words * 1.45)),
        system_prompt=system_prompt,
        stream_placeholder=stream_placeholder,
    )
    cleaned = clean_model_output(raw)
    if answer_format == "Points":
        cleaned = normalize_points(cleaned)
    cleaned = hard_limit_words(cleaned, target_words)
    if answer_format == "Points":
        cleaned = normalize_points(cleaned)
    return cleaned


def _inline_answer_html(text: str, evidence_by_id: Dict[str, dict]) -> str:
    """Escape answer text, then safely restore bold labels and clickable [V#] links."""
    escaped = html.escape(text)

    def cite(match):
        key = match.group(1)
        e = evidence_by_id.get(key)
        if not e:
            return match.group(0)
        href = html.escape(e["url"], quote=True)
        return (f'<a href="{href}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#896A58;font-weight:700;text-decoration:underline;">[{key}]</a>')

    escaped = re.sub(r"\[(V\d+)\]", cite, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def render_cited_answer(answer: str, evidence: List[dict]):
    """Render answers with real paragraphs/headings/bullet lists, not flattened text."""
    evidence_by_id = {e["id"]: e for e in evidence}
    lines = answer.splitlines()
    html_parts = ['<div class="answer-block">']
    in_list = False
    paragraph = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            joined = " ".join(x.strip() for x in paragraph if x.strip())
            if joined:
                html_parts.append(f'<p>{_inline_answer_html(joined, evidence_by_id)}</p>')
            paragraph = []

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append('</ul>')
            in_list = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            close_list()
            continue

        if re.match(r"^(?:#{1,3})\s+", line):
            flush_paragraph(); close_list()
            heading = re.sub(r"^#{1,3}\s+", "", line)
            html_parts.append(f'<h4>{_inline_answer_html(heading, evidence_by_id)}</h4>')
            continue

        if re.match(r"^[-*•]\s+", line):
            flush_paragraph()
            if not in_list:
                html_parts.append('<ul>')
                in_list = True
            item = re.sub(r"^[-*•]\s+", "", line)
            html_parts.append(f'<li>{_inline_answer_html(item, evidence_by_id)}</li>')
            continue

        # Bold video headings are treated as headings, not ordinary paragraphs.
        if re.match(r"^\*\*Video\s+\d+", line, flags=re.I):
            flush_paragraph(); close_list()
            html_parts.append(f'<h4>{_inline_answer_html(line, evidence_by_id)}</h4>')
            continue

        close_list()
        paragraph.append(line)

    flush_paragraph(); close_list()
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ---------- STYLE ----------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Marcellus&family=Jost:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {
            --bg: #567357; --surface: #D9D8D5; --surface-2: #ACAB9E;
            --text-on-bg: #D9D8D5; --text-on-bg-muted: #ACAB9E;
            --text-on-surface: #2A2420; --text-on-surface-muted: #6B6459;
            --accent: #896A58; --accent-hover: #77584A; --dark: #2A2420;
            --border: #ACAB9E;
        }
        html, body, [class*="css"] { font-family:'Jost',sans-serif; }
        .stApp { background-color:var(--bg); }
        h1,h2,h3 { font-family:'Marcellus',serif !important; font-weight:400 !important; color:var(--text-on-bg) !important; }
        .app-tagline {
            margin-top:-0.5rem !important;
            font-family:'Marcellus',serif !important;
            font-size:1.02rem !important;
            font-weight:400 !important;
            letter-spacing:.01em;
            color:var(--text-on-bg) !important;
        }
        .app-tagline span {
            font-family:'IBM Plex Mono',monospace !important;
            font-style:italic;
            color:var(--text-on-bg) !important;
        }
        p,span,label,.stMarkdown,.stCaption { color:var(--text-on-bg) !important; }
        .stCaption { color:var(--text-on-bg-muted) !important; }
        .stButton > button {
            background:var(--accent) !important; color:#F5F1EC !important;
            border:none !important; border-radius:9px !important;
            font-family:'Jost',sans-serif !important; font-weight:600 !important;
        }
        .stButton > button:hover { background:var(--accent-hover) !important; }
        .stDownloadButton > button {
            background:var(--surface) !important; color:var(--text-on-surface) !important;
            border:1px solid var(--border) !important; border-radius:9px !important;
            min-height:46px !important; opacity:1 !important;
        }
        .stDownloadButton > button * {
            color:var(--text-on-surface) !important; opacity:1 !important;
            visibility:visible !important; display:inline !important;
        }
        .stDownloadButton > button p, .stDownloadButton > button span {
            color:var(--text-on-surface) !important; opacity:1 !important;
            visibility:visible !important;
        }
        .stTextInput input,.stNumberInput input,.stTextArea textarea {
            background:var(--surface) !important; color:var(--text-on-surface) !important;
            border:1px solid var(--border) !important; border-radius:9px !important;
        }
        [data-testid="stSidebar"] { background:var(--surface-2); }
        [data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,[data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--text-on-surface) !important; }
        [data-baseweb="select"] > div { background:var(--surface) !important; }
        .video-card {
            background:var(--surface); border:1px solid var(--border); border-radius:12px;
            padding:.9rem; margin-bottom:.7rem; box-shadow:0 3px 10px rgba(0,0,0,.22);
        }
        .video-title { font-family:'Marcellus',serif; font-size:1.08rem; color:var(--text-on-surface); }
        .video-meta { font-family:'IBM Plex Mono',monospace; font-size:.75rem; color:var(--text-on-surface-muted); }
        div[data-testid="stImage"] img { border-radius:10px !important; }
        .answer-block {
            background:var(--surface); border-left:3px solid var(--accent); border-radius:10px;
            padding:1.1rem 1.3rem; margin:.6rem 0 1.2rem; box-shadow:0 3px 12px rgba(0,0,0,.22);
            color:var(--text-on-surface);
        }
        .answer-block, .answer-block * { color:var(--text-on-surface) !important; }
        .pill {
            display:inline-block; padding:.25rem .55rem; border-radius:999px;
            background:var(--surface); color:var(--text-on-surface) !important;
            border:1px solid var(--border); font-family:'IBM Plex Mono',monospace; font-size:.7rem;
            margin-right:.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- APP STATE ----------
st.set_page_config(page_title="Ask Across Videos", page_icon="🔎", layout="centered")
inject_css()

defaults = {
    "processed": {},
    "working_model": None,
    "request_count": 0,
    "qa_history": [],
    "active_provider": None,
    "answer_cache": {},
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("🔎 Ask Across Videos")
st.markdown(
    '<p class="app-tagline">Ask questions across multiple YouTube videos and get '
    '<span>sourced answers.</span></p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Setup")
    provider_name = st.selectbox("AI provider", list(PROVIDERS.keys()), index=0)
    provider = PROVIDERS[provider_name]

    secret_key = None
    try:
        secret_key = st.secrets.get("OPENROUTER_API_KEY") if provider_name == "OpenRouter" else st.secrets.get("AI_API_KEY")
    except Exception:
        secret_key = None

    api_key = st.text_input(
        f"{provider_name} API key",
        value=secret_key or "",
        type="password",
        help="A provider API key is required to run AI synthesis.",
    )
    st.markdown(f"[Get a {provider_name} key →]({provider['signup_url']})")

    st.divider()
    st.caption(f"AI requests this session: {st.session_state.request_count}")
    st.caption("Free-tier limits vary by provider.")

    if st.button("↺ New analysis", use_container_width=True):
        st.session_state.processed = {}
        st.session_state.qa_history = []
        st.session_state.working_model = None
        st.session_state.answer_cache = {}
        st.rerun()

if not api_key:
    st.info("👈 Add an API key in the sidebar to get started.")
    st.stop()

if st.session_state.get("active_provider") != provider_name:
    st.session_state.working_model = None
    st.session_state.active_provider = provider_name

client = OpenAI(base_url=provider["base_url"], api_key=api_key)
model_list = provider["models"]


# ---------- VIDEO INPUT ----------
st.subheader("1. Add your videos")
num_videos = st.number_input(
    "How many videos do you have?",
    min_value=1,
    max_value=MAX_VIDEOS,
    value=2,
    step=1,
    key="num_videos_input",
)

video_urls = []
seen_ids = set()
with st.form("video_links_form", clear_on_submit=False):
    for i in range(int(num_videos)):
        url = st.text_input(
            f"Video {i + 1} link",
            key=f"video_url_{i}",
            placeholder="https://youtu.be/...",
        )
        if url.strip():
            vid = extract_video_id(url.strip())
            if vid not in seen_ids:
                seen_ids.add(vid)
                video_urls.append(url.strip())
            else:
                st.caption(f"Video {i + 1}: duplicate video ignored.")

    with st.expander("📋 Or paste multiple YouTube links at once"):
        bulk_urls_text = st.text_area(
            "Paste links separated by spaces, commas, or new lines",
            key="bulk_urls_text",
            placeholder="https://youtu.be/...\nhttps://www.youtube.com/watch?v=...",
            height=90,
        )
        bulk_urls = extract_video_urls_from_text(bulk_urls_text)
        if bulk_urls:
            st.caption(f"Found {len(bulk_urls)} YouTube link(s). Duplicates will be ignored.")
            for bulk_url in bulk_urls[:MAX_VIDEOS]:
                vid = extract_video_id(bulk_url)
                if vid not in seen_ids and len(video_urls) < MAX_VIDEOS:
                    seen_ids.add(vid)
                    video_urls.append(bulk_url)

    process_clicked = st.form_submit_button(
        "🔍 Process videos",
        use_container_width=True,
    )

if process_clicked and not video_urls:
    st.warning("Add at least one YouTube link before processing.")

if process_clicked and video_urls:
    existing_by_id = {
        v.get("video_id"): (url, v)
        for url, v in st.session_state.processed.items()
        if v.get("video_id")
    }
    results = {}

    urls_to_process = []
    for url in video_urls:
        vid = extract_video_id(url)
        if vid in existing_by_id:
            results[url] = existing_by_id[vid][1]
        else:
            urls_to_process.append(url)

    status = st.empty()
    if urls_to_process:
        with ThreadPoolExecutor(max_workers=min(len(urls_to_process), 6)) as executor:
            futures = {executor.submit(process_one_video, url): url for url in urls_to_process}
            done = 0
            for future in as_completed(futures):
                url = futures[future]
                done += 1
                status.info(f"Processing transcript {done}/{len(urls_to_process)}…")
                try:
                    results[url] = future.result()
                except Exception as exc:
                    results[url] = {
                        "status": "skipped",
                        "title": url,
                        "reason": f"Unexpected error: {type(exc).__name__}",
                    }
    status.empty()
    st.session_state.processed = {u: results[u] for u in video_urls if u in results}
    # Build semantic vectors once per processed video set. If the local embedding
    # model is unavailable, retrieval automatically falls back to lexical scoring.
    ok_for_index = list(st.session_state.processed.values())
    if any(v.get("status") == "ok" for v in ok_for_index):
        status = st.empty()
        status.info("🧠 Indexing transcript meaning for faster questions…")
        try:
            add_semantic_index(ok_for_index)
        except Exception:
            # Never block the app if the embedding model cannot load; lexical
            # retrieval remains available.
            pass
        status.empty()
    # Preserve the user's input order as the permanent video identity.
    for source_index, (u, result) in enumerate(st.session_state.processed.items(), 1):
        result["source_index"] = source_index
    # New video set = new conversation context.
    st.session_state.qa_history = []
    st.session_state.answer_cache = {}

# ---------- VIDEO CARDS ----------
if st.session_state.processed:
    st.subheader("2. Videos")

    ok_results = {
        url: r for url, r in st.session_state.processed.items() if r.get("status") == "ok"
    }
    skipped_results = {
        url: r for url, r in st.session_state.processed.items() if r.get("status") != "ok"
    }

    for url, r in ok_results.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r.get("thumbnail") or PLACEHOLDER_THUMB, use_container_width=True)
        with col2:
            st.markdown(
                f'<div class="video-card">'
                f'<div class="video-title">{html.escape(r["title"])}</div>'
                f'<div class="video-meta">by {html.escape(r.get("author") or "Unknown creator")}</div>'
                f'<div class="video-meta">{r.get("chars",0):,} chars · '
                f'{len(r.get("chunks", []))} transcript chunks · '
                f'lang: {html.escape(str(r.get("lang","?")))}</div>'
                f'<div><span class="pill">Transcript indexed</span>'
                f'<a href="{html.escape(youtube_url(r["video_id"]), quote=True)}" target="_blank">'
                f'Open on YouTube ↗</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if skipped_results:
        with st.expander(f"⚠️ {len(skipped_results)} video(s) skipped"):
            for url, r in skipped_results.items():
                st.write(f"**{r.get('title', url)}** — {r.get('reason','Unknown reason')}")
            st.caption("Fix the link above and click Process videos again.")

    if st.session_state.working_model:
        st.caption(f"Using model: {st.session_state.working_model}")

    if ok_results:
        st.divider()
        st.subheader("3. Ask your questions")

        col_mode, col_fmt, col_len = st.columns(3)
        with col_mode:
            mode = st.selectbox(
                "Research mode",
                ["Synthesize", "Compare", "Consensus", "Contradictions", "Video-by-video"],
            )
        with col_fmt:
            answer_format = st.radio("Answer style", ["Points", "Paragraph"], horizontal=True)
        with col_len:
            target_words = st.select_slider(
                "Answer length (~words)",
                options=[75, 150, 250, 400, 600],
                value=250,
            )

        num_q = st.number_input(
            "How many questions do you have?",
            min_value=1,
            max_value=MAX_QUESTIONS,
            value=max(1, min(MAX_QUESTIONS, len(st.session_state.qa_history) + 1)),
            step=1,
        )

        answered_so_far = len(st.session_state.qa_history)

        if answered_so_far < num_q:
            q_number = answered_so_far + 1
            with st.form(f"question_form_{q_number}", clear_on_submit=False):
                question = st.text_input(
                    f"Question {q_number} of {num_q}",
                    key=f"q_input_{q_number}",
                    placeholder="Ask anything about these videos…",
                )
                ask_clicked = st.form_submit_button("💬 Get answer", use_container_width=True)

            if ask_clicked and not question.strip():
                st.warning("Type a question before submitting.")
            if ask_clicked and question.strip():
                progress = st.empty()
                try:
                    progress.info("🔎 Finding evidence across your videos…")
                    retrieval_context = build_retrieval_query(question.strip(), st.session_state.qa_history)
                    evidence = retrieve_evidence(question.strip(), ok_results, retrieval_context)
                    progress.info("🧠 Generating the answer…")
                    stream_box = st.empty()
                    history_key = "|".join(
                        f"{x.get('question','')}::{x.get('answer','')[:300]}"
                        for x in st.session_state.qa_history[-2:]
                    )
                    source_key = "|".join(
                        f"{e['id']}:{e['video_id']}:{','.join(str(c['chunk_index']) for c in e['chunks'])}"
                        for e in evidence
                    )
                    cache_key = repr((question.strip().lower(), mode, answer_format, target_words, source_key, history_key))
                    cached = st.session_state.answer_cache.get(cache_key)
                    if cached:
                        answer = cached
                    else:
                        answer = answer_question(
                            client, model_list, question.strip(), evidence,
                            answer_format, target_words, mode, st.session_state.qa_history,
                            stream_placeholder=stream_box,
                        )
                        st.session_state.answer_cache[cache_key] = answer
                    stream_box.empty()
                    progress.empty()
                    st.session_state.qa_history.append(
                        {
                            "question": question.strip(),
                            "answer": answer,
                            "mode": mode,
                            "format": answer_format,
                            "words": target_words,
                            "evidence": evidence,
                        }
                    )
                    st.rerun()
                except Exception as exc:
                    progress.empty()
                    st.error("Couldn't get an answer. The AI provider may be busy or have rejected the request.")
                    with st.expander("Show technical details"):
                        st.code(f"{type(exc).__name__}: {exc}")
        else:
            st.caption(f"All {num_q} question(s) answered. Increase the number above to ask more.")

        if st.session_state.qa_history:
            st.divider()
            st.subheader("Answers")
            all_text = ""
            for i, qa in enumerate(st.session_state.qa_history, 1):
                st.caption(f"Q{i} · {qa['mode']} · ~{qa['words']} words")
                st.markdown(f"**{qa['question']}**")
                render_cited_answer(qa["answer"], qa["evidence"])
                all_text += f"Q{i}: {qa['question']}\nMode: {qa['mode']}\n\n{qa['answer']}\n\n"
                all_text += "=" * 60 + "\n\n"

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download all answers",
                    data=all_text,
                    file_name="answers.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with c2:
                if st.button("🧹 Clear answers", use_container_width=True):
                    st.session_state.qa_history = []
                    st.rerun()
    else:
        st.warning("No videos could be processed — check the links above.")
