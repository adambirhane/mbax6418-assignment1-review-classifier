"""Thin client for the class OpenAI-compatible endpoint.

Endpoint / key / model are the ones given in the assignment brief. The
client is deliberately stdlib-only (urllib) so the project has zero
dependencies and runs anywhere.

Two endpoint quirks were discovered during setup and are handled here:
  * The served model (a Qwen3 reasoning model) burns tokens on a "thinking"
    pass before answering. Thinking is disabled via chat_template_kwargs so
    each classification costs ~9 completion tokens instead of ~500, with no
    measurable difference in output for this task.
  * response_format={"type": "json_object"} makes the model emit parseable
    JSON. A tolerant regex fallback still guards against a stray non-JSON
    reply: if it cannot be parsed, the raw text is kept and the row is
    flagged parse_fail rather than silently dropped.

The rating is never passed in; sentiment classes and emotions are the only
things the model is asked for, and everything originates in prompt.py.
"""

import json
import re
import time
import urllib.error
import urllib.request

from prompt import CLASSES_2, NRC_EMOTIONS, build_prompt

BASE_URL = "http://dobolyi.com:9001/v1"
API_KEY = "6418"
MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


class LLMError(Exception):
    """Raised when the endpoint stays unreachable after retries."""


def chat_completion(system, user, temperature=0.0, max_tokens=512,
                    json_mode=True, thinking=False, max_attempts=3,
                    timeout=120):
    """One chat-completion call with retries on network/5xx failures."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    body = json.dumps(payload).encode()
    last_err = None
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            BASE_URL + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            content = msg.get("content")
            if content is None:
                # Model emitted only reasoning; treat as retryable empty answer.
                raise LLMError("empty assistant content (thinking-only reply)")
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                KeyError, json.JSONDecodeError, LLMError) as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(2 ** attempt)  # 2s, 4s backoff
    raise LLMError(f"endpoint unreachable after {max_attempts} attempts: {last_err}")


_SENT_RE = re.compile(r'"sentiment"\s*:\s*"([^"]+)"')
_EMO_RE = re.compile(r'"emotion"\s*:\s*"([^"]+)"')

# The 35B model sometimes answers with the most natural review emotion even
# when it is outside the NRC-8 vocabulary (e.g. "relief", "happiness").
# Both takes on emotion (LLM and word list) must live in the same 8-emotion
# space to be comparable, so known near-synonyms are normalized to their
# closest NRC emotion. Unmapped values stay emotion_missing.
EMOTION_SYNONYMS = {
    "relief": "joy", "happiness": "joy", "happy": "joy", "love": "joy",
    "delight": "joy", "enjoyment": "joy", "content": "joy",
    "gratitude": "trust", "grateful": "trust", "thankful": "trust",
    "appreciation": "trust", "confidence": "trust",
    "frustration": "anger", "frustrated": "anger", "annoyance": "anger",
    "annoyed": "anger", "irritated": "anger", "outrage": "anger",
    "hate": "anger", "disappointment": "sadness", "disappointed": "sadness",
    "boredom": "sadness", "bored": "sadness", "regret": "sadness",
    "excitement": "anticipation", "excited": "anticipation",
    "enthusiasm": "anticipation", "hopeful": "anticipation",
    "worry": "fear", "worried": "fear", "anxiety": "fear", "afraid": "fear",
}


def parse_model_output(content, classes, include_emotion):
    """Extract sentiment/emotion from the model's reply as leniently as possible.

    Returns (sentiment, emotion, sentiment_ok, emotion_ok, raw).
    sentiment_ok is True when the sentiment was recovered; emotion_ok is True
    when an emotion was requested AND recovered. The raw content is always
    kept for the saved run. A missing/out-of-vocabulary emotion never
    invalidates an otherwise good sentiment row.
    """
    sentiment, emotion = None, None
    classes = tuple(classes)

    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            sentiment = obj.get("sentiment")
            if include_emotion:
                emotion = obj.get("emotion")
    except (json.JSONDecodeError, TypeError):
        pass

    if sentiment is None:
        m = _SENT_RE.search(content or "")
        sentiment = m.group(1) if m else None
    if include_emotion and emotion is None:
        m = _EMO_RE.search(content or "")
        emotion = m.group(1) if m else None

    # Normalize and validate against the allowed vocabulary.
    if isinstance(sentiment, str):
        sentiment = sentiment.strip().upper()
        if sentiment not in classes:
            sentiment = None
    else:
        sentiment = None
    sentiment_ok = sentiment is not None

    if include_emotion:
        if isinstance(emotion, str):
            emotion = emotion.strip().lower()
            if emotion not in NRC_EMOTIONS:
                emotion = EMOTION_SYNONYMS.get(emotion)  # None if unmapped
        else:
            emotion = None
    emotion_ok = (not include_emotion) or emotion is not None
    return sentiment, emotion, sentiment_ok, emotion_ok, content


def classify_review(title, text, classes=CLASSES_2, include_emotion=False,
                    max_attempts=3):
    """Classify one review. Returns a dict with everything the run needs.

    Retries only while the SENTIMENT is missing (the only fatal failure
    mode); an out-of-vocabulary emotion is normalized via EMOTION_SYNONYMS
    or flagged emotion_missing, never retried — the endpoint is
    deterministic at temperature 0, so retrying cannot change the answer.
    """
    system, user = build_prompt(title, text, classes=classes,
                                include_emotion=include_emotion)
    content = None
    sentiment = emotion = None
    sentiment_ok = emotion_ok = False
    for attempt in range(max_attempts):
        content = chat_completion(system, user)
        sentiment, emotion, sentiment_ok, emotion_ok, _ = parse_model_output(
            content, classes, include_emotion)
        if sentiment_ok:
            break
    return {
        "sentiment_pred": sentiment,
        "emotion_pred": emotion,
        "raw_model_output": content,
        "parse_fail": not sentiment_ok,
        "emotion_missing": sentiment_ok and include_emotion and not emotion_ok,
    }
