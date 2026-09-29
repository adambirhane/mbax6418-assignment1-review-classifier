"""Prompt construction for review classification.

The prompt is the contract between the code and the model: it takes a
review's title and text and demands a strict, machine-readable JSON answer.
The star rating NEVER appears in the prompt — the rating is only ever used
afterwards, as the "correct answer" to score against.

The build is parameterized so later assignment steps slot in:
  * classes          — 2-class (POSITIVE/NEGATIVE) for steps 1-5,
                       3-class (POSITIVE/NEUTRAL/NEGATIVE) from step 6 on
  * include_emotion  — step 5 adds the primary-emotion output.
"""

CLASSES_2 = ("POSITIVE", "NEGATIVE")
CLASSES_3 = ("POSITIVE", "NEUTRAL", "NEGATIVE")

# The 8 emotions of the NRC emotion lexicon (Mohammad & Turney, 2013).
# The LLM is asked to pick from exactly this set so that its answers are
# comparable to the word-list derivation, which can only see these 8.
NRC_EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy",
                "sadness", "surprise", "trust")


def _class_instruction(classes):
    if tuple(classes) == CLASSES_3:
        return (
            "- Answer with exactly one of POSITIVE, NEUTRAL, NEGATIVE. Reserve NEUTRAL "
            "for reviews that are genuinely flat or balanced — no praise, no complaint, "
            "no clear lean in either direction. Mild positives are still POSITIVE and "
            "mild complaints are still NEGATIVE; a review that leans at all gets its "
            "leaning label."
        )
    return (
        "- Answer with exactly one of POSITIVE, NEGATIVE. There is no neutral option: "
        "if a review does not clearly lean either way, assign the overall tone of the "
        "majority of its text."
    )


def _emotion_instruction():
    return (
        "- Also name the single dominant emotion of the review, chosen exactly from: "
        + ", ".join(NRC_EMOTIONS)
        + ". Pick the strongest feeling the review actually conveys, even when the "
        "overall sentiment is clear from tone alone (e.g. a positive review can carry "
        "joy, trust, or anticipation; a negative one can carry anger, fear, disgust, "
        "or sadness). The emotion value MUST be one of those 8 words, verbatim — never "
        "a synonym or paraphrase (write \"joy\", not \"happiness\" or \"relief\"). If no "
        "word fits perfectly, choose the closest of the 8."
    )


def system_prompt(classes=CLASSES_2, include_emotion=False):
    """The system (role) prompt. Constant across every review of a run."""
    classes = tuple(classes)
    rules = [
        "- Judge the review as a whole; the title and the body text carry meaning together.",
        "- If the title and the text conflict, trust the TEXT — the body is what actually happened.",
        "- Short or terse reviews still express sentiment (\"love it\" is positive; \"junk\" is "
        "negative). Classify by the words actually used, even when there is very little text.",
        "- Sarcasm and irony count: judge the intended meaning, not the literal wording.",
        _class_instruction(classes),
    ]
    if include_emotion:
        rules.append(_emotion_instruction())
    parts = [
        "You are a review-classification model for the Amazon \"Gift Cards\" category.",
        "",
        "You are given a review's TITLE and TEXT and must judge the sentiment the "
        "reviewer actually expresses. You are never shown the star rating, and you "
        "must not guess or assume one — judge the words alone.",
        "",
        "Rules:",
        *rules,
        "",
        "Output ONLY valid JSON with no commentary, in exactly this shape:",
        ('{"sentiment": "<SENTIMENT>", "emotion": "<EMOTION>"}' if include_emotion
         else '{"sentiment": "<SENTIMENT>"}'),
        "Do not include any text outside the JSON object.",
    ]
    return "\n".join(parts)


def user_prompt(title, text, text_cap=1200):
    """The per-review user message. Never receives anything but title + text."""
    title = (title or "").strip() or "(no title)"
    text = (text or "").strip() or "(no text)"
    if len(text) > text_cap:
        text = text[:text_cap] + " [...]"
    return f"Title: {title}\nText: {text}"


def build_prompt(title, text, classes=CLASSES_2, include_emotion=False):
    """Return (system_message, user_message) for one review."""
    return system_prompt(classes=classes, include_emotion=include_emotion), \
        user_prompt(title, text)
