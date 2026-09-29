"""Word-list derivation of primary emotion (NRC emotion lexicon).

Usage:
    python3 score_emotions.py --run step5_first100

Takes an LLM-scored run (output/runs/<run>.jsonl) and adds the second,
model-free take on emotion: every word of the review's title+text is looked
up in the NRC emotion lexicon (Mohammad & Turney, 2013 — public word-emotion
association list covering anger, anticipation, disgust, fear, joy, sadness,
surprise, trust), the per-emotion scores are summed, and the highest total
is the review's word-list primary emotion.

This never calls the model; it runs over the saved predictions only.

Output:
    output/runs/<run>_emotions.jsonl          original rows + wordlist fields
    output/runs/<run>_emotions_summary.json   comparison: LLM vs word list

Deliberate, documented choices:
  * Tokens = lowercase runs of letters/apostrophes; NRC matches exact forms
    (the lexicon already lists inflections such as happy/happier/happiest).
  * A word counts once per emotion association. No negation handling and no
    context: "not happy" still scores joy. That blindness is a feature for
    the comparison — it is a word COUNT, not a reader.
  * Ties for the top score are broken by the first emotion in the fixed
    NRC order below, and flagged in the row (wordlist_tie) so the report
    can discuss them instead of hiding them.
"""

import argparse
import collections
import json
import os
import re
import sys

LEXICON_PATH = "data/NRC-emotion-lexicon-wordlevel-v0.92.txt"
LEXICON_URL = ("https://raw.githubusercontent.com/dinbav/LeXmo/master/"
               "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust")
POLEMOTIONS = ("positive", "negative")

PARTS_RE = None


def ensure_lexicon(path=LEXICON_PATH):
    """Download the (public) NRC lexicon on first use so the repo stays
    lean; the file is cached under data/ afterwards."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading NRC emotion lexicon from {LEXICON_URL}")
    import urllib.request
    urllib.request.urlretrieve(LEXICON_URL, path)


def load_lexicon(path=LEXICON_PATH):
    """word -> {emotion: True} for the 8 emotions; word -> polarity flags."""
    ensure_lexicon(path)
    emo = collections.defaultdict(set)
    pol = collections.defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            w, cat, flag = parts
            if flag != "1":
                continue
            if cat in EMOTIONS:
                emo[w].add(cat)
            elif cat in POLEMOTIONS:
                pol[w].add(cat)
    return emo, pol


def tokenize(text):
    global PARTS_RE
    if PARTS_RE is None:
        PARTS_RE = re.compile(r"[a-z']+")
    return PARTS_RE.findall((text or "").lower())


def derive_emotion(title, text, emo_lex, pol_lex):
    """Return (primary, scores, tie, polarity counts, token_hits)."""
    scores = {e: 0 for e in EMOTIONS}
    pol = {p: 0 for p in POLEMOTIONS}
    hits = 0
    for w in tokenize(f"{title} {text}"):
        for e in (emo_lex.get(w) or ()):
            scores[e] += 1
            hits += 1
        for p in (pol_lex.get(w) or ()):
            pol[p] += 1
    if hits:
        top = max(scores.values())
        tied = [e for e in EMOTIONS if scores[e] == top]
        primary = tied[0]
    else:
        primary, tied, top = None, [], 0
    return primary, scores, bool(len(tied) > 1), pol, hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)

    run_path = f"output/runs/{args.run}.jsonl"
    try:
        rows = [json.loads(line) for line in open(run_path, encoding="utf-8")]
    except FileNotFoundError:
        sys.exit(f"run file not found: {run_path}")

    emo_lex, pol_lex = load_lexicon()
    print(f"lexicon: {sum(len(v) for v in emo_lex.values())} emotion associations")

    out = []
    n_llm_emo = n_wl_emo = n_both = agree = 0
    per_emo_agree = collections.defaultdict(lambda: [0, 0])
    per_sent_agree = collections.defaultdict(lambda: [0, 0])
    divergences = []

    for r in rows:
        primary, scores, tie, pol, hits = derive_emotion(
            r.get("title"), r.get("text"), emo_lex, pol_lex)
        rec = dict(r)
        rec.update({
            "wordlist_emotion": primary,
            "wordlist_scores": scores,
            "wordlist_tie": tie,
            "wordlist_polarity": pol,
            "wordlist_hits": hits,
        })
        out.append(rec)

        llm_emo = r.get("emotion_pred")
        if llm_emo:
            n_llm_emo += 1
        if primary:
            n_wl_emo += 1
        if llm_emo and primary:
            n_both += 1
            per_emo_agree[llm_emo][1] += 1
            per_sent_agree[r["sentiment_pred"]][1] += 1
            if llm_emo == primary:
                agree += 1
                per_emo_agree[llm_emo][0] += 1
                per_sent_agree[r["sentiment_pred"]][0] += 1
            else:
                divergences.append(rec)

    n = len(rows)
    summary = {
        "run": args.run, "rows": n,
        "n_llm_emotion": n_llm_emo, "n_wordlist_emotion": n_wl_emo,
        "n_both": n_both, "agreement": (agree / n_both) if n_both else None,
        "per_emotion_agreement": {k: {"agree": v[0], "n": v[1]}
                                  for k, v in sorted(per_emo_agree.items())},
        "per_sentiment_agreement": {k: {"agree": v[0], "n": v[1]}
                                    for k, v in sorted(per_sent_agree.items())},
    }
    out_path = f"output/runs/{args.run}_emotions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    with open(f"output/runs/{args.run}_emotions_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nrows: {n} | LLM emotion: {n_llm_emo} | wordlist emotion: {n_wl_emo} "
          f"| both: {n_both}")
    print(f"primary-emotion agreement: {agree}/{n_both} = "
          f"{100 * agree / n_both:.1f}%" if n_both else "n/a")
    print("\nagreement by LLM emotion:")
    for e, v in sorted(per_emo_agree.items()):
        print(f"  {e:13s} {v[0]:3d}/{v[1]:3d}")
    print("agreement by sentiment class:")
    for s, v in sorted(per_sent_agree.items()):
        print(f"  {s:9s} {v[0]:3d}/{v[1]:3d}")
    print("\nshowing up to 8 divergences (LLM vs wordlist):")
    for d in divergences[:8]:
        print(f"  ★{d['rating']:.0f} {d['title'][:38]!r:42s} "
              f"llm={d['emotion_pred']:13s} wl={d['wordlist_emotion']:13s} "
              f"scores={json.dumps(d['wordlist_scores'])}")
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
