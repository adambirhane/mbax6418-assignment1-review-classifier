"""Score Amazon Gift-Cards reviews against the LLM classifier.

Usage:
    python3 score.py --mode first --n 100 --classes 2 --name step2_first100
    python3 score.py --mode balanced --per-class 50 --seed 42 --classes 3 \
        --emotions --name step6_balanced3

Modes:
    first     take the first N rows in file order (assignment step 2).
    balanced  take ~per_class rows from each rating class, picked with a
              fixed random seed so the same set comes out every time
              (assignment step 6).

The rating is recorded purely to score against afterwards; it is never sent
to the model (llm.classify_review receives only title and text).

Output:
    output/runs/<name>.jsonl          one line per review, everything saved
    output/runs/<name>_summary.json   agreement/confusion metrics vs rating
"""

import argparse
import collections
import gzip
import json
import os
import random
import sys
import time

from llm import classify_review
from prompt import CLASSES_2, CLASSES_3

DATA_PATH = "data/Gift_Cards.jsonl.gz"

KEEP_FIELDS = ("rating", "title", "text", "verified_purchase",
               "helpful_vote", "timestamp")


def load_reviews(path=DATA_PATH):
    """All reviews, in file order. (152,410 rows; a few seconds to parse.)"""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def rating_to_class(rating, classes):
    """The 'correct answer' derived from the star rating only."""
    classes = tuple(classes)
    if classes == CLASSES_3:
        if rating >= 4:
            return "POSITIVE"
        if rating == 3:
            return "NEUTRAL"
        return "NEGATIVE"
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def select_first(reviews, n):
    return [(i, r) for i, r in enumerate(reviews[:n])]


def select_balanced(reviews, per_class, seed, classes):
    """~per_class rows per CLASS (rating-derived), fixed seed, whole file.

    Classes are the rating-derived labels (4-5 positive, 3 neutral, 1-2
    negative), so this yields roughly equal class sizes — about 2:1
    positive:negative in raw ratings, but equal after grouping.
    """
    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for i, r in enumerate(reviews):
        by_class[rating_to_class(r["rating"], classes)].append(i)
    picked = []
    for cls in sorted(by_class):  # deterministic class order
        idx = rng.sample(by_class[cls], k=min(per_class, len(by_class[cls])))
        picked.extend((i, reviews[i]) for i in sorted(idx))
    picked.sort(key=lambda pair: pair[0])  # file order, for stable output
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("first", "balanced"), required=True)
    ap.add_argument("--n", type=int, default=100, help="rows for --mode first")
    ap.add_argument("--per-class", type=int, default=50, help="per class for --mode balanced")
    ap.add_argument("--seed", type=int, default=42, help="random seed (balanced mode)")
    ap.add_argument("--classes", type=int, choices=(2, 3), default=2)
    ap.add_argument("--emotions", action="store_true",
                    help="also ask the LLM for the primary emotion (step 5+ / 6+)")
    ap.add_argument("--name", required=True, help="run name, used for output files")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="safety cap on API calls (default: n or 3*per_class)")
    args = ap.parse_args(argv)

    classes = CLASSES_3 if args.classes == 3 else CLASSES_2

    reviews = load_reviews()
    print(f"loaded {len(reviews):,} reviews")

    if args.mode == "first":
        chosen = select_first(reviews, args.n)
        print(f"mode=first: rows 0..{args.n - 1} of the file in order")
    else:
        chosen = select_balanced(reviews, args.per_class, args.seed, classes)
        dist = collections.Counter(r["rating"] for _, r in chosen)
        cls_dist = collections.Counter(rating_to_class(r["rating"], classes) for _, r in chosen)
        print(f"mode=balanced: seed={args.seed}, per_class={args.per_class} (per class)")
        print(f"  class mix={dict(sorted(cls_dist.items()))}, rating mix={dict(sorted(dist.items()))}")

    # Sanity check for the record: the rating must never reach the model.
    # classify_review() only ever receives r["title"] and r["text"].

    out_path = f"output/runs/{args.name}.jsonl"
    os.makedirs("output/runs", exist_ok=True)
    print(f"classifying {len(chosen)} reviews ({'w/ emotion' if args.emotions else '2-class'})…")

    records, failures = [], 0
    t0 = time.time()
    for pos, (i, r) in enumerate(chosen):
        pred = classify_review(r["title"], r["text"],
                               classes=classes, include_emotion=args.emotions)
        record = {
            "index": i,
            "correct_class": rating_to_class(r["rating"], classes),
            **{k: r[k] for k in KEEP_FIELDS},
            "classes_mode": "3-class" if args.classes == 3 else "2-class",
            **pred,
        }
        records.append(record)
        failures += 1 if pred["parse_fail"] else 0
        if (pos + 1) % 25 == 0 or pos + 1 == len(chosen):
            dt = time.time() - t0
            print(f"  {pos + 1}/{len(chosen)}  ({dt:.0f}s elapsed, "
                  f"{(pos + 1) / dt:.1f} reviews/s)")
    t1 = time.time()
    print(f"done in {t1 - t0:.0f}s; parse failures: {failures}")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"saved {out_path}")

    summary = summarize(records, classes)
    with open(f"output/runs/{args.name}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"run": args.name, "mode": args.mode, "n": len(records),
                   "classes": list(classes), "emotions": args.emotions,
                   "seed": args.seed, "per_class": args.per_class, **summary},
                  f, indent=2)
    print_summary(summary, classes)
    return 0


def summarize(records, classes):
    classes = tuple(classes)
    n = len(records)
    parsed = [r for r in records if not r["parse_fail"]]
    total_ok = len(parsed)

    conf = collections.defaultdict(collections.Counter)  # correct -> pred
    for r in parsed:
        conf[r["correct_class"]][r["sentiment_pred"]] += 1

    per_class_recall = {}
    for c in classes:
        denom = sum(conf[c].values())
        per_class_recall[c] = {
            "n": denom,
            "correct": conf[c].get(c, 0),
            "recall": (conf[c].get(c, 0) / denom) if denom else None,
        }

    correctness = {c: sum(conf[c].get(p, 0) for p in (c,)) for c in classes}
    accuracy = total_ok and (sum(correctness.values()) / total_ok) or 0.0

    return {
        "n_total": n,
        "n_parsed": total_ok,
        "n_parse_fail": n - total_ok,
        "accuracy": accuracy,
        "agreement": int(sum(correctness.values())),
        "confusion": {c: dict(conf[c]) for c in classes},
        "per_class_recall": per_class_recall,
        "correct_class_dist": dict(collections.Counter(r["correct_class"] for r in parsed)),
    }


def print_summary(summary, classes):
    print("\n=== summary ===")
    print(f"rows: {summary['n_total']}  (parsed {summary['n_parsed']}, "
          f"parse_fail {summary['n_parse_fail']})")
    print(f"accuracy vs rating: {summary['accuracy']:.1%}  "
          f"({summary['agreement']}/{summary['n_parsed']} agree)")
    print("rows by correct class:", summary["correct_class_dist"])
    print("\nconfusion (rows: correct -> predicted)")
    for c in tuple(classes):
        print(f"  {c:9s} -> {summary['confusion'][c]}")
    print("\nper-class recall")
    for c in tuple(classes):
        v = summary["per_class_recall"][c]
        r = f"{v['recall']:.1%}" if v["recall"] is not None else "n/a"
        print(f"  {c:9s} {v['correct']}/{v['n']} = {r}")


if __name__ == "__main__":
    sys.exit(main())
