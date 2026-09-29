# MBAX 6418 — Assignment 1: Sentiment & Emotion Classification of Amazon Gift-Card Reviews

For this project I built a sentiment classifier for Amazon's "Gift Cards" review category. The model reads a review's title and text and classifies it as **POSITIVE, NEUTRAL, or NEGATIVE**. It also names the review's primary emotion, and I compare that against a second, completely different take: a word list that adds up how many emotion words appear in the review. Every prediction is scored against the reviewer's star rating, which the model never sees. All of the results are presented in a single HTML dashboard (`dashboard.html`) that works offline, so anyone can open it and check the numbers themselves.

**The short version of what I found:** on a balanced sample of 150 reviews, the model agrees with the star rating **72.0%** of the time (108/150). That number matters because the data is heavily skewed — 88.5% of the category is 4–5 stars — so a model that labels everything positive would score 93% on an ordinary batch without reading a single word. When I sampled equal numbers of each class, that naive baseline dropped to 33.3% and the model's real performance came through: it gets positive reviews right 100% of the time and negative reviews right 98% of the time. The interesting failure is in the middle. 3-star reviews collapse into NEGATIVE (33 out of 50), even though the rating scale gives them their own class. The LLM and the word list also pick the same primary emotion only 17.7% of the time, and the reason for that is more important than the number itself.

![Dashboard](assets/dashboard_final.png)

*The dashboard. It is a single self-contained HTML file: open it, filter the table (All / Matched / Mismatched, class buttons, search), click any row for the full text and the model's raw output, and hover the red emotion chips for an explanation.*

---

## 1. The data

The task uses the Amazon 2023 **"Gift Cards"** review category, part of the large-scale Amazon Reviews '23 dataset collected by the McAuley Lab at UC San Diego:

- Dataset page: <https://amazon-reviews-2023.github.io>
- Raw review file (gzipped JSON Lines): `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz`

The file contains **152,410 reviews**. Before building anything I made sure I could actually read it.

The star-rating distribution is very lopsided:

| Rating | Count | Share |
|---|---:|---:|
| ★1 | 12,326 | 8.1% |
| ★2 | 1,873 | 1.2% |
| ★3 | 3,271 | 2.1% |
| ★4 | 6,692 | 4.4% |
| ★5 | 128,248 | **84.1%** |

As rating-derived classes: **POSITIVE 88.5%, NEUTRAL 2.1%, NEGATIVE 9.3%**. I kept this imbalance in mind the whole way through, because it is the reason the early numbers have to be read carefully (Section 3.1).

From each review I kept the fields that seemed useful: `rating` (ground truth only, never sent to the model), `title`, `text`, `verified_purchase`, `helpful_vote`, and `timestamp`. The raw file is not committed to this repo — it is large and re-downloadable — `score.py` reads it from `data/Gift_Cards.jsonl.gz`.

## 2. How I built it

The whole pipeline is Python with zero external packages, so it runs anywhere. The flow is:

```
data/Gift_Cards.jsonl.gz
   │  score.py         samples reviews (first N, or balanced by a fixed seed),
   │                   classifies them through the LLM, saves raw rows + metrics
   ▼
output/runs/<run>.jsonl
   │  score_emotions.py  adds the NRC word-list emotion take (no model calls)
   ▼
output/runs/<run>_emotions.jsonl
   │  build_dashboard.py  recomputes every number from the saved rows and embeds
   │                      them with the raw rows into one HTML file
   ▼
dashboard.html
```

**The prompt (`prompt.py`).** A parameterized instruction that takes only `title` and `text` and demands a strict JSON answer (`{"sentiment": "...", "emotion": "..."}`). The rating never appears anywhere in the prompt. I decided the edge cases up front and wrote them into the prompt text: when the title and body conflict, trust the body; terse reviews ("Junk") still carry sentiment; sarcasm counts; in two-class mode there is no neutral option, and in three-class mode NEUTRAL is reserved for reviews that are genuinely flat; the emotion has to be one of the NRC's 8 words, verbatim. A spot-check on obviously positive and negative reviews, including the tricky ones, came out right before I scored anything real.

**The endpoint (`llm.py`).** The class-provided OpenAI-compatible endpoint (`http://dobolyi.com:9001/v1`, model `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`). Two quirks came up in setup and are handled in the client:

1. The model is a *reasoning* model, so every answer was spending ~490 tokens on a hidden thinking pass before replying. I disabled thinking (`chat_template_kwargs: {"enable_thinking": false}`), which cut each classification from ~500 to **~9 completion tokens** and changed nothing in the answers.
2. `response_format: {"type": "json_object"}` forces machine-readable JSON, and a tolerant fallback parser catches stray replies. Rows that still cannot be parsed are saved and flagged (`parse_fail`), never silently dropped.

**The word list (`score_emotions.py`).** The NRC emotion lexicon (Mohammad & Turney, 2013) links words to 8 emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust. Each review's words are looked up, the scores per emotion are added up, and the highest total wins. I made three decisions here and documented them in the code: no stemming (the lexicon already contains inflections like happy/happier/happiest), no negation handling (a deliberate blindness — "not happy" still scores joy), and ties broken by the first emotion in a fixed order and **flagged** (`wordlist_tie`) instead of hidden. Reviews with no matching words simply get no word-list emotion. The lexicon is too big to commit comfortably, so the script downloads it to `data/` on first use and caches it; the source is in Section 6.

## 3. What I found

### 3.1 Why the first run looked so good, and what balancing changed

**Step 2** scored the first 100 rows in file order, two-class (≥4★ positive, else negative). It looked great:

| Run | Accuracy | Naive baseline | Model edge |
|---|---:|---:|---:|
| First 100 rows (imbalanced) | **98.0%** (98/100) | 93.0% ("always POSITIVE") | +5.0 pts |
| Balanced 150 (seed 42, 3-class) | **72.0%** (108/150) | 33.3% ("always one class") | +38.7 pts |

The first run's 98% is mostly an illusion of the data: 93 of those 100 rows are 4–5 stars, so a classifier that prints "positive" without reading anything scores 93%. The model only adds about 5 points over that. When I sampled 50 reviews from each class across the whole file (50/50/50, 150 rows, fixed random seed 42, so the same rows come up every time), the baseline fell to 33.3% and the accuracy to an honest 72.0% — but the model's real edge is now 38.7 points, not 5. The balanced run is doing actual work; the lopsided run was hiding that. The dashboard shows the same story visually: ★5 is 84.1% of the population but only 32% of the balanced sample.

### 3.2 Where the mistakes go

The balanced three-class confusion matrix (rows = the rating-derived correct answer, columns = what the model predicted), 150 rows, 0 parse failures:

| correct → predicted | POSITIVE | NEUTRAL | NEGATIVE | recall |
|---|---:|---:|---:|---:|
| **POSITIVE** (★4–5) | **50** | 0 | 0 | 100% |
| **NEUTRAL** (★3) | 8 | **9** | **33** | 18% |
| **NEGATIVE** (★1–2) | 0 | 1 | **49** | 98% |

Two patterns stand out:

- **3-star reviews collapse into NEGATIVE, not the other way around.** 33 of the 50 neutral reviews were labeled NEGATIVE; only 8 were called POSITIVE and 9 were correctly NEUTRAL. Meanwhile, negative reviews almost never get called neutral (1 of 50). So the confusion is one-directional. Reading the actual texts explains why: "don't get change if you get a 50 dollar card", "Why the hell we only have an option with free state tax", "Bad Experience". 3-star writers complain, and the prompt told the model that a review leaning at all gets its leaning label. The rating calls ★3 NEUTRAL by definition; the text often reads as negative. That tension is the root cause, not a bug in the model.
- **The model never undercuts a positive review** (0 of 50 positives mislabeled) and almost never misses a negative one. Its failures are almost entirely in the middle class.

The two mismatches in the Step-2 run pointed at this before the balanced run existed: ★3 "Easy to use" ("Very easy to use. I wish I knew about it earlier" — the text reads positive, but two-class truth forced NEGATIVE on it) was exactly the class-boundary problem that the balanced run went on to quantify.

### 3.3 Why the LLM and the word list disagree about emotions

Both takes pick from the same 8 NRC emotions, so the comparison is fair. They agree on the primary emotion only **23.5%** of the time (20/85) on the 100-row run and **17.7%** (22/124) on the balanced run. The disagreement is structural:

| | LLM (balanced run) | Word list (balanced run) |
|---|---:|---:|
| joy | 49 | 16 |
| **anticipation** | **1** | **72** |
| trust | 17 | 11 |
| anger | 56 | 10 |
| disgust | 13 | 1 |
| sadness / surprise / fear | 14 | 14 |

- **The word list counts vocabulary, not meaning.** The word "gift" appears in 99 of the first 100 reviews, and in the NRC list it maps to anticipation, joy, and surprise at the same time. So almost every review starts with a base score in all three, and 52 of the first 100 reviews end in a tie; the documented tie-break favors anticipation — that is why the word list says anticipation 72 times while the LLM says it once. The word list also cannot read a complaint. The ★1 "Not $10 Gift Cards" review — a card that had $6.52 on it instead of $10 — scores 8 joy hits from words like *gift, friends, daughter*, while the LLM reads the situation and says anger. 15 of the 100 rows have zero NRC words at all, so the word list has no answer where the LLM always has one.

The contradiction counts make the difference concrete: **54** word-list emotions contradict the review's own sentiment (joy/trust/anticipation on 1–2★ complaints, anger/disgust on a few 5★ reviews) versus **2** for the LLM, both "surprise" on negative reviews ("GRAND but sukish"). The dashboard marks every contradicted emotion chip in red with a tooltip so a reader can see the failure without drilling in. In the end, the LLM reads what happened in the review; the word list just describes the topic. Gift-card reviews are full of positive words no matter what went wrong.

### 3.4 Bugs I hit along the way

- **The model was thinking out loud.** The endpoint's Qwen3 model spent ~490 tokens "thinking" before each answer. I discovered this when classifications came back empty — the whole token budget had gone to reasoning. Disabling thinking via `chat_template_kwargs` made each review ~50× cheaper, and the answers were identical (verified by re-running 100 reviews).
- **Emotions outside the vocabulary.** The model answered "relief" and "happiness" — neither is one of the NRC's 8 emotions. I fixed it two ways: the prompt now demands one of the 8 words verbatim, and a documented synonym map (`relief → joy`, etc.) normalizes whatever still slips through, so both takes live in the same emotion space. Anything unmapped is recorded as `emotion_missing`, not dropped.
- **A save-path crash.** The scorer tried to write to `output/runs/` before the folder existed, and crashed after 100 successful classifications with all the work lost. One `os.makedirs` fixed it.
- **Three JavaScript bugs in the dashboard**, found by re-reading the code and then re-rendering in a browser: a duplicated header (a `firstChild` + `children` mistake), two legends that vanished (an `append()` result used as if it were an element), and a live-count update that threw because it ran before its section was attached to the page. The last one is the kind of bug that makes a page look fine in a screenshot while it is actually broken — it only showed up in a DOM probe.
- **A literal `<small>` tag** printed as text in a KPI card because one helper sets textContent rather than HTML. Cosmetic, but it looked broken.
- **A float-vs-int key bug in the new population chart.** The sample-vs-population rating bars showed "—" for every population row: the population side had counted star ratings as floats (`1.0`) while the sample side used ints (`1`). I only caught it because Step 7 requires checking the numbers in the browser rather than by eye — the check compared every rendered value against the saved output and this one failed the first time.
- **Zero-width bar risk.** The assignment warned that small chart elements can collapse to zero width. Every bar gets a minimum width plus an always-visible count label, and a layout audit measured all 35 bars on the final page: none zero-width, none overflowing.

The lesson I took from these: screenshots are not verification. Every number on the final page was pulled from the rendered DOM and compared programmatically against a fresh recompute of the saved run output, and that process is exactly what caught the population chart bug.

## 4. Can this be reproduced?

Yes, and I checked rather than assumed:

- **Fixed seed.** The balanced sample uses `random.Random(42)`; re-running selects the identical 150 rows (verified against the saved run's row indices).
- **Fixed settings.** Temperature 0, thinking disabled, deterministic endpoint. The Step-2 and Step-5 runs produced identical sentiment labels on all 100 shared rows.
- **Deterministic word list.** Re-deriving emotions over both saved runs reproduced every row exactly (0/250 differences).
- **Deterministic dashboard.** Regenerating `dashboard.html` from the same run is byte-identical (matching sha256).

Every number in this report comes from `output/runs/step6_balanced3_emotions.jsonl` (the balanced run's raw output, committed to this repo) and its summaries. The dashboard recomputes from the same file, so the page and this report cannot drift apart.

## 5. How to run

```bash
# 0. (once) download the data into data/; let the NRC lexicon auto-download
#    (score_emotions.py). Nothing to pip install — standard library only.

# Step-2 style run: first 100 rows, two-class
python3 score.py --mode first --n 100 --classes 2 --name step2_first100

# Step 5/6 run: balanced 50 per class, fixed seed 42, three-class + emotions
python3 score.py --mode balanced --per-class 50 --seed 42 --classes 3 \
                 --emotions --name step6_balanced3

# word-list emotions + LLM-vs-word-list comparison
python3 score_emotions.py --run step6_balanced3

# regenerate the dashboard from a saved run (final dashboard):
python3 build_dashboard.py --run step6_balanced3_emotions -o dashboard.html
open dashboard.html            # self-contained, works offline
```

Runs land in `output/runs/` as JSON Lines (one review per line: rating, title, text, correct class, predictions, and the model's raw output) plus a `*_summary.json` with the metrics. The endpoint, key, and model name are config constants at the top of `llm.py`. The data download is the McAuley Lab URL from Section 1, saved as `data/Gift_Cards.jsonl.gz`.

## 6. Credits

- **Data:** the Amazon Reviews '23 dataset, "Gift Cards" category — McAuley Lab, UC San Diego. Dataset page: <https://amazon-reviews-2023.github.io>. Raw review file hosted on the McAuley Lab public dataset server.
- **Emotion lexicon:** Mohammad, S. M., & Turney, P. D. (2013). *Crowdsourcing a Word-Emotion Association Lexicon* (NRC Emotion Lexicon v0.92), National Research Council Canada. The word-level file is downloaded from a public mirror of the lexicon, since the original distribution requires a request: `github.com/dinbav/LeXmo`.
- **Model:** `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, served on the class OpenAI-compatible endpoint (vLLM).
- Built with Python 3.13, standard library only; the dashboard is hand-built HTML/CSS/JS with no external libraries.
