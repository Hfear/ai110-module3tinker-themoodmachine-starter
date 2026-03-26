# Model Card: Mood Machine

This model card covers **both versions** of the Mood Machine classifier:

1. A **rule-based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

---

## 1. Model Overview

**Model type:**
Both models were built and compared. The rule-based model was the primary focus; the ML model was trained on the same dataset to observe differences in behavior.

**Intended purpose:**
Classify short text posts (social media-style messages) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**
- *Rule-based:* Each post is lowercased and split into tokens. Each token is compared against a list of positive and negative words. Tokens preceded by a negation word (e.g., "not", "never") flip the sentiment of the word that follows. A numeric score is totaled and mapped to a label using thresholds.
- *ML model:* Posts are converted to bag-of-words vectors using `CountVectorizer`. A `LogisticRegression` classifier is trained on those vectors and the human-assigned labels from `TRUE_LABELS`.

---

## 2. Data

**Dataset description:**
The dataset contains 22 posts in `SAMPLE_POSTS`. The original 6 starter posts were kept as-is; 16 new posts were added to cover a wider range of language styles, including a dedicated set of posts designed to test the "cooked" pronoun-context rule.

**Labeling process:**
Labels were assigned by reading each post and choosing the label that best matched its overall tone. Several posts were genuinely difficult to label:

- `"just got the job offer 😭😭😭"` — crying emojis are commonly used to express joy, so labeled `positive`, but a model that treats 😭 as sadness would label it `negative`.
- `"I'm fine 🙂"` — the words say "fine" (neutral) but the understated emoji can imply sarcasm or suppressed sadness. Labeled `neutral` as a conservative choice.
- `"I absolutely love sitting in traffic for two hours"` — sarcasm. The word "love" is positive, but the intent is clearly negative. Labeled `negative`.

**Important characteristics of the dataset:**

- Contains informal slang: "lowkey", "highkey", "no cap", "fire", "sick", "eats", "cooked", "flop"
- Includes emojis: 🔥, 😭, 💀, 🙂
- Includes two clear sarcasm examples ("I love falling", "I love how you don't care about being good")
- Several posts express genuinely mixed emotions
- Short, fragmented phrases (typical of texting/social media)
- Includes context-dependent slang where the same word ("cooked") means opposite things based on grammar

**Possible issues with the dataset:**

- 22 examples is very small. The ML model trains and evaluates on the same data, so its accuracy is not meaningful for generalization.
- The dataset skews toward English-language Gen Z slang; it would not generalize to other dialects or languages.
- Several labels (e.g., "I'm fine 🙂") could reasonably be argued either way.
- The two sarcasm posts are labeled `negative` but the rule-based model will predict `mixed` or `positive` — these are documented failures, not data errors.

---

## 3. How the Rule-Based Model Works

**Scoring rules:**

- Each token is looked up in `POSITIVE_WORDS` or `NEGATIVE_WORDS` (sets for fast lookup).
- Positive match → score +1. Negative match → score -1.
- **Negation handling:** If the token immediately before a sentiment word is a negation word (`not`, `never`, `no`, `don't`, `doesn't`, `didn't`, `isn't`, `wasn't`, `can't`, `won't`, `nor`), the sentiment is flipped. So "not happy" subtracts 1 instead of adding 1.
- **Slang/pronoun-context rule for "cooked" (primary chosen enhancement):** The word "cooked" is not in either word list because its meaning depends on grammatical context. Instead, the scorer scans all tokens before "cooked" in the sentence:
  - If a contraction like `i'm`, `he's`, `she's`, `they're`, `we're` is found → score -1 ("I'm cooked" = in trouble, stressed, failing)
  - If a bare pronoun like `i`, `he`, `she`, `they`, `we` is found → score +1 ("I cooked" = succeeded, did well)
  - If neither is found → "cooked" contributes nothing to the score

**The "cooked" rule in detail:**

This rule exists because "cooked" is a classic example of context-dependent slang. The same word carries opposite meaning based solely on whether the subject uses a contraction or not:

| Sentence | Subject | Meaning | Score change |
|----------|---------|---------|-------------|
| "I'm cooked" | contraction | in trouble, overwhelmed | -1 |
| "I cooked" | bare pronoun | did great, succeeded | +1 |
| "I'm highkey cooked" | contraction | definitely in trouble | -1 |
| "I lowkey cooked" | bare pronoun | subtly did really well | +1 |

**Known failure case for the "cooked" rule:**
"I got cooked" — "I" is a bare pronoun, so the rule predicts positive (+1 → `mixed`), but the intended meaning is negative (someone beat you badly). The rule breaks when an auxiliary verb ("got") separates the pronoun from "cooked" and changes the meaning. This is a documented limitation.

**Label thresholds:**

| Score | Label |
|-------|-------|
| ≥ 2 | `positive` |
| ≤ -2 | `negative` |
| +1 or -1 | `mixed` |
| 0 | `neutral` |

The threshold of ±2 was chosen because a single matched word is a weak signal — requiring two or more positive/negative word hits before committing to a strong label reduces false positives.

**Strengths:**

- Transparent and inspectable: you can trace exactly which words affected the score.
- Negation handling correctly flips "I am not happy about this" to `negative`.
- Slang words ("fire", "sick", "lowkey", "eats", "flop") and emoji tokens ("🔥", "💀") added to the word lists give the model some coverage of informal language.
- The "cooked" pronoun-context rule correctly handles the contraction vs. bare-pronoun distinction for most direct uses of the word.

**Weaknesses:**

- Sarcasm is invisible to it. "I love falling" and "I love how you don't care about being good" both score positive (from "love") and get labeled `mixed` or `positive` instead of `negative`.
- The "cooked" rule breaks for indirect constructions like "I got cooked" — "I" triggers the bare-pronoun path, predicting positive when the meaning is actually negative.
- Context beyond adjacent negation is ignored. "I never thought this could be so terrible" handles the negation of "terrible" correctly, but more complex sentence structure breaks it.
- Unknown slang or new emoji not in the word lists are silently ignored.
- The mixed label threshold is arbitrary — a single word hit always triggers `mixed` regardless of the rest of the sentence.

---

## 4. How the ML Model Works

**Features used:**
Bag-of-words representation via `CountVectorizer`. Each post becomes a vector of word counts across the full vocabulary of the training set.

**Training data:**
Trained on all 22 posts in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**
Because training and evaluation use the same 16 posts, the model can effectively memorize the training data. Accuracy reported by `ml_experiments.py` reflects training accuracy, not generalization. Adding more posts with consistent labels improved stability; adding ambiguously-labeled posts (e.g., flipping a "neutral" to "mixed") caused more variation in predictions.

**Strengths and weaknesses:**

- *Strength:* Learns patterns from data automatically — does not require manually crafting word lists.
- *Weakness:* With only 22 examples, the model cannot learn meaningful generalizations. It memorizes rather than learns.
- *Weakness:* `CountVectorizer` treats every word as an independent feature and ignores word order entirely, so "not happy" is treated the same as "happy not."
- *Difference from rule-based:* The ML model sometimes misclassifies examples the rule-based model gets right (e.g., when it latches onto a rare word that happened to appear in training) and vice versa.

---

## 5. Evaluation

**How the model was evaluated:**
Both models were evaluated against `TRUE_LABELS` on the full 22-post dataset. This is in-sample evaluation — neither model has been tested on unseen data.

**Examples of correct predictions (rule-based):**

| Post | Predicted | True | Why correct |
|------|-----------|------|-------------|
| "no cap that concert was fire 🔥" | positive | positive | "fire" → +1, "🔥" → +1; score = +2 → `positive` ✓ |
| "I am not happy about this" | negative | negative | negation flips "happy": score = -1 → `mixed`... actually -1 only. **See note below.** |
| "I lowkey cooked" | positive | positive | "lowkey" → +1, cooked rule (bare pronoun "i") → +1; score = +2 → `positive` ✓ |

**Examples of incorrect predictions (rule-based):**

| Post | Predicted | True | Why wrong |
|------|-----------|------|-----------|
| "I love this class so much" | mixed | positive | Only one positive hit ("love") → score = 1 → `mixed`. The ±2 threshold is too strict for short single-signal posts. |
| "I love falling" | mixed | negative | Sarcasm. "love" → +1, no negation → score = +1 → `mixed`. Model has no way to detect irony. |
| "I love how you don't care about being good" | mixed | negative | Sarcasm again. "love" +1, negation flips "good" to -1, net = 0 → `neutral`. Still wrong for different reasons. |
| "I'm highkey cooked" | mixed | negative | "highkey" → +1 (positive word), cooked rule (contraction "i'm") → -1; net score = 0 → `neutral`. The intensifier cancels the cooked rule. |

**Key takeaway:**
The ±2 threshold reduces false positives on genuinely ambiguous posts, but causes single-signal posts to fall into `mixed` instead of a clear label. Sarcasm is the hardest failure mode — the model reads "I love falling" as positive because it has no understanding of ironic intent. The "cooked" rule works for direct uses ("I'm cooked", "I cooked") but struggles when an intensifier like "highkey" sits between the pronoun and the word.

---

## 6. Limitations

- **Dataset size:** 22 posts is far too small for meaningful evaluation or ML training.
- **No held-out test set:** Both models are evaluated on training data, so accuracy numbers do not reflect real-world performance.
- **No sarcasm detection:** The rule-based model has no mechanism for recognizing irony. Sarcasm is a hard problem even for large language models.
- **Threshold rigidity:** A score of +1 always maps to `mixed`, even if the post is clearly expressing happiness in a short phrase.
- **Slang coverage is partial:** Only specific slang terms in the word list are recognized. New slang or regional expressions are silently ignored.
- **Emoji coverage is partial:** Only a handful of emoji tokens were added. Most emoji — and crucially, context-dependent emoji like 😭 (which can mean happy or sad) — are not handled.
- **Language scope:** The dataset is English-only and skews toward a specific register of informal American English.

---

## 7. Ethical Considerations

- **Misclassifying distress:** A post like "I'm fine 🙂" — which might mask real distress — gets labeled `neutral`. If this system were used in a mental health or moderation context, false negatives on distressed posts could be harmful.
- **Cultural and linguistic bias:** The word lists and training examples reflect a narrow slice of English. Expressions of mood common in other languages, dialects, or communities would be missed or misread.
- **Label subjectivity:** Human labels contain personal judgment. Two labelers might disagree on ambiguous posts. The model inherits whatever biases the labeler introduced.
- **Privacy:** Analyzing personal messages without consent raises privacy concerns, even with a simple bag-of-words model.
- **Automation risk:** A low-accuracy system used to make automated decisions (e.g., flagging content, routing support tickets) could harm users if errors are not surfaced to a human reviewer.

---

## 8. Ideas for Improvement

- **More labeled data:** Even 200–500 examples would dramatically improve ML model generalization.
- **Real train/test split:** Evaluate on posts the model has never seen.
- **TF-IDF instead of CountVectorizer:** Down-weights very common words, giving more signal to meaningful terms.
- **Negation-aware ML features:** Add bigrams (e.g., "not_happy" as a single feature) so the ML model can learn from negation patterns.
- **Emoji lookup table:** Map individual emoji to sentiment scores rather than treating them as unknown tokens.
- **Sarcasm detection heuristics:** Flag sentences that contain positive words with negative context markers (e.g., "love" + "stuck", "traffic", "waiting").
- **Transformer-based model:** A small pre-trained model (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) would handle sarcasm, context, and nuance far better than bag-of-words approaches.
- **Confidence scores:** Instead of hard labels, output a probability distribution so downstream systems can flag low-confidence predictions for human review.
