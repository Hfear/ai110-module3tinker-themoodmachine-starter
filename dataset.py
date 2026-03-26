"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # slang / informal positives
    "fire",
    "lit",
    "sick",
    "lowkey",
    "highkey",
    "gorgeous",
    "loved",
    "eats",
    "W",
    "slay",
    # emoji signals (as text tokens)
    "🔥",
    "😂",
    "😊",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # slang / informal negatives
    "ugh",
    "missed",
    "spilled",
    "late",
    "flop",
    "mid",
    "L",
    # emoji signals
    "💀",
    "😤",
    "😒",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # --- added posts ---
    "lowkey stressed but this sunset is gorgeous",          # slang + mixed
    "I absolutely love sitting in traffic for two hours",   # sarcasm → negative
    "just got the job offer 😭😭😭",                          # emoji ambiguity (crying = happy)
    "this homework is fine I guess whatever",               # flat / neutral-ish
    "no cap that concert was fire 🔥",                       # slang + positive
    "sick beat but the lyrics are kinda sad",               # mixed
    "woke up late, missed the bus, spilled my coffee 💀",   # negative
    "not bad for a Monday honestly",                        # negation + mild positive
    "highkey loved every second of that movie",             # slang + positive
    "I'm fine 🙂",                                           # ambiguous — emoji masks negativity
    # --- cooked / slang context posts ---
    "This eats down",                                        # slang positive
    "I lowkey cooked",                                       # slang + pronoun-context positive
    "I'm highkey cooked",                                    # contraction-context negative
    "Was a tremendous flop",                                 # negative slang
    "I love falling",                                        # sarcasm → negative
    "I love how you don't care about being good",            # sarcasm → negative
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # --- added labels ---
    "mixed",     # lowkey stressed but this sunset is gorgeous
    "negative",  # I absolutely love sitting in traffic... (sarcasm)
    "positive",  # just got the job offer 😭😭😭 (happy-cry emojis)
    "neutral",   # this homework is fine I guess whatever
    "positive",  # no cap that concert was fire 🔥
    "mixed",     # sick beat but the lyrics are kinda sad
    "negative",  # woke up late, missed the bus, spilled my coffee 💀
    "positive",  # not bad for a Monday honestly
    "positive",  # highkey loved every second of that movie
    "neutral",   # I'm fine 🙂 — emoji masks true feeling; labeled neutral
    # --- cooked / slang context labels ---
    "positive",  # This eats down
    "positive",  # I lowkey cooked
    "negative",  # I'm highkey cooked
    "negative",  # Was a tremendous flop
    "negative",  # I love falling (sarcasm)
    "negative",  # I love how you don't care about being good (sarcasm)
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
