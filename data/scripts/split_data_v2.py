"""
Split the NL2Bash dataset into train, dev, and test sets.

Reproduces the logic from the original split_data.py but without the
bashlint/nlp_tools dependencies (which are broken on Python 3.10+).

Key properties preserved from the original:
  1. Pairs are grouped by normalized NL description before splitting,
     so paraphrases of the same intent stay in the same split.
  2. Dev/test pairs whose bash command also appears in train are moved
     to train (no command leakage across splits).
  3. Split ratio is 10:1:1 (train:dev:test), matching the original.
  4. Uses the same random seed (100) and 12-fold assignment scheme.

Usage:
    python data/scripts/split_data_v2.py [--data-dir data/bash]

Outputs:
    data/bash/train.csv
    data/bash/dev.csv
    data/bash/test.csv
"""

import argparse
import collections
import csv
import os
import random
import re


RANDOM_SEED = 100
NUM_FOLDS = 12


def normalize_nl(nl: str) -> str:
    """Simplified NL normalization for grouping.

    Approximates the original basic_tokenizer(nl) by lowercasing,
    stripping punctuation, and collapsing whitespace.  This is used
    only to group near-duplicate descriptions so they end up in the
    same split — the raw text is preserved in the output files.
    """
    nl = nl.lower().strip()
    nl = re.sub(r"\([^)]*\)", "", nl)       # remove parenthesized content
    nl = re.sub(r"[,;:.\-\"'`]", " ", nl)   # remove common punctuation
    nl = re.sub(r"\s+", " ", nl).strip()     # collapse whitespace
    return nl


def load_parallel_file(path: str):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def split_data(data_dir: str) -> None:
    nl_path = os.path.join(data_dir, "all.nl")
    cm_path = os.path.join(data_dir, "all.cm")

    nl_lines = load_parallel_file(nl_path)
    cm_lines = load_parallel_file(cm_path)
    assert len(nl_lines) == len(cm_lines), (
        f"Mismatched line counts: {len(nl_lines)} NL vs {len(cm_lines)} CM"
    )
    print(f"Loaded {len(nl_lines)} pairs from {data_dir}")

    # ── Group pairs by normalized NL template ─────────────────────
    pairs: dict[str, list[tuple[str, str]]] = collections.OrderedDict()
    for nl, cm in zip(nl_lines, cm_lines):
        key = normalize_nl(nl)
        if key not in pairs:
            pairs[key] = []
        pairs[key].append((nl, cm))

    num_groups = len(pairs)
    print(f"Unique NL template groups: {num_groups}")

    # ── Assign each group to a fold ───────────────────────────────
    random.seed(RANDOM_SEED)
    random_tokens = [random.randint(0, NUM_FOLDS - 1) for _ in range(num_groups)]

    train_pairs: list[tuple[str, str]] = []
    dev_pairs: list[tuple[str, str]] = []
    test_pairs: list[tuple[str, str]] = []

    for i, key in enumerate(sorted(pairs.keys())):
        ind = random_tokens[i]
        if ind < NUM_FOLDS - 2:          # folds 0-9  → train
            train_pairs.extend(pairs[key])
        elif ind == NUM_FOLDS - 2:       # fold 10    → dev
            dev_pairs.extend(pairs[key])
        elif ind == NUM_FOLDS - 1:       # fold 11    → test
            test_pairs.extend(pairs[key])

    print(f"Before dedup — Train: {len(train_pairs)} | Dev: {len(dev_pairs)} | Test: {len(test_pairs)}")

    # ── Move dev/test pairs whose command appears in train ────────
    train_commands = {cm for _, cm in train_pairs}

    dev_clean: list[tuple[str, str]] = []
    moved = 0
    for nl, cm in dev_pairs:
        if cm in train_commands:
            train_pairs.append((nl, cm))
            moved += 1
        else:
            dev_clean.append((nl, cm))
    print(f"Moved {moved} dev pairs to train (command overlap)")

    test_clean: list[tuple[str, str]] = []
    moved = 0
    for nl, cm in test_pairs:
        if cm in train_commands:
            train_pairs.append((nl, cm))
            moved += 1
        else:
            test_clean.append((nl, cm))
    print(f"Moved {moved} test pairs to train (command overlap)")

    # ── Write CSV files ───────────────────────────────────────────
    def write_csv(path: str, data: list[tuple[str, str]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["natural_language", "bash_command"])
            for nl, cm in data:
                writer.writerow([nl, cm])
        print(f"  Wrote {len(data):>6} pairs → {path}")

    print()
    write_csv(os.path.join(data_dir, "train.csv"), train_pairs)
    write_csv(os.path.join(data_dir, "dev.csv"), dev_clean)
    write_csv(os.path.join(data_dir, "test.csv"), test_clean)

    total = len(train_pairs) + len(dev_clean) + len(test_clean)
    print(f"\nTotal: {total} | Train: {len(train_pairs)} | Dev: {len(dev_clean)} | Test: {len(test_clean)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split NL2Bash into train/dev/test")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bash"),
        help="Directory containing all.nl and all.cm (default: nl2bash/data/bash/ relative to script)",
    )
    args = parser.parse_args()
    split_data(args.data_dir)
