import csv
import os
import random

from config import RANDOM_STATE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

CSV_FILE = os.path.join(ROOT_DIR, "dataset", "diamonds.csv")
PROLOG_FILE = os.path.join(BASE_DIR, "facts_and_rules.pl")

CATEGORICAL_COLS = ["cut", "color", "clarity"]
COLS = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"]

START_MARKER = "% ===== GENERATED FACTS START ====="
END_MARKER = "% ===== GENERATED FACTS END ====="


def format_value(col, value):
    value = value.strip()
    if col in CATEGORICAL_COLS:
        return value.lower().replace(" ", "_")
    return value


def generate_facts(rows):
    lines = []
    for idx, row in enumerate(rows, start=1):
        diamond_id = f"diamond_{idx}"
        for col in COLS:
            lines.append(f"prop({diamond_id}, {col}, {format_value(col, row[col])}).")
    return lines


def build_facts_block(rows):
    return [START_MARKER] + generate_facts(rows) + [END_MARKER]


def insert_facts(prolog_path, facts_block):
    with open(prolog_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    start_idx = next((i for i, l in enumerate(lines) if l.strip() == START_MARKER), None)
    end_idx = next((i for i, l in enumerate(lines) if l.strip() == END_MARKER), None)

    if start_idx is not None and end_idx is not None:
        new_lines = lines[:start_idx] + facts_block + lines[end_idx + 1:]
    else:
        comment_start = next((i for i, l in enumerate(lines) if l.strip() == "/*"), None)
        comment_end = next((i for i, l in enumerate(lines) if l.strip() == "*/"), None)
        if comment_start is not None and comment_end is not None and comment_end > comment_start:
            new_lines = lines[:comment_start] + facts_block + lines[comment_end + 1:]
        else:
            new_lines = lines + facts_block

    with open(prolog_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(new_lines) + "\n")


def execute_insert_facts(num_diamonds: int=500) -> None:
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if num_diamonds > len(rows):
        num_diamonds = len(rows)

    rng = random.Random(RANDOM_STATE)
    sampled_rows = rng.sample(rows, num_diamonds)

    facts_block = build_facts_block(sampled_rows)
    insert_facts(PROLOG_FILE, facts_block)

    total_facts = num_diamonds * len(COLS)
    print(f"Scritti {total_facts} fatti per {num_diamonds} diamanti in {PROLOG_FILE}")