#!/usr/bin/env python3
"""Balance SFT parquet rows by oversampling minority actions."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


def get_nested_value(value: Any, path: str) -> Any:
    """Read a dotted path from nested dict/list values."""
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


def extract_action(row: pd.Series, action_field: str) -> str:
    if action_field == "auto":
        for candidate in ("reward_model.ground_truth", "response", "messages.1.content"):
            try:
                return extract_action(row, candidate)
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        raise ValueError("Could not infer action field from row")

    if "." in action_field:
        column, nested_path = action_field.split(".", 1)
        value = get_nested_value(row[column], nested_path)
    else:
        value = row[action_field]

    if isinstance(value, str):
        match = ACTION_RE.search(value)
        return match.group(1).strip() if match else value.strip()

    return str(value)


def format_counts(counts: pd.Series) -> str:
    return "\n".join(f"{action}: {count}" for action, count in counts.sort_index().items())


def balance_actions(df: pd.DataFrame, actions: pd.Series, seed: int) -> pd.DataFrame:
    counts = actions.value_counts()
    if counts.empty:
        raise ValueError("Input parquet has no rows")

    target_count = int(counts.max())
    balanced_parts = []
    for action, count in counts.items():
        action_rows = df.loc[actions == action]
        need = target_count - int(count)
        if need <= 0:
            balanced_parts.append(action_rows)
            continue

        replace = need > len(action_rows)
        sampled = action_rows.sample(n=need, replace=replace, random_state=seed)
        balanced_parts.append(pd.concat([action_rows, sampled], ignore_index=True))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oversample minority actions in an SFT parquet until every action has the same count.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sft/skill_sft_data_train.parquet"),
        help="Input parquet path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sft/skill_sft_data_train_balanced.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument(
        "--action-field",
        default="reward_model.ground_truth",
        help=(
            "Column or dotted nested field containing the action. Use 'auto' to try "
            "reward_model.ground_truth, response, then messages.1.content."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and shuffling.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the before/after counts; do not write output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    actions = df.apply(lambda row: extract_action(row, args.action_field), axis=1)

    before_counts = actions.value_counts()
    target_count = int(before_counts.max()) if not before_counts.empty else 0
    total_after = target_count * len(before_counts)

    print("Before balancing:")
    print(format_counts(before_counts))
    print(f"Total rows: {len(df)}")

    print("\nAfter balancing:")
    after_counts = pd.Series({action: target_count for action in before_counts.index})
    print(format_counts(after_counts))
    print(f"Total rows: {total_after}")

    if args.dry_run:
        return

    balanced = balance_actions(df, actions, args.seed)
    output_actions = balanced.apply(lambda row: extract_action(row, args.action_field), axis=1)
    output_counts = output_actions.value_counts()
    if not all(math.isclose(count, target_count) for count in output_counts):
        raise RuntimeError(f"Unexpected output distribution:\n{format_counts(output_counts)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_parquet(args.output, index=False)
    print(f"\nWrote balanced parquet: {args.output}")


if __name__ == "__main__":
    main()
