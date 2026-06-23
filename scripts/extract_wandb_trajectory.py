#!/usr/bin/env python3
"""Extract one env_index trajectory from a W&B table JSON file."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
from pathlib import Path
from typing import Any


DEFAULT_TABLE = (
    "wandb/run-20260616_152004-g2b4idnx/files/media/table/eval/"
    "llm_steps_0_18c0365c1084225cd5b3.table.json"
)


def load_table(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        table = json.load(f)

    if not isinstance(table, dict) or "columns" not in table or "data" not in table:
        raise ValueError(f"{path} is not a W&B table JSON with 'columns' and 'data'")

    columns = table["columns"]
    data = table["data"]
    if not isinstance(columns, list) or not isinstance(data, list):
        raise ValueError(f"{path} has invalid W&B table shape")

    rows = [dict(zip(columns, row)) for row in data]
    return columns, rows


def extract_rows(
    rows: list[dict[str, Any]],
    env_index: int,
    global_step: int | None = None,
) -> list[dict[str, Any]]:
    trajectory = [row for row in rows if row.get("env_index") == env_index]
    if global_step is not None:
        trajectory = [row for row in trajectory if row.get("global_step") == global_step]

    trajectory.sort(
        key=lambda row: (
            row.get("global_step", -1),
            row.get("rollout_step", -1),
        )
    )
    return trajectory


def format_text(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        header = (
            f"[global_step={row.get('global_step')} "
            f"rollout_step={row.get('rollout_step')} "
            f"env_index={row.get('env_index')}]"
        )
        fields = [
            header,
            f"reward: {row.get('reward')}",
            f"done: {row.get('done')}",
            f"llm_output: {row.get('llm_output')}",
            f"projected_action: {row.get('projected_action')}",
            f"is_action_valid: {row.get('is_action_valid')}",
            f"teacher_skill: {row.get('teacher_skill')}",
            f"action_status: {row.get('action_status')}",
            "",
            "prompt:",
            str(row.get("prompt", "")),
        ]
        parts.append("\n".join(fields))
    return "\n\n" + ("-" * 80 + "\n\n").join(parts) if parts else ""


def write_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    output_format: str,
    output_path: Path | None,
) -> None:
    if output_format == "json":
        text = json.dumps(rows, ensure_ascii=False, indent=2)
    elif output_format == "jsonl":
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    elif output_format == "text":
        text = format_text(rows)
    elif output_format == "csv":
        if output_path is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            return

        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    if output_path is None:
        print(text)
    else:
        output_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a W&B llm_steps table JSON and extract one env_index trajectory."
    )
    parser.add_argument(
        "table_json",
        nargs="?",
        default=DEFAULT_TABLE,
        help=f"W&B table JSON path. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "-e",
        "--env-index",
        type=int,
        required=True,
        help="Environment index to extract.",
    )
    parser.add_argument(
        "-g",
        "--global-step",
        type=int,
        default=None,
        help="Optional global_step filter.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("text", "json", "jsonl", "csv"),
        default="text",
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List available env_index values and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table_path = Path(args.table_json)
    columns, rows = load_table(table_path)

    if "env_index" not in columns:
        raise ValueError(f"{table_path} does not contain an env_index column")

    if args.list_envs:
        envs = sorted({row.get("env_index") for row in rows})
        print(" ".join(str(env) for env in envs))
        return 0

    trajectory = extract_rows(rows, args.env_index, args.global_step)
    if not trajectory:
        step_msg = "" if args.global_step is None else f" at global_step={args.global_step}"
        print(f"No rows found for env_index={args.env_index}{step_msg}", file=sys.stderr)
        return 1

    write_rows(trajectory, columns, args.format, args.output)
    print(
        f"Extracted {len(trajectory)} rows for env_index={args.env_index}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(main())
