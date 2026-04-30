import argparse
import json
from typing import List, Dict, Any

import pandas as pd


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    return records


def convert_jsonl_to_parquet(jsonl_path: str, parquet_path: str) -> None:
    """Convert DiscoveryWorld SFT JSONL to parquet for SFTDataset.

    Expected JSONL schema (from generate_discoveryworld_sft.py):
      {"prompt": str, "response": str, ...}

    This function keeps at least the `prompt` and `response` columns, and
    also preserves all other keys under a `raw` column for debugging if needed.
    """
    data = read_jsonl(jsonl_path)
    if not data:
        raise ValueError(f"No valid records found in {jsonl_path}")

    rows: List[Dict[str, Any]] = []
    for obj in data:
        prompt = obj.get("prompt", "")
        response = obj.get("response", "")
        rows.append({
            "prompt": prompt,
            "response": response,
            "raw": obj,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(df)} rows to {parquet_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DiscoveryWorld SFT JSONL to parquet for fsdp_sft_trainer.",
    )
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Path to discoveryworld_sft.jsonl produced by generate_discoveryworld_sft.py")
    parser.add_argument("--output_parquet", type=str, required=True,
                        help="Output parquet file path, e.g. ~/data/discoveryworld/train.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_jsonl_to_parquet(args.input_jsonl, args.output_parquet)


if __name__ == "__main__":
    main()
