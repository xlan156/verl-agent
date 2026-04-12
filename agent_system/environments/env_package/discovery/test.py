import json
def _extract_last_braced_object(text: str):
    """Return the last balanced {...} substring if present."""
    if "{" not in text or "}" not in text:
        return None

    depth = 0
    end_idx = None
    start_idx = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    end_idx = i

    if start_idx is None or end_idx is None:
        return None

    return text[start_idx : end_idx + 1]


if __name__ == "__main__":
    # Example usage
    test_str = 'Some text before. {"type": "move", "direction": "north"} Some text after. Another action: {"type": "pick", "object": "key"}.'
    print(_extract_last_braced_object(test_str))
    j = json.loads(_extract_last_braced_object(test_str))
    print(type(j))