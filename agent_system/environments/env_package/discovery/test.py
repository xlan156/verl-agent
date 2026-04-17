import json
import math

import torch
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


def func(x):
    res = math.ceil(8.0 / math.log(math.e + x)) / 10.0
    print(f"x={x}, res={res}")
    
def entropy_bonus(probs):
    return -(probs * torch.log(probs + 1e-8)).sum()

def kl_to_uniform(probs):
    n = len(probs)
    uniform = torch.ones_like(probs) / n
    kl = -(probs * (torch.log(probs + 1e-8) - torch.log(uniform))).sum()
    # Quantize to the nearest 0.1 increment.
    return torch.round(kl / 0.1) * 0.1

if __name__ == "__main__":
    # Example usage
    probs = torch.tensor([0.7,0.3,0.0,0.0,0.0])
    print("Entropy bonus:", entropy_bonus(probs).item())
    print("KL to uniform:", kl_to_uniform(probs).item())
    
    probs = torch.tensor([0.6, 0.4])
    print("Entropy bonus:", entropy_bonus(probs).item())
    print("KL to uniform:", kl_to_uniform(probs).item())