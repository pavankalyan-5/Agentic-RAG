# USD per 1M tokens. Verify against platform.openai.com/pricing before real runs.
PRICES: dict[str, dict[str, float]] = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    # KeyError on unknown models is deliberate: an unpriced model must not count as free.
    price = PRICES[model]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
