#!/usr/bin/env python3
"""Measure native Inkling prefill, decode, and SSD prefix reuse over HTTP."""

import argparse
import ast
import json
import os
import re
import time
import urllib.request
import uuid
from pathlib import Path

from tokenizers import Tokenizer

MODEL = "Inkling-2-Mini-oQ4e-mtp"
POEM = (
    "Write a 12-line poem about Austin, Texas. Include a scene by Lady Bird Lake, "
    "the Congress Avenue bats and the sound of live music. Give the poem a clear "
    "morning-to-evening arc. Return only the poem, without a title."
)
WEATHER = (
    "Write a standalone Python 3 standard-library script that estimates tomorrow's Austin weather "
    "using a local CSV. Columns are date (ISO format), high_f, low_f and precip_in. Accept the CSV "
    "path as the only command-line argument. Sort the observations by date. Estimate tomorrow's "
    "high and low from the mean of the seven most recent observations (or use all available rows "
    "when fewer than seven exist). Estimate rain_probability as the fraction of those days with "
    "positive precipitation. Print one JSON object with date, high_f, low_f and rain_probability. "
    "Give an actionable error for empty input. Treat this as a historical estimate and make no "
    "network requests. Return one fenced Python code block."
)


def padded_prompt(tokenizer, goal, header, rows, footer=""):
    low, high = 0, len(rows)
    while low < high:
        mid = (low + high + 1) // 2
        trial = header + "".join(rows[:mid]) + footer
        if len(tokenizer.encode(trial).ids) <= goal:
            low = mid
        else:
            high = mid - 1
    return header + "".join(rows[:low]) + footer


def complete(url, messages, budget, cache_store=True):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": budget,
        "seed": 0,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if not cache_store:
        headers["X-oMLX-Cache-Store"] = "off"
    request = urllib.request.Request(
        url + "/v1/chat/completions", data=json.dumps(payload).encode(), headers=headers
    )
    start = time.monotonic()
    first_token = first_answer = finish = None
    answer = []
    usage = {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            event = line[5:].strip()
            if event == b"[DONE]":
                break
            chunk = json.loads(event)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                elapsed = time.monotonic() - start
                if first_token is None and (reasoning or text):
                    first_token = elapsed
                if text:
                    if first_answer is None:
                        first_answer = elapsed
                    answer.append(text)
                finish = choice.get("finish_reason") or finish
    if first_token is None or finish is None or not usage:
        raise RuntimeError("Incomplete streaming response")
    return {
        "first_token_s": first_token,
        "first_answer_s": first_answer,
        "elapsed_s": time.monotonic() - start,
        "finish": finish,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "completion_tokens": usage["completion_tokens"],
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "decode_tokens_per_s": usage.get("generation_tokens_per_second"),
    }, "".join(answer)


def emit(case, metrics, **extra):
    print(json.dumps({"case": case, **metrics, **extra}), flush=True)


def prefix_benchmark(url, tokenizer):
    header = f"Reference run {uuid.uuid4().hex}. Context for a validation response.\n"
    rows = [
        f"Reference entry {i}. Record a timestamp, owner, and observation before writing the report.\n"
        for i in range(1000)
    ]
    prefix = padded_prompt(tokenizer, 8400, header, rows)
    results = []
    for label, word in (("prefix-cold", "CEDAR"), ("prefix-reused", "OAK")):
        messages = [
            {"role": "system", "content": prefix},
            {"role": "user", "content": f"Return only the validation code {word}."},
        ]
        result, answer = complete(url, messages, 768)
        valid = answer.strip() == word and result["finish"] == "stop"
        emit(label, result, output_passed=valid)
        results.append(result["cached_tokens"])
    if results[0] != 0 or results[1] < 0.8 * 8400:
        raise SystemExit(f"Expected a cold request followed by a large cache hit, got {results}")


def output_passed(workload, answer, finish):
    if finish != "stop":
        return False
    if workload == "poem":
        return sum(bool(line.strip()) for line in answer.splitlines()) == 12
    fence = re.search(r"\x60{3}(?:python|py)\s*\n(.*?)\x60{3}", answer, re.DOTALL | re.IGNORECASE)
    try:
        ast.parse(fence.group(1) if fence else answer)
    except SyntaxError:
        return False
    return bool(answer.strip())


def speed_benchmark(url, tokenizer, budget):
    stations = ("Riverside", "East Austin", "Mueller", "Zilker")
    rows = [
        f"Observation {i:04d} | {stations[i % len(stations)]} | "
        f"high {76 + i % 24} F | low {58 + i % 15} F | wind {i % 13 + 2} mph.\n"
        for i in range(1600)
    ]
    for tier, target in (("small", 512), ("medium", 8192), ("long", 30720)):
        for workload, task in (("poem", POEM), ("weather", WEATHER)):
            header = f"Reference set {uuid.uuid4().hex}. Synthetic Austin weather archive.\n"
            footer = "\nEnd of archive.\nAssignment\n" + task
            prompt = padded_prompt(tokenizer, target, header, rows, footer)
            result, answer = complete(
                url, [{"role": "user", "content": prompt}], budget, cache_store=False
            )
            emit(
                f"{tier}-{workload}", result,
                output_passed=output_passed(workload, answer, result["finish"]),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("prefix", "speed", "all"), default="prefix")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--budget", type=int, default=1536, help="Output tokens for the speed suite")
    args = parser.parse_args()
    root = Path(os.environ.get("INKLING_HOME", Path.home() / ".inkling-m6-optimized"))
    tokenizer = Tokenizer.from_file(str(root / "models" / MODEL / "tokenizer.json"))
    url = args.url.rstrip("/")
    warmup, _ = complete(
        url, [{"role": "user", "content": "Reply with only 'ready'."}], 192, cache_store=False
    )
    emit("warmup", warmup)
    if args.suite in ("prefix", "all"):
        prefix_benchmark(url, tokenizer)
    if args.suite in ("speed", "all"):
        speed_benchmark(url, tokenizer, args.budget)


if __name__ == "__main__":
    main()
