"""Part 2 -- the measurement. Needs an API key; run once, on the projector.

Does two things:

1. One real request per language, so you can see a response and the ``usage``
   block that comes back with it. Three calls, because the answer length is
   itself language-dependent -- assuming it away is how you get a wrong bill.
   This is the only part of the lab that spends money.
2. Token counts for every text in :mod:`texts`, in every language, using the
   ``count_tokens`` endpoint, plus one combined count per language for the
   real request shape (system prompt + complaint, one call) that Part 3
   actually prices. Counting tokens is free and does not run the model --
   it just asks the tokenizer.

Results are written to ``measurements.json`` for Part 3.

Run:
    python3 part2_measure.py            # count tokens only, no model call
    python3 part2_measure.py --call     # also answer the complaint in en, ru, kk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import anthropic
from dotenv import load_dotenv

from prices import MODELS, DEFAULT_MODEL
from texts import CORPUS, LANGUAGES

#: Loads ANTHROPIC_API_KEY from a .env file next to this script, if present.
#: A real environment variable, if already set, is left untouched.
load_dotenv()

OUTPUT_PATH = Path(__file__).with_name("measurements.json")

#: Deliberately bounded, not deliberately tiny -- this is a lab about cost,
#: running on one shared key, so a runaway 128k-token answer is exactly the
#: failure we are here to talk about. It has to clear thinking, though:
#: claude-opus-5 runs adaptive thinking by default, and thinking tokens are
#: billed at the output rate but never appear in the reply. Measured on this
#: corpus, thinking alone ran 176-342 tokens per language before a single
#: visible word -- 512 was not enough headroom and cut answers off mid-word.
#: 2048 clears that with margin. Do not "fix" this by disabling thinking:
#: the lecture's own warning is that thinking tokens are an invisible cost,
#: and hiding them here would remove the exact trap the lab is meant to show.
MAX_TOKENS = 2048


def count_tokens(client: anthropic.Anthropic, model_id: str, text: str) -> int:
    """Return the number of input tokens ``text`` costs on ``model_id``.

    Token counts are model-specific: the same string counts differently on
    different Claude generations. Do not substitute a third-party tokenizer --
    ``tiktoken`` is OpenAI's and gives the wrong answer here.

    Args:
        client: An authenticated Anthropic client.
        model_id: The exact model id, e.g. ``"claude-opus-5"``.
        text: The text to count as a single user message.

    Returns:
        Input tokens, including the few tokens of message framing the API adds.
    """
    result = client.messages.count_tokens(
        model=model_id,
        messages=[{"role": "user", "content": text}],
    )
    return result.input_tokens


def count_request_tokens(client: anthropic.Anthropic, model_id: str, lang: str) -> int:
    """Return input tokens for one real request: system prompt + complaint.

    Every ``count_tokens`` call bills the same handful of message-framing
    tokens as a real request would -- but only once per call. Summing the
    system prompt's and the complaint's *standalone* counts (each counted as
    its own single-message call) therefore double-counts that framing. This
    counts them the way :func:`one_real_request` actually sends them: one
    call, system plus one user message, so the framing is billed once.

    Args:
        client: An authenticated Anthropic client.
        model_id: The exact model id, e.g. ``"claude-opus-5"``.
        lang: Which language version of the corpus to count.

    Returns:
        Input tokens for one real support request in that language.
    """
    result = client.messages.count_tokens(
        model=model_id,
        system=CORPUS["system_prompt"][lang],
        messages=[{"role": "user", "content": CORPUS["complaint"][lang]}],
    )
    return result.input_tokens


def one_real_request(
    client: anthropic.Anthropic, model_id: str, lang: str
) -> Optional[Dict[str, int]]:
    """Send one request and print the answer and its billed usage.

    Args:
        client: An authenticated Anthropic client.
        model_id: The exact model id to bill against.
        lang: Which language version of the corpus to send.

    Returns:
        A mapping with ``input_tokens`` and ``output_tokens`` as billed, or
        ``None`` if the model declined to answer.
    """
    response = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        system=CORPUS["system_prompt"][lang],
        messages=[{"role": "user", "content": CORPUS["complaint"][lang]}],
    )

    # Always check stop_reason before reading content. A refusal returns
    # HTTP 200 with an empty answer, not an exception.
    if response.stop_reason == "refusal":
        print(f"  model declined: {response.stop_details}")
        return None

    print(f"  stop_reason: {response.stop_reason}")
    for block in response.content:
        if block.type == "text":
            print("  --- answer ---")
            print("  " + block.text.replace("\n", "\n  "))

    usage = response.usage
    print(f"  billed: {usage.input_tokens} in, {usage.output_tokens} out")
    if response.stop_reason == "max_tokens":
        print(f"  NOTE: answer was cut off at max_tokens={MAX_TOKENS}.")
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def main() -> int:
    """Count tokens for the whole corpus, optionally make one real request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(MODELS),
        help=f"which model to price against (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--call",
        action="store_true",
        help="also answer the complaint in each language (this costs money)",
    )
    args = parser.parse_args()

    model_id = MODELS[args.model].model_id

    try:
        client = anthropic.Anthropic()
    except anthropic.AnthropicError as exc:
        print(f"could not build a client: {exc}", file=sys.stderr)
        print("set ANTHROPIC_API_KEY, or run `ant auth login`.", file=sys.stderr)
        return 1

    counts: Dict[str, Dict[str, int]] = {}
    print(f"counting tokens on {model_id} (free, no model run)")
    try:
        for item_id, versions in CORPUS.items():
            counts[item_id] = {
                lang: count_tokens(client, model_id, versions[lang])
                for lang in LANGUAGES
            }
            row = "  ".join(f"{lang}={counts[item_id][lang]}" for lang in LANGUAGES)
            print(f"  {item_id:<14} {row}")

        request_tokens: Dict[str, int] = {
            lang: count_request_tokens(client, model_id, lang) for lang in LANGUAGES
        }
        row = "  ".join(f"{lang}={request_tokens[lang]}" for lang in LANGUAGES)
        print(f"  {'request':<14} {row}  (system + complaint, one call -- what Part 3 prices)")
    except anthropic.RateLimitError as exc:
        print(f"rate limited: {exc}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:
        print(f"API error {exc.status_code}: {exc}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"could not reach the API: {exc}", file=sys.stderr)
        return 1

    billed: Dict[str, Dict[str, int]] = {}
    if args.call:
        print(f"\nanswering the same complaint on {model_id}, in each language:")
        for lang in LANGUAGES:
            print(f"\n[{lang}]")
            try:
                result = one_real_request(client, model_id, lang)
            except anthropic.APIStatusError as exc:
                print(f"API error {exc.status_code}: {exc}", file=sys.stderr)
                return 1
            if result is not None:
                billed[lang] = result

    payload = {
        "model": args.model,
        "model_id": model_id,
        "token_counts": counts,
        "request_tokens": request_tokens,
        "one_request_billed": billed or None,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUTPUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
