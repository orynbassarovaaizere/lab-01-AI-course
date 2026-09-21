import argparse
import json
import os
import sys
import time

from google import genai
from google.genai import types

from texts import CORPUS, LANGUAGES

DEFAULT_MODEL = "gemini-3.6-flash"


def with_retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            wait = 15 * (i + 1)
            print("   error, retry in " + str(wait) + "s: " + str(e)[:80])
            time.sleep(wait)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--call", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Set GEMINI_API_KEY first.")
        return 1
    client = genai.Client(api_key=key)

    result = {"model": args.model, "tokenizer": "gemini",
              "token_counts": {}, "request": {}, "answers": {}}

    print("\nGemini token counts (" + args.model + ")\n" + "-" * 60)
    for item, versions in CORPUS.items():
        row = {}
        for lang in LANGUAGES:
            row[lang] = client.models.count_tokens(
                model=args.model, contents=versions[lang]).total_tokens
        result["token_counts"][item] = row
        print(item.ljust(14),
              "  ".join(l.upper() + "=" + str(row[l]) for l in LANGUAGES),
              "  KK/EN=%.2fx  RU/EN=%.2fx" % (row["kk"] / row["en"], row["ru"] / row["en"]))

    print("\nFull request (system prompt + complaint, joined as one text)\n" + "-" * 60)
    for lang in LANGUAGES:
        n = client.models.count_tokens(
            model=args.model,
            contents=CORPUS["system_prompt"][lang] + "\n\n" + CORPUS["complaint"][lang],
        ).total_tokens
        result["request"][lang] = n
    base = result["request"]["en"]
    print("  ".join(l.upper() + "=" + str(result["request"][l]) for l in LANGUAGES),
          "  KK/EN=%.2fx  RU/EN=%.2fx" % (result["request"]["kk"] / base,
                                          result["request"]["ru"] / base))

    def save():
        with open("measurements_gemini.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    save()

    if args.call:
        print("\nReal answers\n" + "-" * 60)
        for lang in LANGUAGES:
            try:
                r = with_retry(lambda: client.models.generate_content(
                    model=args.model,
                    contents=CORPUS["complaint"][lang],
                    config=types.GenerateContentConfig(
                        system_instruction=CORPUS["system_prompt"][lang]),
                ))
            except Exception as e:
                print(lang.upper() + ": FAILED after retries: " + str(e)[:100])
                continue
            u = r.usage_metadata
            visible = u.candidates_token_count or 0
            thinking = getattr(u, "thoughts_token_count", 0) or 0
            result["answers"][lang] = {"input": u.prompt_token_count,
                                       "visible_output": visible,
                                       "thinking": thinking,
                                       "text": r.text}
            print(lang.upper() + ": input=" + str(u.prompt_token_count),
                  "visible_out=" + str(visible), "thinking=" + str(thinking))
            print("   " + (r.text or "").replace("\n", " ")[:300] + "\n")
            save()

    print("Saved: measurements_gemini.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
