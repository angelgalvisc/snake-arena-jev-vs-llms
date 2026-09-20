#!/usr/bin/env python3
"""Show that a System One answer's size is set by the question, not the state.

    python output_shape.py

Asks Jev the same question about two very different states, then varies the
shape of the question. Output tokens track the answer space and ignore the
content, which is what "the model does not generate text" means in practice.
Costs a fraction of a cent: output tokens are free and the states are tiny.
"""
import http.client
import json
import os
import time

HOST = "api.typesafe.ai"
EASY = "The customer wrote: thanks, all sorted!"
HARD = ("The customer was charged twice for order A-104, disputes one charge, "
        "mentions a prior unresolved ticket, and asks to escalate to a manager.")


def ask(key, state, questions):
    conn = http.client.HTTPSConnection(HOST, timeout=60)
    started = time.perf_counter()
    conn.request("POST", "/v1/systemone",
                 body=json.dumps({"model": "jev-latest", "state": state,
                                  "questions": questions}).encode(),
                 headers={"Authorization": f"Bearer {key}",
                          "Content-Type": "application/json"})
    payload = json.loads(conn.getresponse().read())
    conn.close()
    if "usage" not in payload:
        raise RuntimeError(json.dumps(payload)[:300])
    return payload["usage"], round((time.perf_counter() - started) * 1000)


def choice(n):
    return {"q": {"type": "choice", "instructions": "Pick one.",
                  "criteria": {f"opt{i}": f"Option number {i}" for i in range(n)}}}


def main():
    key = os.environ["TYPESAFE_API_KEY"]
    shapes = [
        ("choice, 2 options", choice(2)),
        ("choice, 4 options", choice(4)),
        ("choice, 8 options", choice(8)),
        ("noul, 1 question", {"a": {"type": "noul", "instructions": "Is this resolved?"}}),
        ("noul, 2 questions", {"a": {"type": "noul", "instructions": "Is this resolved?"},
                               "b": {"type": "noul", "instructions": "Is the customer upset?"}}),
        ("score, 3 levels", {"q": {"type": "score", "instructions": "How urgent?",
                                   "criteria": ["not urgent", "urgent", "very urgent"]}}),
    ]
    print(f"\n  {'question shape':<20} {'out tok':>8} {'out tok':>8}   "
          f"{'in tok':>7} {'in tok':>7}")
    print(f"  {'':<20} {'easy':>8} {'hard':>8}   {'easy':>7} {'hard':>7}")
    for label, questions in shapes:
        easy, _ = ask(key, EASY, questions)
        hard, _ = ask(key, HARD, questions)
        flag = "" if easy["output_tokens"] == hard["output_tokens"] else "  <-- differs"
        print(f"  {label:<20} {easy['output_tokens']:8d} {hard['output_tokens']:8d}   "
              f"{easy['input_tokens']:7d} {hard['input_tokens']:7d}{flag}")
    print("\n  Output size follows the question. The state moves only the input count.\n")


if __name__ == "__main__":
    main()
