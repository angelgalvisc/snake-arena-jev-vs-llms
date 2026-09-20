"""The two contenders. Both are constrained to the same four answers.

TypeSafe returns a Choice over the option keys we send.
Anthropic uses structured outputs with an enum of the same keys.
Neither can reply with free text, so no move is ever lost to a parser.
"""
import http.client
import json
import os
import time

from snake import DIRS, STRATEGY, describe

COMMIT_OPTIONS = {
    "1": "Move once, then look at the board again.",
    "2": "Move twice on this heading before looking again.",
    "3": "Move three times on this heading before looking again.",
    "5": "Move five times on this heading before looking again.",
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"direction": {"type": "string", "enum": list(DIRS)}},
    "required": ["direction"],
    "additionalProperties": False,
}

# Input $/1M, output $/1M. Update when the providers change their rates.
# USD per million tokens, (input, output), read off each vendor's own price page.
PRICES = {
    "jev-latest": (0.042, 0.0),          # output tokens are free on this one
    "jev-preview": (0.042, 0.0),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-6-astra": (10.00, 50.00),
    "kimi-k2.6": (0.95, 4.00),
    "kimi-k2.7-code": (0.95, 4.00),
    "kimi-k2.7-code-highspeed": (1.90, 8.00),
    "kimi-k3": (3.00, 15.00),
}


class Tally:
    """Token counters for one game."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, tin, tout):
        self.input_tokens += tin
        self.output_tokens += tout

    def cost(self, model):
        pin, pout = PRICES.get(model, (0.0, 0.0))
        return self.input_tokens / 1e6 * pin + self.output_tokens / 1e6 * pout


class TypeSafe:
    """Jev via POST /v1/systemone.

    A System One model answers a typed question about a state and returns the
    answer directly: for a `choice`, the selected option plus a probability for
    every option and a confidence score. The confidence is what the other
    players have no equivalent of, so every answer is kept in `decisions` for
    the replay rather than thrown away once the direction is read.
    """

    label = "TypeSafe"

    def __init__(self, model="jev-latest"):
        self.model = model
        self.decisions = []
        self.tally = Tally()
        self._key = os.environ["TYPESAFE_API_KEY"]
        self._conn = http.client.HTTPSConnection("api.typesafe.ai", timeout=60)

    def move(self, game):
        view = describe(game)
        questions = {"move": {"type": "choice", "instructions": STRATEGY,
                              "criteria": view["options"]}}
        body = json.dumps({"model": self.model, "state": view["state"],
                           "questions": questions})
        self._conn.request("POST", "/v1/systemone", body=body.encode(), headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        payload = json.loads(self._conn.getresponse().read())
        if "answers" not in payload:
            raise RuntimeError(json.dumps(payload)[:300])
        self.tally.add(payload["usage"]["input_tokens"], payload["usage"]["output_tokens"])
        self.resolved = payload.get("model", self.model)
        answer = payload["answers"]["move"]
        self.decisions.append({"choice": answer["choice"],
                               "confidence": answer.get("confidence"),
                               "probabilities": answer.get("probabilities")})
        return answer["choice"]

    def close(self):
        self._conn.close()


class Anthropic:
    """Claude via POST /v1/messages with a structured-output enum.

    Thinking is configured per model generation. Opus 5 takes adaptive thinking
    plus `output_config.effort`; `budget_tokens` is rejected there with a 400.
    Haiku 4.5 still takes an explicit budget. `display: "summarized"` is what
    makes any reasoning visible -- the default returns empty thinking blocks --
    so it is set whenever thinking is on, and `traces` holds whatever came back.
    """

    label = "Anthropic"
    ADAPTIVE = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
                "claude-sonnet-5", "claude-fable-5-1")

    def __init__(self, model="claude-haiku-4-5", effort=None, thinking=False):
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.adaptive = model in self.ADAPTIVE
        self.tally = Tally()
        self.traces = []
        self._key = os.environ["ANTHROPIC_API_KEY"]
        self._workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        self._conn = http.client.HTTPSConnection("api.anthropic.com", timeout=120)

    def move(self, game):
        view = describe(game)
        options = "\n".join(f"  {k}: {v}" for k, v in view["options"].items())
        out = {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
        if self.effort:
            out["effort"] = self.effort
        req = {
            "model": self.model,
            "max_tokens": 4096 if (self.adaptive or self.thinking) else 64,
            "system": STRATEGY,
            "output_config": out,
            "messages": [{"role": "user", "content":
                          f"State: {json.dumps(view['state'])}\n"
                          f"Options:\n{options}\n\nWhich direction?"}],
        }
        if self.adaptive:
            req["thinking"] = {"type": "adaptive", "display": "summarized"}
        elif self.thinking:
            req["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        headers = {"x-api-key": self._key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        if self._workspace:
            headers["anthropic-workspace-id"] = self._workspace
        self._conn.request("POST", "/v1/messages", body=json.dumps(req).encode(),
                           headers=headers)
        payload = json.loads(self._conn.getresponse().read())
        if "usage" not in payload:
            raise RuntimeError(json.dumps(payload)[:300])
        self.tally.add(payload["usage"]["input_tokens"], payload["usage"]["output_tokens"])
        self.resolved = payload.get("model", self.model)
        blocks = payload["content"]
        self.traces.append("".join(b.get("thinking", "")
                                   for b in blocks if b["type"] == "thinking").strip())
        text = "".join(b.get("text", "") for b in blocks if b["type"] == "text")
        return json.loads(text)["direction"]

    def close(self):
        self._conn.close()


class Moonshot:
    """Kimi via POST /v1/chat/completions, OpenAI-shaped.

    k3 always reasons, so every answer arrives with a reasoning_content trace.
    The trace is kept on `self.traces` for the replay to show later.
    """

    HOST = "api.moonshot.ai"

    def __init__(self, model="kimi-k3", effort="max"):
        self.model = model
        self.effort = effort
        self.tally = Tally()
        self.traces = []
        self._key = os.environ["MOONSHOT_API_KEY"]
        self._conn = http.client.HTTPSConnection(self.HOST, timeout=300)

    def move(self, game):
        view = describe(game)
        options = "\n".join(f"  {k}: {v}" for k, v in view["options"].items())
        req = {"model": self.model, "reasoning_effort": self.effort,
               "max_tokens": 4096,
               "response_format": {"type": "json_schema", "json_schema":
                   {"name": "move", "strict": True, "schema": ANSWER_SCHEMA}},
               "messages": [{"role": "system", "content": STRATEGY},
                            {"role": "user", "content":
                             f"State: {json.dumps(view['state'])}\n"
                             f"Options:\n{options}\n\nWhich direction?"}]}
        self._conn.request("POST", "/v1/chat/completions", body=json.dumps(req).encode(),
                           headers={"Authorization": f"Bearer {self._key}",
                                    "Content-Type": "application/json"})
        payload = json.loads(self._conn.getresponse().read())
        if "usage" not in payload:
            raise RuntimeError(json.dumps(payload)[:300])
        u = payload["usage"]
        self.tally.add(u["prompt_tokens"], u["completion_tokens"])
        self.resolved = payload.get("model", self.model)
        msg = payload["choices"][0]["message"]
        self.traces.append(msg.get("reasoning_content") or "")
        return json.loads(msg["content"])["direction"]

    def close(self):
        self._conn.close()


class OpenAI:
    """GPT via POST /v1/responses, so the reasoning summary comes back too.

    Luna reasons by exception: on an obvious move it spends no reasoning tokens
    and returns no summary, so `traces` holds an empty string for those turns.
    """

    HOST = "api.openai.com"

    def __init__(self, model="gpt-5.6-luna", effort="high"):
        self.model = model
        self.effort = effort
        self.tally = Tally()
        self.traces = []
        self._key = os.environ["OPENAI_API_KEY"]
        self._conn = http.client.HTTPSConnection(self.HOST, timeout=300)

    def move(self, game):
        view = describe(game)
        options = "\n".join(f"  {k}: {v}" for k, v in view["options"].items())
        req = {"model": self.model,
               "input": [{"role": "system", "content": STRATEGY},
                         {"role": "user", "content":
                          f"State: {json.dumps(view['state'])}\n"
                          f"Options:\n{options}\n\nWhich direction?"}],
               "reasoning": {"effort": self.effort, "summary": "detailed"},
               "text": {"format": {"type": "json_schema", "name": "move",
                                   "strict": True, "schema": ANSWER_SCHEMA}}}
        self._conn.request("POST", "/v1/responses", body=json.dumps(req).encode(),
                           headers={"Authorization": f"Bearer {self._key}",
                                    "Content-Type": "application/json"})
        payload = json.loads(self._conn.getresponse().read())
        if "usage" not in payload:
            raise RuntimeError(json.dumps(payload)[:300])
        u = payload["usage"]
        self.tally.add(u["input_tokens"], u["output_tokens"])
        self.resolved = payload.get("model", self.model)
        text, think = "", []
        for item in payload.get("output", []):
            if item.get("type") == "reasoning":
                think += [t.get("text", "") for t in item.get("summary", [])]
            elif item.get("type") == "message":
                text += "".join(c.get("text", "") for c in item.get("content", []))
        self.traces.append(" ".join(think).strip())
        return json.loads(text)["direction"]

    def close(self):
        self._conn.close()
