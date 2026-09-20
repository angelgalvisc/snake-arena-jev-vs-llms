#!/usr/bin/env python3
"""Race two or three players on the same board and record a replay.

    python race.py --seconds 60 --seed 1
    python race.py --seconds 60 --with-kimi
    python race.py --tick-ms 1000

Every contender plays the same seed at the same time, each in its own thread on
its own connection. Each move is flushed to the replay as it happens, together
with the reasoning trace when the model returns one, so a run that dies halfway
still leaves behind every move you paid for.
"""
import argparse
import json
import threading
import time

import players
from snake import Game

LAB = {"jev": "TypeSafe", "claude": "Anthropic", "kimi": "Moonshot",
       "openai": "OpenAI"}

# Every entrant the arena knows: short name -> (lab, how to build it).
# --lineup names them in display order, so the race order is the board order.
ROSTER = {
    "jev":       ("jev",    lambda a: (players.TypeSafe(a.jev_model), a.jev_model)),
    "haiku":     ("claude", lambda a: (players.Anthropic(a.claude_model), a.claude_model)),
    "opus":      ("claude", lambda a: (players.Anthropic("claude-opus-5", a.opus_effort),
                                       "claude-opus-5")),
    "luna":      ("openai", lambda a: (players.OpenAI("gpt-5.6-luna", a.openai_effort),
                                       "gpt-5.6-luna")),
    "sol":       ("openai", lambda a: (players.OpenAI("gpt-5.6-sol", a.openai_effort),
                                       "gpt-5.6-sol")),
    "kimi-fast": ("kimi",   lambda a: (players.Moonshot("kimi-k2.7-code-highspeed",
                                                        a.kimi_effort),
                                       "kimi-k2.7-code-highspeed")),
    "kimi":      ("kimi",   lambda a: (players.Moonshot(a.kimi_model, a.kimi_effort),
                                       a.kimi_model)),
}


def run_player(player, name, seed, tick_ms, max_steps, sink, lock, t0, budget_s):
    game = Game(seed)
    calls = 0
    last = time.perf_counter()
    while game.dead is None and game.steps < max_steps:
        if budget_s and time.perf_counter() - t0 >= budget_s:
            game.dead = "time up"
            break
        started = time.perf_counter()
        direction = player.move(game)
        calls += 1
        if not budget_s:
            skipped = int((time.perf_counter() - last) * 1000 / tick_ms)
            last = time.perf_counter()
            for _ in range(skipped):
                if game.dead or game.steps >= max_steps:
                    break
                game.step(game.dir)
        if not game.dead:
            game.step(direction)
        row = {"t": round(time.perf_counter() - t0, 3), "who": name,
               "snake": [list(c) for c in game.snake], "food": list(game.food or []),
               "score": game.score, "steps": game.steps, "calls": calls,
               "cost": round(player.tally.cost(player.model), 6),
               "tok_in": player.tally.input_tokens,
               "tok_out": player.tally.output_tokens,
               "ms": round((time.perf_counter() - started) * 1000),
               "dead": game.dead}
        traces = getattr(player, "traces", None)
        if traces:
            row["why"] = traces[-1]
        decisions = getattr(player, "decisions", None)
        if decisions:                      # System One answers carry a distribution
            row["confidence"] = decisions[-1]["confidence"]
            row["probabilities"] = decisions[-1]["probabilities"]
        with lock:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--tick-ms", type=int, default=6000)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--claude-model", default="claude-haiku-4-5")
    ap.add_argument("--opus-effort", default="low",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--kimi-model", default="kimi-k3")
    ap.add_argument("--kimi-effort", default="low", choices=["low", "high", "max"])
    ap.add_argument("--openai-model", default="gpt-5.6-luna")
    ap.add_argument("--openai-effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--lineup", default="jev,haiku",
                    help="who races, in display order: " + ", ".join(ROSTER))
    ap.add_argument("--out", default="race.jsonl")
    args = ap.parse_args()

    line = []
    for key in (k.strip() for k in args.lineup.split(",") if k.strip()):
        if key not in ROSTER:
            ap.error(f"unknown entrant {key!r}; pick from {', '.join(ROSTER)}")
        lab, build = ROSTER[key]
        player, name = build(args)
        line.append((player, name, lab))

    lock, t0 = threading.Lock(), time.perf_counter()
    sink = open(args.out, "w")
    sink.write(json.dumps({"seed": args.seed, "seconds": args.seconds,
                           "tick_ms": None if args.seconds else args.tick_ms,
                           "players": [{"name": n, "lab": LAB[k]} for _, n, k in line]
                           }) + "\n")
    sink.flush()
    mode = f"{args.seconds:g}s each" if args.seconds else f"{args.tick_ms}ms per tick"
    print(f"  seed {args.seed}, {mode}, {len(line)} players\n")

    threads = [threading.Thread(target=run_player,
               args=(p, n, args.seed, args.tick_ms, args.max_steps, sink, lock,
                     t0, args.seconds)) for p, n, _ in line]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for p, _, _ in line:
        p.close()
    sink.close()

    rows = [json.loads(l) for l in open(args.out)][1:]
    for _, name, _ in line:
        mine = [r for r in rows if r["who"] == name]
        if not mine:
            print(f"  {name:<20} no moves"); continue
        u = mine[-1]
        served = next((p for p, n, _ in line if n == name), None)
        build = getattr(served, "resolved", None)
        print(f"  {name:<26} {u['score']:>3} pts  {u['steps']:>3} steps  "
              f"{u['cost']*100:>7.3f} c  {u['tok_in']+u['tok_out']:>7} tok  "
              f"{u['dead'] or 'alive':<5}"
              + (f"  served by {build}" if build and build != name else ""))
    print(f"\n  replay: {args.out} ({len(rows)} moves)")


if __name__ == "__main__":
    main()
