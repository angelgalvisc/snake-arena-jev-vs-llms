#!/usr/bin/env python3
"""Snake arena: TypeSafe Jev against a Claude model on the same boards.

    python arena.py --lite            # decision quality, clock paused
    python arena.py --seconds 60      # equal wall clock, who scores more
    python arena.py                   # real time, clock runs while you think
    python arena.py --seeds 1 2 3 4 5 --tick-ms 6000
    python arena.py --model claude-sonnet-5

Two modes, because they answer different questions. Report them separately;
never mix them.

  --lite    The board waits for each answer. Latency cannot affect the score,
            so this isolates decision quality.

  --seconds Both players get the same wall clock. The board advances on every
            answer, so nobody is killed for being slow - a slow player simply
            fits fewer moves into the minute. No tick to argue about.

  default   The board runs on a clock. While a player thinks the snake keeps
            moving on its current heading, and if that walks it into a wall it
            dies. Set --tick-ms slow enough that both contenders answer in
            time: the arena counts missed ticks and refuses to name a winner
            if anyone was late, because then the result is about speed.

This mirrors VideoGameBench, which offers an async mode and a paused "lite"
mode for exactly this reason.
"""
import argparse
import json
import statistics as st
import sys
import time

import players
from snake import Game

DEFAULT_TICK_MS = 6000


def play(player, seed, tick_ms, max_steps, lite=False, budget_s=None):
    """One game in one of three modes.

    lite        the board waits for the answer; latency cannot affect the score
    budget_s    a fixed wall-clock budget; a slow player simply gets fewer moves
    otherwise   the board runs on a tick and a late answer misses its turn
    """
    game = Game(seed)
    missed = 0
    started = last = time.perf_counter()
    while game.dead is None and game.steps < max_steps:
        if budget_s and time.perf_counter() - started >= budget_s:
            game.dead = "time up"
            break
        direction = player.move(game)
        if not lite and not budget_s:
            skipped = int((time.perf_counter() - last) * 1000 / tick_ms)
            last = time.perf_counter()
            for _ in range(skipped):      # the clock ran while the player thought
                if game.dead or game.steps >= max_steps:
                    break
                game.step(game.dir)
            missed += skipped
            if game.dead:
                break
        game.step(direction)
    if game.dead is None:
        game.dead = "step cap"
    return game, missed


def run(make_player, name, seeds, tick_ms, max_steps, lite=False, budget_s=None):
    player = make_player()
    rows = []
    try:
        for seed in seeds:
            game, missed = play(player, seed, tick_ms, max_steps, lite, budget_s)
            rows.append({"seed": seed, "score": game.score, "steps": game.steps,
                         "end": game.dead, "missed_ticks": missed,})
            print(f"  {name:<22} seed {seed}: {game.score:>3} points, "
                  f"{game.steps:>3} steps, {game.dead}"
                  + (f"  [MISSED {missed} TICKS]" if missed else ""))
    finally:
        player.close()
    return rows, player.tally.cost(player.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tick-ms", type=int, default=DEFAULT_TICK_MS)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--model", default="claude-haiku-4-5", help="the Claude model")
    ap.add_argument("--thinking", action="store_true", help="extended thinking for Claude")
    ap.add_argument("--lite", action="store_true",
                    help="pause the board while a player thinks (quality only)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="give each player this many seconds of wall clock")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.json:
        mode = ("lite - board paused, decision quality only" if args.lite else
                f"{args.seconds:g}s each - points per minute" if args.seconds else
                f"real time - {args.tick_ms}ms per tick")
        print(f"\nSnake arena ({mode}), {args.max_steps}-step cap, "
              f"seeds {args.seeds}\n")

    jev_rows, jev_cost = run(lambda: players.TypeSafe(args.jev_model),
                             args.jev_model, args.seeds, args.tick_ms, args.max_steps,
                             args.lite, args.seconds)
    print()
    cla_rows, cla_cost = run(lambda: players.Anthropic(args.model, thinking=args.thinking),
                             args.model, args.seeds, args.tick_ms, args.max_steps,
                             args.lite, args.seconds)

    jev = [r["score"] for r in jev_rows]
    cla = [r["score"] for r in cla_rows]
    wins = sum(a > b for a, b in zip(jev, cla))
    losses = sum(a < b for a, b in zip(jev, cla))
    stalled = sum(r["missed_ticks"] for r in jev_rows + cla_rows)

    if args.json:
        print(json.dumps({"mode": "lite" if args.lite else "real-time",
                          "tick_ms": None if args.lite else args.tick_ms,
                          "seeds": args.seeds,
                          args.jev_model: {"runs": jev_rows, "mean": st.mean(jev),
                                           "cost_usd": round(jev_cost, 6)},
                          args.model: {"runs": cla_rows, "mean": st.mean(cla),
                                       "cost_usd": round(cla_cost, 6)},
                          "missed_ticks_total": stalled}, indent=1))
        return

    print(f"\n{'seed':>6} {args.jev_model:>14} {args.model:>22}   winner")
    for r_j, r_c in zip(jev_rows, cla_rows):
        who = args.jev_model if r_j["score"] > r_c["score"] else (
            args.model if r_c["score"] > r_j["score"] else "tie")
        print(f"{r_j['seed']:>6} {r_j['score']:>14} {r_c['score']:>22}   {who}")
    print(f"{'mean':>6} {st.mean(jev):>14.1f} {st.mean(cla):>22.1f}")
    print(f"{'cost':>6} {'$' + format(jev_cost, '.5f'):>14} "
          f"{'$' + format(cla_cost, '.5f'):>22}")

    print()
    if wins > losses:
        print(f"  {args.jev_model} wins {wins}-{losses} "
              f"({st.mean(jev) - st.mean(cla):+.1f} points on average)")
    elif losses > wins:
        print(f"  {args.model} wins {losses}-{wins} "
              f"({st.mean(cla) - st.mean(jev):+.1f} points on average)")
    else:
        print(f"  Tied {wins}-{losses}")

    if args.lite:
        print("  The board waited for every answer, so this is decision "
              "quality alone.")
    elif args.seconds:
        print(f"  Equal time: {args.seconds:g}s each. A slower player simply "
              "got fewer moves.")
    elif stalled:
        for rows, name in ((jev_rows, args.jev_model), (cla_rows, args.model)):
            late = sum(r["missed_ticks"] for r in rows)
            turns = sum(r["steps"] for r in rows)
            if late:
                print(f"  {name} was late for {late} of {turns} turns "
                      f"({late / turns * 100:.0f}%); the snake ran on without it.")
    else:
        print("  Both answered every tick in time.")


if __name__ == "__main__":
    main()
