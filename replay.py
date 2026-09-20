#!/usr/bin/env python3
"""Re-render a race recorded by race.py. No API calls, no cost, any number of takes.

    python replay.py --file race.jsonl
    python replay.py --file race.jsonl --speed 4
    python replay.py --file race.jsonl --speed 0.5 --loop

Every move carries the timestamp at which it happened, so playback keeps the
real pacing of the race: whoever answered faster still pulls ahead on screen.
"""
import argparse
import json
import time

N = 10
HEAD, BODY, FOOD, EMPTY = "@", "#", "*", "."
COLUMN = 24        # characters reserved per player


def board_lines(snake, food):
    """The board as a list of text rows."""
    cells = {tuple(c): BODY for c in snake[1:]}
    if snake:
        cells[tuple(snake[0])] = HEAD
    if food:
        cells[tuple(food)] = FOOD
    return ["".join(cells.get((x, y), EMPTY) for x in range(N)) for y in range(N)]


def render(sides, order, seed, elapsed):
    """Every board side by side. Returns the frame as one string."""
    pad = " " * (COLUMN - N - 2)
    row = lambda cells: "  " + "  ".join(f"{c:<{COLUMN}}" for c in cells)
    out = [f"  SNAKE ARENA   seed {seed}   {elapsed:5.1f}s", ""]
    out.append(row(s["name"][:COLUMN] for s in (sides[n] for n in order)))
    edge = "+" + "-" * N + "+" + pad
    out.append(row([edge] * len(order)))
    grids = [board_lines(sides[n]["snake"], sides[n]["food"]) for n in order]
    for line in zip(*grids):
        out.append(row(f"|{g}|{pad}" for g in line))
    out.append(row([edge] * len(order)))
    out.append(row(f"score {sides[n]['score']:<3} moves {sides[n]['steps']:<4}" for n in order))
    out.append(row(f"{sides[n]['cost'] * 100:7.3f}c  {sides[n]['status']}" for n in order))
    return "\n".join(out)


def load(path):
    with open(path) as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    if not lines:
        raise SystemExit(f"{path} is empty")
    header = lines[0]
    if "players" not in header:
        raise SystemExit(f"{path} has no player header; record it with race.py")
    moves = sorted(lines[1:], key=lambda e: e["t"])   # threads append as they finish
    return header, moves


def play(header, moves, speed):
    order = [p["name"] for p in header["players"]]
    start = moves[0]
    sides = {n: {"name": n, "snake": start["snake"], "food": start["food"],
                 "score": 0, "steps": 0, "cost": 0.0, "status": "playing"}
             for n in order}
    print("\033[2J", end="")
    t0 = time.perf_counter()
    for move in moves:
        wait = move["t"] / speed - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)
        sides[move["who"]].update(
            {"snake": move["snake"], "food": move["food"], "score": move["score"],
             "steps": move["steps"], "cost": move["cost"],
             "status": move["dead"] or "playing"})
        print("\033[H" + render(sides, order, header["seed"], move["t"]), flush=True)
    return sides, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="race.jsonl")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    header, moves = load(args.file)
    if not moves:
        raise SystemExit(f"{args.file} holds no moves")
    while True:
        sides, order = play(header, moves, args.speed)
        print()
        for name in order:
            s = sides[name]
            print(f"  {s['name']:<26} {s['score']:>3} points  {s['steps']:>4} moves  "
                  f"{s['cost'] * 100:7.3f} cents  {s['status'] if s['status'] != 'playing' else 'alive'}")
        if not args.loop:
            return
        time.sleep(1.5)


if __name__ == "__main__":
    main()
