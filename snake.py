"""Snake: game rules and the state view every player receives.

No model code lives here. The view built by `describe()` is byte-identical for
every player, so any score difference comes from the decision, not the prompt.
"""
from collections import deque
import random

N = 10
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


class Game:
    """A 10x10 Snake board. The snake starts length 3, moving right."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        mid = N // 2
        self.snake = deque([(mid, mid), (mid - 1, mid), (mid - 2, mid)])
        self.dir = "right"
        self.food = self._place_food()
        self.steps = 0
        self.score = 0
        self.dead = None

    # ---------------------------------------------------------------- board
    def _place_food(self):
        free = [(x, y) for x in range(N) for y in range(N) if (x, y) not in self.snake]
        return self.rng.choice(free) if free else None

    def _cell(self, cell) -> str:
        """What occupies `cell` next tick: 'wall', 'body' or 'free'."""
        x, y = cell
        if not (0 <= x < N and 0 <= y < N):
            return "wall"
        body = set(self.snake)
        if len(self.snake) > 1:
            body.discard(self.snake[-1])      # the tail vacates this tick
        return "body" if cell in body else "free"

    def reachable(self, cell) -> int:
        """Flood fill from `cell`. Small numbers mean a pocket you cannot escape."""
        if self._cell(cell) != "free":
            return 0
        body = set(self.snake)
        body.discard(self.snake[-1])
        seen, queue, n = {cell}, deque([cell]), 0
        while queue:
            cx, cy = queue.popleft()
            n += 1
            for dx, dy in DIRS.values():
                nb = (cx + dx, cy + dy)
                if 0 <= nb[0] < N and 0 <= nb[1] < N and nb not in body and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return n

    # ----------------------------------------------------------------- play
    def step(self, direction: str) -> None:
        if direction is None or direction == OPP[self.dir]:
            direction = self.dir              # a reversal is ignored, as in the arcade game
        self.dir = direction
        dx, dy = DIRS[direction]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        x, y = head
        if not (0 <= x < N and 0 <= y < N):
            self.dead = "wall"
            return
        if head in set(list(self.snake)[:-1]):
            self.dead = "self"
            return
        self.snake.appendleft(head)
        self.steps += 1
        if head == self.food:
            self.score += 1
            self.food = self._place_food()
            if self.food is None:
                self.dead = "board full"
        else:
            self.snake.pop()


# ------------------------------------------------------------------- view
STRATEGY = (
    "You are the snake in a game of Snake. Choose the direction of the next move.\n"
    "STRATEGY: your biggest risk is not starving, it is TRAPPING YOURSELF. Each "
    "option states how many squares you could still reach if you move there. When "
    "that number is smaller than your length, it is a dead end: you will enter it "
    "and die against your own body within a few turns.\n"
    "RULES, in priority order:\n"
    "1. Never choose a blocked direction (wall or body): that is instant death.\n"
    "2. Never choose a direction whose reachable count is below your length.\n"
    "3. Among the directions that survive 1 and 2, pick the one closer to the food.\n"
    "Prefer surviving over eating: a long game eats more than a short one."
)


def describe(g: Game) -> dict:
    """The board as every player sees it: facts only, never a recommendation."""
    hx, hy = g.snake[0]
    fx, fy = g.food
    options = {}
    for d, (dx, dy) in DIRS.items():
        nb = (hx + dx, hy + dy)
        if d == OPP[g.dir]:
            options[d] = "FORBIDDEN: that reverses onto your own neck."
            continue
        occupied = g._cell(nb)
        if occupied != "free":
            options[d] = f"INSTANT DEATH: {occupied} in that square."
            continue
        room = g.reachable(nb)
        closer = abs(nb[0] - fx) + abs(nb[1] - fy) < abs(hx - fx) + abs(hy - fy)
        safety = (f"TRAP: only {room} squares reachable and you are {len(g.snake)} long."
                  if room < len(g.snake) + 1
                  else f"Safe: {room} squares reachable (you are {len(g.snake)} long).")
        options[d] = ("Free. " + safety
                      + (" Moves TOWARD the food." if closer else " Moves AWAY from the food."))
    state = {
        "board": f"{N}x{N}",
        "snake_length": len(g.snake),
        "current_direction": g.dir,
        "food_is": {"up": fy < hy, "down": fy > hy, "left": fx < hx, "right": fx > hx},
        "if_you_move": {d: g._cell((hx + dx, hy + dy)) for d, (dx, dy) in DIRS.items()},
        "reachable_if_you_move": {d: g.reachable((hx + dx, hy + dy))
                                  for d, (dx, dy) in DIRS.items()},

    }
    return {"state": state, "options": options}

