# Snake Arena

**Jev against six LLMs. One minute each.**

How many decisions can a model make in a minute, and what do they cost? This
benchmark puts **Jev**, TypeSafe's System One model, next to six general-purpose
LLMs on one concrete task: **score as many points as you can in sixty seconds of
Snake.**

Snake is a stand-in for a class of problem that is otherwise hard to compare
fairly. The board moves on a clock. Every tick needs one small decision out of
four options. Nobody gets to think for a while and catch up later: time spent
deciding is board you have already lost. That is the shape of a lot of real
automation work — routing a request, classifying an event, approving or holding
a transaction — where the decision is narrow, the volume is high, and latency is
part of the answer.

The question the repository is built to answer is not "which model is smartest".
It is **what changes when the decision model is built for decisions instead of
for text.**

## What is being compared

The seven players are not seven brands. Six of them are general-purpose LLMs;
the seventh was post-trained for a different output contract. That is the
comparison.

TypeSafe's own framing — its
[AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer) — puts
it as three branches off one tree. RLHF and RLVR are the field's established
terms; RLCD is TypeSafe's, for the branch it says it added:

| Branch | Objective | What it produces |
|---|---|---|
| **RLHF** | responses people prefer | chat models |
| **RLVR** | verifiable rewards, e.g. mathematics | reasoning models, "slower and more expensive" |
| **RLCD** | calibrated decisions | decision models |

All three start from a pretrained language model.

**The first two rows are not a partition of the field, and no model in this
lineup sits in only one of them.** What the vendors publish says so directly:

- Anthropic's recent Claude system cards describe post-training as using
  "a variety of techniques including reinforcement learning from human feedback
  (RLHF) and reinforcement learning from AI feedback", on models that *also*
  ship extended thinking
  ([system cards](https://www.anthropic.com/claude-opus-5-system-card)).
- Moonshot's [Kimi K3 report](https://github.com/MoonshotAI/Kimi-K3) describes RL
  across general, agentic and coding domains at *multiple reasoning-effort
  levels* — nine teacher models at three reasoning intensities, distilled back
  into one unified model.
- OpenAI ships Luna, Terra and Sol as capability tiers of a single family, with
  reasoning effort as a per-request parameter on each
  ([GPT-5.6 system card](https://deploymentsafety.openai.com/gpt-5-6)).

So deliberation, in all six, is **a dial on one model** rather than a separate
model from a separate branch. The real line in this lineup is whether that dial
exists at all:

| Entrant | How deliberation is controlled | What it did in the recorded run |
|---|---|---|
| `claude-haiku-4-5` | `thinking.budget_tokens` | off here; 9 output tokens, every call |
| `claude-opus-5` | `output_config.effort`, `low`..`max` | emits thinking only at `max`; nothing at `low` |
| `gpt-5.6-luna` | `reasoning.effort`, `none`..`max` | wrote on 10 of 43 moves |
| `gpt-5.6-sol` | `reasoning.effort`, `none`..`max` | wrote on 7 of 40 moves |
| `kimi-k2.7-code-highspeed` | `reasoning_effort` | wrote on all 34 moves |
| `kimi-k3` | `reasoning_effort` | wrote on all 14 moves |
| **`jev-latest`** | **no such parameter exists** | 45 output tokens, all 204 calls |

That last row is not a configuration choice. `POST /v1/systemone` accepts exactly
three fields — `state`, `model`, `questions` — and the published OpenAPI schema
contains no `effort`, `reasoning`, `thinking`, `budget`, `temperature` or
`max_tokens` anywhere in it. There is no dial to turn.

The split is the output contract, and it is what the race is actually measuring:

- A generating model **produces a sequence**. Structured output is a constraint
  applied on top of that generator: you give it a JSON schema and it produces
  text that satisfies it.
- A decision model **returns a distribution over an answer space you defined**.
  TypeSafe states it plainly: "the model does not generate text". There is no
  sequence to constrain and nothing to parse.

That difference shows up in the token counts. In the shipped replay Jev returned
**exactly 45 output tokens on all 204 calls**, standard deviation 0.00, while
`gpt-5.6-luna` ranged from 12 to 190 on the same task depending on whether it
stopped to think.

`output_shape.py` shows where the 45 comes from. It asks Jev the same questions
about a trivial state and a complicated one:

```
  question shape        out tok  out tok    in tok  in tok
                           easy     hard      easy    hard
  choice, 2 options          34       34       320     342
  choice, 4 options          50       50       352     374
  choice, 8 options          82       82       416     438
  noul, 1 question           20       20       279     301
  noul, 2 questions          36       36       291     313
  score, 3 levels            17       17       304     326
```

The output count never moves with the state; only the input count does. For a
Choice it is `18 + 8 x options`. **The size of the answer is fixed when you write
the question, before the model has seen anything** — which is also why TypeSafe
charges for input tokens and gives output tokens away. There is nothing there to
meter.

## What a System One model returns

Jev answers typed questions about a `state`. There are three question types; this
benchmark uses one:

| Primitive | Asks | Returns |
|---|---|---|
| **Choice** | pick one option from a set you define | the option, a probability for every option, and a confidence |
| Score | rate the state against ordered levels | a score, level probabilities, and a confidence |
| Noul | is this statement true? | a probability between 0 and 1 |

The name is a reference to Kahneman: System 1 is the fast, intuitive judgement,
System 2 the slow deliberate one. The design goal is the fast half.

Two properties matter for this benchmark:

- **The answer space is fixed by the caller.** Jev returns a distribution over
  the options you supplied, so there is no parsing step and no way to answer
  off-schema.
- **Every answer carries a confidence.** The documented definition, for a choice
  over `n` options with the winning option at probability `p`, is
  `(n * p - 1) / (n - 1)` — 1.0 when all the mass sits on one option, 0 when the
  distribution is flat. We verified this against all 204 answers in the recorded
  run; the largest disagreement was 0.017, which is rounding.

Calibration is a property of *groups* of predictions, not of any single answer:
outcomes given probability 0.8 should occur about 80% of the time. It is not a
guarantee that a particular move is right.

Documentation: [System One](https://docs.typesafe.ai/concepts/system-one),
[State](https://docs.typesafe.ai/concepts/state),
[Choice](https://docs.typesafe.ai/primitives/choice),
[Confidence](https://docs.typesafe.ai/confidence).

## The main experiment: sixty seconds each

Every player gets its own board from the same seed and sixty seconds of wall
clock. The board advances only when that player answers, so nobody is punished
for the network and nobody is rescued by it. A model that answers in 250 ms gets
roughly four times as many moves as one that answers in a second, and moves are
what points are made of.

```bash
python race.py --seed 1 --seconds 60 \
  --lineup jev,haiku,opus,luna,sol,kimi-fast,kimi \
  --out race_seven.jsonl
```

One recorded run ships with the repository as `race_seven.jsonl`. Seed 1, sixty
seconds, seven players, all still alive at the buzzer:

| Model | Points | Moves | Median latency | Total | Per move | Moves with a written rationale |
|---|---:|---:|---:|---:|---:|---:|
| `jev-latest` | **21** | **204** | **263 ms** | 0.591 ¢ | **0.0029 ¢** | 0/204 |
| `claude-haiku-4-5` | 9 | 73 | 776 ms | 4.491 ¢ | 0.0615 ¢ | 0/73 |
| `claude-opus-5` | 3 | 25 | 2031 ms | 9.624 ¢ | 0.3850 ¢ | 0/25 |
| `gpt-5.6-luna` | 5 | 43 | 1080 ms | 0.508 ¢ | 0.0118 ¢ | 10/43 |
| `gpt-5.6-sol` | 5 | 40 | 1309 ms | 8.058 ¢ | 0.2014 ¢ | 7/40 |
| `kimi-k2.7-code-highspeed` | 5 | 34 | 1724 ms | 8.519 ¢ | 0.2506 ¢ | 34/34 |
| `kimi-k3` | 2 | 14 | 4447 ms | 3.347 ¢ | 0.2391 ¢ | 14/14 |

Three models finished on exactly 5 points having spent 0.51 ¢, 8.06 ¢ and
8.52 ¢. A point cost Jev 0.03 ¢ and Opus 3.21 ¢.

**Read this as throughput, not intelligence.** Nobody crashed, and the per-move
decision quality is indistinguishable. Comparing raw points per move would be
unfair — it falls as the snake grows, so a model that only played the easy first
30 moves looks better per move than one that played 204. Comparing at the *same
move number* removes that:

| At move | Rival's score | Jev's score at that move |
|---|---:|---:|
| 73 (`claude-haiku-4-5`) | 9 | **10** |
| 43 (`gpt-5.6-luna`) | 5 | 5 |
| 40 (`gpt-5.6-sol`) | 5 | 5 |
| 34 (`kimi-k2.7-code-highspeed`) | 5 | 5 |
| 25 (`claude-opus-5`) | 3 | 3 |
| 14 (`kimi-k3`) | 2 | 2 |

Jev matches every rival move for move and is one ahead of Haiku. **The entire
score difference is how many moves each model got to make**, and that is latency.
On a task where decisions arrive on a clock, being built for decisions is worth
more than being able to reason about them.

### What the reasoning models spend their time on

Four of the seven return some form of written rationale, and the repository
records it. Reading those rationales turned up something about the *harness*
rather than the models: the state field `food_is` reports a boolean per
direction, so when the food sits diagonally, two directions are both `true` and
both reduce Manhattan distance by exactly one. They are genuinely tied. The rule
list says "pick the one closer to the food" and is silent on ties, so a model
that deliberates has to invent a tie-break.

`gpt-5.6-luna` writes a rationale on 10 of its 43 moves in the recorded run, and
all 10 of them land on one of these ties. That is a defect in the rule set, not
in the model, and it is the kind of defect that stays
invisible until something writes down why it hesitated.

### The same ambiguity, seen from the other side

Jev writes nothing. On those same tied positions it returns a flatter
distribution:

```
move 60   up 0.56   right 0.43   down 0.01   left 0.00   confidence 0.41
```

Across the run, confidence averages 0.901 with a median of 0.990, and **36% of
moves come back below 0.90**. The four least certain are near-coin-flips
(0.56/0.43, 0.56/0.43, 0.57/0.42, 0.60/0.40) — the same class of position the
reasoning models write paragraphs about.

The difference that matters for automation is what uncertainty costs. Jev's
median latency is 263 ms overall and 280 ms on its sixteen least certain moves:
**doubt is a smaller number in the same response, not a longer one.** For a
model that reasons, doubt is extra tokens and extra seconds: `gpt-5.6-luna` takes
1013 ms on a move it answers outright and 2038 ms on one it writes about.

### Turning that into a routing decision

TypeSafe documents
[confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing):
act automatically above a threshold, ask for confirmation in a middle band,
escalate below it. The recorded run is small enough to price that out exactly.
Escalation cost here is one `gpt-5.6-sol` call at its measured median latency
and per-move cost from this same race:

| Gate | Handled automatically | Escalated | Added latency | Added cost |
|---:|---:|---:|---:|---:|
| none | 204 | 0 | — | — |
| 0.50 | 198 | 6 | +7.9 s | +1.21 ¢ |
| 0.70 | 188 | 16 | +20.9 s | +3.22 ¢ |
| 0.90 | 131 | 73 | +95.6 s | +14.71 ¢ |

A gate at 0.70 hands 92% of the decisions to the fast model and buys a second
opinion on the 8% that were genuinely ambiguous. That costs 0.59 ¢ + 3.22 ¢ =
**3.81 ¢**, against 8.06 ¢ to run `gpt-5.6-sol` on every move — less than half
the price, for 204 decisions instead of 40. **This is the automation shape the
benchmark is really about:** a cheap calibrated model on the whole stream, an
expensive one on the tail that the cheap model flags itself.

The repository does not implement this gate. It records the confidence so the
question can be answered from data instead of asserted.

## The second experiment: decision quality with the clock stopped

The race conflates two things — how good a move is, and how many moves you get.
`arena.py` separates them. In `--lite` mode the board is frozen while the player
thinks, so latency does not matter and only the choice does.

```bash
python arena.py --lite --seeds 1 2 3 4 5
```

It also has a real-time mode where the snake keeps moving on its current heading
while a player thinks, and it refuses to declare a winner if anyone missed a
tick, on the grounds that the result would then be about speed rather than play.
This mode is currently two-player (Jev against one Claude model) and is the
older of the two harnesses; the seven-way race is the one the repository is
built around.

## How the comparison is kept fair

**One view, built by code that does not know who is asking.** `snake.py` builds
the state and the per-direction descriptions once. It has no notion of which
player will consume them, so every model sees byte-identical input.

**One answer space.** Jev is constrained by the `criteria` of a Choice question;
the Anthropic, OpenAI and Moonshot players are constrained by a JSON schema with
the same four-value enum and `strict: true`. No player can answer off-schema, so
no move is ever lost to a parser.

**All four directions are always offered, including the illegal ones**, described
as such ("INSTANT DEATH: wall in that square"). Filtering them for one player and
not the others would break the shared view. It also keeps Jev's confidence on a
constant four-option scale for the whole run, which matters because the
confidence formula is normalised by option count.

**The resolved build is logged.** `jev-latest` is an alias that moves when a new
version ships, and the other providers resolve aliases too. Every run prints
which build actually answered — `jev-1.13.0`, `claude-haiku-4-5-20251001` — so a
recorded result stays interpretable later.

**Reasoning effort is stated, not hidden.** The defaults in `race.py` are what
produced the shipped replay: `low` for Opus 5 and both Kimis, `high` for Luna and
Sol. Running the command above with no effort flags reproduces that configuration.
Opus 5 turns out to emit thinking blocks only at `max` effort, so it records no
rationale at all here; that is a measured result, not a misconfiguration.

## Running it

Python 3.9 or newer. No dependencies — each of the four API clients is under
sixty lines of `http.client`, which keeps the request shape visible and
structurally identical across providers. TypeSafe publishes
[Python and JavaScript SDKs](https://docs.typesafe.ai/sdk) that you would
normally prefer in an application.

```bash
cp .env.example .env      # fill in the providers you want to race
set -a && . ./.env && set +a
python race.py --seed 1 --seconds 60 --lineup jev,haiku
```

Only the providers you actually name in `--lineup` need a key.

| Entrant | Model | Key |
|---|---|---|
| `jev` | `jev-latest` | `TYPESAFE_API_KEY` |
| `haiku` | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| `opus` | `claude-opus-5` | `ANTHROPIC_API_KEY` |
| `luna` | `gpt-5.6-luna` | `OPENAI_API_KEY` |
| `sol` | `gpt-5.6-sol` | `OPENAI_API_KEY` |
| `kimi-fast` | `kimi-k2.7-code-highspeed` | `MOONSHOT_API_KEY` |
| `kimi` | `kimi-k3` | `MOONSHOT_API_KEY` |

`.env` is gitignored. Keys are read from the environment and never written to
disk or into a replay file.

Replay a recorded race in the terminal, with no API calls and no cost:

```bash
python replay.py --file race_seven.jsonl --speed 4
```

## What this does and does not show

**One seed, one run, one task.** The numbers above are a single sixty-second
race. Latencies move with time of day and load; none of these providers is
deterministic. Treat the ordering as indicative and re-run before relying on a
margin.

**Snake is a proxy, and a narrow one.** It rewards a fast, local, four-way choice
with full information and no memory. Work that needs multi-step planning, tool
use, or an explanation a person will read is not what this measures, and Jev does
not do those things.

**The harness is part of the result.** Every version of this benchmark that
fixed a defect in the shared view moved every player's score. The replay records
each rationale and each confidence so that the next such defect is findable in
the data, rather than silently attributed to a model.

**Prices are a local table.** Costs come from `PRICES` in `players.py`, taken
from each provider's published rates. Update it when they change.

## Files

| File | Purpose |
|---|---|
| `snake.py` | game rules and the one shared view every player receives |
| `players.py` | the four API clients, the price table, the shared answer schema |
| `race.py` | the main experiment: N players, sixty seconds each, writes a replay |
| `arena.py` | the secondary experiment: decision quality, clock paused or running |
| `replay.py` | re-render a recorded race in the terminal |
| `output_shape.py` | shows that a System One answer's size follows the question, not the state |
| `race_seven.jsonl` | the recorded run the tables above are computed from |

## License

MIT — see [LICENSE](LICENSE).

Not affiliated with TypeSafe, Anthropic, OpenAI or Moonshot AI. Model names are
the property of their respective owners.
