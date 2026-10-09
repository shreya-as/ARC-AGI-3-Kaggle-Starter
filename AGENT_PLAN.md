# ARC-AGI-3 Agent Plan

Working plan, started 2026-10-02. The repo is back to the random starter
(`agent/my_agent.py`); earlier agent versions, score dumps and research notes
were deleted. Every number below was measured on this machine on 2026-10-02
unless marked otherwise. The six workstreams are §2 (why 0 levels), §3 (game
mechanics), §4 (analyzing frames), §5 (planning the agent), §6 (designing
experiments) and §7 (paper reasoning). §8 covers how to use Opus 5.5 at
High/Max effort for each one.

---

## 0. Summary

**Where we are.** Run the way Kaggle runs it, the starter clears **0 of 183
levels** on the 25 public games (ARC score 0.00). "The way Kaggle runs it"
means a competition-mode gateway over HTTP, with one thread per game. Each
game stops after 81 actions. A second run cleared sp80 level 1 by luck
(ARC 0.096). The starter seeds from the clock, so its results vary from run to
run.

**Why.** This is not a bug. It is arithmetic (details in §2):

1. The cap is 81 actions per game. A first-time human needs a median of 30
   actions (range 7–78) for level 1 alone.
2. Random play is hopeless at any realistic budget. With 3,000 actions per
   game (37× the cap) it cleared level 1 in at most 7 of 25 games, at a median
   35–56× the human action count. At that cost a level scores 0.03–0.08 points
   out of 100.
3. The starter samples all 7 actions and ignores `available_actions`. In the 6
   click-only games, 6 of every 7 actions are wasted.
4. Each attempt at a level has a hard action budget, usually drawn as a 1-pixel
   bar along a screen edge. Running out means GAME_OVER, and the level
   restarts. Budgets range from 13 to 1,024 actions, depending on the game and
   level, and are typically 1–4× the human baseline. Random play burns through
   them.
5. The starter keeps no memory, so nothing it observes changes its next action.

**Where the points are.** Each level scores `min(1.15, (human/agent)²)`.
Levels are weighted by their number, and an unfinished level scores 0. Two
reference points from the public games:

- Clearing **only level 1 in every game at human efficiency** is worth just
  **3.5%**.
- Clearing **3 levels per game at 2× human cost** is worth 5.3%.

So the points are in levels 2+. The agent must learn a game's rules on level 1
and reuse them on later levels.

**Plan in one sentence.** Build a game-agnostic, object-centric explorer that:

1. never spends actions on illegal or no-op inputs;
2. learns what each action does from a few cheap probes;
3. plans with that learned model instead of walking randomly;
4. carries the rules it learned, and its hypothesis about what triggered the
   level-up, into the next level.

Measure it in a local competition-mode simulator, on held-out games, before
spending Kaggle submissions.

**Do first.**

1. Commit the competition-mode simulator (Appendix A) as `scripts/kaggle_sim.py`.
   Every local number from now on is measured the way Kaggle measures it.
2. Ship the safety shell (§5.3-A):
   - legal actions only;
   - crash-proof stepping;
   - stable per-game seeding;
   - bounded frame memory;
   - thousands of actions per game, guarded by a wall-clock limit.
3. Build perception with HUD/budget-bar masking (§4), plus the
   probe-each-action-once explorer (§5.3-D).
   Target: level 1 cleared in at least 8 of the 18 dev games.

---

## 1. Ground truth: how the competition actually runs

Sources checked:

- the framework in `vendor/ARC-AGI-3-Agents`;
- `arc_agi` 0.9.9 and `arcengine` 0.9.3 in `.venv`;
- docs.arcprize.org (methodology, actions, competition mode; fetched 2026-10-02).

### 1.1 Execution path on Kaggle

```
Kaggle rerun -> notebook cell 3 -> python main.py --agent myagent
  main.py : GET /api/games from the gateway  (hidden, unseen games)
  Swarm   : opens ONE scorecard, makes one env + one MyAgent per game,
            starts ALL games at once, one thread each
  Agent.main():
      while not is_done(...) and action_counter <= MAX_ACTIONS:
          action = choose_action(frames, latest_frame)
          frame  = take_action(action)        # HTTP POST to the gateway
```

Facts that shape the design:

- **Action cap.** `MAX_ACTIONS` defaults to 80. The loop is inclusive, so a game
  gets 81 actions.
- **Animation frames.** `latest_frame.frame` is a **list** of 64×64 grids: every
  animation frame one action produced. In the public games this is usually 1,
  and up to 118 for an sb26 program run. The resulting state is `frame[-1]`.
- **No engine to copy.** Online, the env is a `RemoteEnvironmentWrapper`. Any
  planner that deep-copies the engine works locally and fails on Kaggle.
- **One error kills a game.** Any gateway error makes the wrapper return
  `None`. `Agent._convert_raw_frame_data` then raises `ValueError`, and that
  game's thread dies. One example: sending a non-RESET action while the state is
  GAME_OVER or WIN returns HTTP 400 (stated in the docs).
- **Shared CPU.** All games run concurrently in one Python process, so the cost
  of each step is shared by ~25+ agents.
- **Unbounded recording.** The framework appends every frame to
  `self.frames`, and with `record=True` writes it as JSON to `RECORDINGS_DIR`.
  On Kaggle that directory is `/kaggle/working/server_recording`. Both grow
  without bound.
  - **Measured:** 2.3 MB of JSON per game for 81 actions on average, and
    8.2 MB for sb26 because of its 42-frame animations.
  - **Extrapolated:** at 5,000 actions per game, roughly 3.5 GB per run, plus
    the same frames held in RAM as Python lists (about 36 KB per grid).
  - At 81 actions this is harmless. At thousands of actions it costs time,
    disk and memory.
- **Local cap can differ from Kaggle's.** `scripts/play_local.py` sets
  `MAX_ACTIONS = min(MAX_ACTIONS, --max-steps)`, so the two can silently diverge.
- **No outside help.** There is no internet during evaluation, so external LLM
  APIs are unusable.

### 1.2 Actions and RESET

| Action | Meaning (docs) |
| --- | --- |
| RESET | start / restart |
| ACTION1–4 | up / down / left / right ("semantically mapped"; game-dependent) |
| ACTION5 | interact / select / rotate / attach / execute |
| ACTION6 | click at (x, y), 0–63 display coordinates |
| ACTION7 | undo |

Engine rule (`arcengine/base_game.py`, `handle_reset`): RESET with no action
taken since the level started is a full game reset back to level 1. Otherwise
it restarts the current level.

The local `play_local.py` runner is **not** in competition mode. A RESET at
level start there really does wipe your progress back to level 1. That's one
more reason to measure only in the simulator.

### 1.3 Competition mode (enforced on Kaggle)

- **One scorecard, one `make` per game.** There is no second copy to simulate
  on, and no fresh play in which to replay a solution.
- **Game resets become level resets.** A RESET sent on a level's first step
  returns the current observation. It still counts as an action (`arc_agi/api.py`).
- **No scorecard mid-run.** Reading it returns 403. The agent only sees
  `levels_completed`, `state`, `available_actions` and frames.
- **Unplayed games score 0.** Scoring covers every available environment.

### 1.4 Scoring (from `arc_agi/scorecard.py` and the methodology page)

- **Level score:** `100 × min(1.15, (h / a)²)`.
  - `h` is the human baseline for that level: the upper median, by fewest
    actions, of first-time players. These are `baseline_actions` in each
    game's `metadata.json`.
  - `a` is **every** action taken while on that level, including failed
    attempts, RESETs and undos.
  - An unfinished level scores 0.
- **Game score:** the weighted mean of its level scores, where the weight is the
  level number (1..N).
- **Total score:** the mean of the game scores.

Score ceilings on the 25 public games:

| First k levels cleared in every game | 1× human | 2× | 3× | 5× | 10× |
| --- | ---: | ---: | ---: | ---: | ---: |
| k = 1 | 3.52 | 0.88 | 0.39 | 0.14 | 0.04 |
| k = 2 | 10.57 | 2.64 | 1.17 | 0.42 | 0.11 |
| k = 3 | 21.14 | 5.29 | 2.35 | 0.85 | 0.21 |
| k = 4 | 35.24 | 8.81 | 3.92 | 1.41 | 0.35 |
| k = 6 | 73.99 | 18.50 | 8.22 | 2.96 | 0.74 |
| all levels | 100 | 25.00 | 11.11 | 4.00 | 1.00 |

What this implies:

1. **Depth beats breadth.** On average, level 1 carries 3.5% of a game's weight.
   The money is in levels 2+.
2. **Efficiency is quadratic.** At 2× human cost a level keeps 25% of its value;
   at 10× it keeps 1%.
3. **Exploration only taxes its own level.** Actions spent on level k never touch
   the count for level k+1. Stopping early earns nothing, so `is_done` should
   only return true on WIN or when the wall clock runs out.
4. **Deaths are charged to the level.** Dying and retrying is allowed. Every
   wasted attempt lowers that level's score but not the others.

---

## 2. Why the agent gets 0 levels

Evidence comes from two sources:

- the competition-mode simulator in Appendix A, which runs the real framework
  `Swarm` over HTTP with all games in parallel threads (25 games took 16 s);
- offline probes against the same engine. Raw numbers are in Appendix B.

### 2.1 Root causes for the starter

| # | Cause | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | 81-action cap | Simulator: 25/25 games stop at 81 actions. Run 1 cleared 0 levels; run 2 cleared 1 (sp80). The level-1 human baseline has a median of 30 and runs 7–78. | MAX_ACTIONS in the thousands, plus a wall-clock guard (§5.3-A) |
| 2 | Random play does not scale | 3,000 actions × 3 seeds. Starter policy: L1 in 5/25 games, median step 1,780 (48× human). Legal-only: 6/25, 884 steps (35×). Legal-only plus clicks on objects: 7/25, 1,170 steps (56×). Max 2 levels in any run. | Model-based exploration (§5) |
| 3 | Ignores `available_actions` | Click-only games (ft09, lp85, r11l, s5i5, tn36, vc33): 6/7 of actions wasted. 4-action keyboard games (ls20, tr87, tu93): 3/7 wasted. Wasted actions still count toward the score. Unavailable actions are not even inert: they drain the attempt budget in 6 games (r11l, ft09, bp35, g50t, cn04, dc22). In bp35, unlisted ACTION1/2 walk left; in lf52, an unlisted ACTION5 works. | Only sample `latest_frame.available_actions` |
| 4 | Clicks on random pixels | Click targets are small: 32 of 4,096 px for vc33's level-1 buttons, about 2% of the screen for lp85's buttons, 1.6% for dc22's. In vc33, 80 random actions hit a button about 0.09 times on average, and level 1 needs 3 hits. | Click connected components, one probe per object class (§5.3-D) |
| 5 | Per-attempt budgets | Each decoded game has a per-level budget (13–1,024 actions, refilled on level restart). Repeating one action reaches GAME_OVER within 16–200 actions in 21/25 games. The other 4 charge only for effective actions, such as moves that aren't blocked or clicks that hit something. Random play spends most of its actions in attempts that end in GAME_OVER. | Budget-aware planning that reads the bar (§4.2) |
| 6 | No memory | Every decision is independent, so the agent can never stop repeating a failure | Online effect model and state graph (§5.3-C/D) |
| 7 | The HUD makes every frame "new" | The budget bar changes as actions are spent. In 10 of the 19 games with clicks, 100% of random clicks "changed the frame". Mostly that was the bar; r11l is the exception, because a floor click really moves the selected leg. In cd82, ACTION1/2 change one HUD pixel and nothing else. | Mask the HUD before hashing or measuring novelty (§4.2) |

The starter is not exactly zero. Over 300 episodes per game of its own policy
(80 actions):

- sp80: level 1 cleared in 31% of episodes. Its level 1 is three right moves
  followed by ACTION5.
- cd82: 2.7%.
- m0r0: 0.3%.
- every other game checked (60–300 episodes each): 0.

So its expected score on the public set is about 0.03–0.1. A leaderboard 0
just means no sp80-style luck happened on the hidden games.

### 2.2 Kaggle-only failure modes (traps for any future agent)

| Trap | Consequence | Guard |
| --- | --- | --- |
| Engine cloning (deepcopy of `env._game`) | Works locally. Online there is no engine, so the planner silently degrades or crashes. | Never touch `arc_env` internals inside the agent. Run every evaluation through the simulator. |
| Game-id branches, opening books, saved traces | Worthless on hidden games, and they inflate local scores | A test that greps the agent for public game ids |
| A non-RESET action while GAME_OVER/WIN | HTTP 400, then `None`, then `ValueError`; the game's thread dies | Gate on state, and wrap `take_action` in try/except |
| Planning on a full reset | Disabled in competition mode; the call still costs an action | Never rely on a full reset |
| Heavy per-step compute | 25+ threads share one interpreter, and the runtime limit is unknown | Per-step time budget, measured under the simulator |
| `hash(game_id)` seeds, global `random.seed` | `hash()` is salted per process, and the global RNG is shared across threads, so runs can't be reproduced | `random.Random(zlib.crc32(game_id.encode()))` for each agent |
| Unbounded `self.frames` and recorder | RAM and disk blow-up at large MAX_ACTIONS | Override `append_frame` / `start_recording` (§5.3-A) |
| Mutating the shared `GameAction.ACTION6` singleton across threads | Click data could, in theory, leak between games | **Measured: 0 mismatches in 4,989 concurrent clicks.** Low risk. Keep `set_data` immediately before `return`. |

### 2.3 Lessons from earlier iterations

These numbers are quoted from `AGENT_RESEARCH_LOG.md`, `OBSERVED_MODEL.md` and
`STATE_HASH_EVALUATION.md`, read before they were deleted. They were measured
with the old local runner, not the simulator.

- **Engine-dependent results don't transfer.** The best local results (≤ 5/183
  levels at 120 actions, local ARC ≤ 0.86) depended on engine-copy BFS/MCTS and
  on static per-game traces. Neither exists on Kaggle.
- **Without them, little remained.** The generalizing variants reached 1–3/183
  levels. The observed-transition model managed 3 levels per trial, with a mean
  local ARC of 0.23 over 2 seeds.
- **Exact full-frame hashing scored 0/183.** It is now explained: the budget bar
  makes every frame unique, so novelty search degenerates. A coarse 16×16 hash
  goes too far the other way and merges states that differ.
- **Soft beats hard.** Hard state-action bans and aggressive priors caused
  regressions; soft biases survived.
- **Score reports mixed best-of-trials with means.** From now on, report the
  mean and SD over seeds only.

---

## 3. Understanding game mechanics

### 3.1 Method and ground rules

- **How the games were decoded.** The 25 game sources have obfuscated names.
  Five parallel Opus subagents decoded them, five games each. They
  cross-checked every win and loss claim by running the game in memory, and
  found scripted level-1 solutions for 15 of the 25 games.
- **Purpose.** Learn the *genre* so the agent can carry the right priors. Do
  **not** write solutions: the hidden games are new.
- **Ground rule.** The agent contains no game ids, no per-game constants and no
  stored solutions. The guard test in §2.2 enforces this.
- **Dev / holdout split.**
  - 18 dev games.
  - 7 **sealed holdout** games: dc22, g50t, lp85, s5i5, sc25, sk48, tr87. That
    is 2 click-only, 2 keyboard-only and 3 mixed games, 49 levels in total.

  The holdout cards were decoded too. They sit in a collapsed block at the end
  of this file. Don't open them, or give them to Claude, before the design
  freeze (M3, about Oct 23). Once read, a game no longer tests generalization.

### 3.2 Dev-game catalog

Column key:

- **Inputs:** A1–A7 = ACTION1–7.
- **Budget:** the level-1 action budget per attempt, refilled on every level
  restart.
- **L1 solve:** the shortest verified level-1 solution, with the human baseline
  in brackets.

| Game | Legal | Inputs | Win | Budget (L1) and other losses | L1 solve (human) |
| --- | --- | --- | --- | --- | --- |
| ar25 | 1–7 | A1–4 move the selected piece or mirror 1 cell; A5 cycles selection; A6 selects; A7 undo | Every target cell is covered by a piece or by its mirror reflection | 64 energy: moves and A5 cost; clicks, undo and blocked moves are free | 15 (32) |
| bp35 | 3,4,6,7 | A3/A4 walk, falling off ledges; A6 clicks a block to break it, toggle it, or flip gravity; A7 undo, which costs a step | The player reaches the gem, usually off-screen while the camera scrolls | 64 steps; spikes; a rising flood | — (21) |
| cd82 | 1–6 | A1–4 move a paint basket around an 8-slot ring; A5 pours a half or a triangle; A6 picks a color swatch or the nozzle | The canvas equals the 10×10 target image (diagonals ignored), checked after a pour | 99 actions | — (55) |
| cn04 | 1–6 | A6 selects a jigsaw piece; A1–4 move it 1 cell; A5 rotates it 90° | No piece has an unmatched connector. Each connector must overlap one of the same hidden type; the two types are drawn identically, so a board can look solved without winning | 75, counting every action, including off-target clicks and A7 | — (29) |
| ft09 | 6 | A6 cycles a tile's color; on L5–6, pattern tiles also cycle their neighbors | Every clue glyph's neighbor rules hold: a neighbor equals the clue's center color where the clue's border pixel is 0, and differs elsewhere | 32 energy; unavailable actions also cost | 4 (43) |
| ka59 | 1–4,6 | A6 selects a green-bordered block; A1–4 move it 3 px; moving into a box kicks the box 15 px | Every goal frame holds a matching block or box, inset 1 px | 100: every move and every click costs | 11 (28) |
| lf52 | 1–4,6,7 | A6 on a peg, then on a jump marker, makes the peg jump; A1–4 slide rail platforms; A7 undo (refunds the budget) | 1 non-blue peg remains (2 on L6–7) | 64, on a hidden counter (the row-0 bar lags) | 8 (32) |
| ls20 | 1–4 | A1–4 move the player one 5-px cell; changer tiles cycle the player's shape, color or rotation | The player stands on each goal while carrying that goal's key icon (shape × color × rotation) | 42 moves per life, 3 lives (GAME_OVER at move 129); losing a life resets the level state | — (22) |
| m0r0 | 1–6 | A1–4 move two players, one of them mirrored; A6 selects a block to move instead | The two players merge | 150; hazards reset positions without refunding the budget | ≥ 15 by BFS (30) |
| r11l | 6 | A6 on a leg selects it; A6 elsewhere teleports it there; each body sits at the centroid of its legs | Every body overlaps its goal (color mixing on L5–6) | 60, counting any action id; 5 hazard hits | 3 (22) |
| re86 | 1–5 | A1–4 move the selected shape 3 px; A5 cycles selection | Every target dot lies under a line of the same color | 100; walls deform shapes on later levels | 20 (26) |
| sb26 | 5,6,7 | A6 selects a palette token, then a slot to place it (or swap); A5 runs the program; A7 undo (no refund) | The program's output equals the target color row | 64 energy, spent on placements and runs | 9 (18) |
| sp80 | 1–6 | A6 selects a bar or L-piece; A1–4 move it; A5 runs the liquid | The flow fills every cup without spilling | 30; the 5th failed run loses; some levels are rotated 180° | 4 (39) |
| su15 | 6,7 | A6 pulls in objects within 8 px; fruits of the same size merge, 2048-style; chasers roam; A7 undo | The goal zones hold the counts the recipe asks for | 32 clicks; penalties; losing all fruit | 7 (22) |
| tn36 | 6 | A6 toggles program bits; a Run button executes the program; tabs run demos | After a run, the piece's position, scale, rotation and color equal the target's | 61 clicks | 7 (32) |
| tu93 | 1–4 | A1–4 move one maze node along a link (8-frame slide); an invalid direction still costs a step | The player is on the exit after the enemies move | 50 steps; sentries, patrols and chasers kill on contact | — (19) |
| vc33 | 6 | A6 on a button pumps fluid between two paired tanks; A6 on a glowing gate swaps the floaters | Every floater is level with its marker of the same color | 50 clicks | 3 (7) |
| wa30 | 1–5 | A1–4 move one 4-px cell and set the player's facing; A5 grabs or releases the box in front, or deletes a thief | Every box is inside a target and none is being carried | 200 steps; helper and thief NPCs move every turn | — (71) |

### 3.3 Cross-game regularities and the priors they justify

| # | Regularity (public games) | Prior / agent behavior |
| --- | --- | --- |
| R1 | **Deterministic.** No randomness affects play (tr87 uses fixed per-level seeds; lf52's RNG is cosmetic). A level restart reproduces the same start state. | Assume deterministic dynamics. A recorded action sequence replays exactly after a level restart. Two different outcomes for the same `(state, action)` signal hidden state. |
| R2 | **Every game has a per-level action budget** (13–1,024), refilled on restart. It is almost always drawn as a 1–2 px bar on a screen edge, with quirks (§4.1). Budget exhaustion is by far the most common GAME_OVER. Some human baselines exceed a single attempt's budget (wa30 L2: 119 vs 70; cn04 L4: 300 vs 125; re86 L7: 424 vs 300), so the humans retried too. | Mask edge bars. Estimate the remaining budget from the bar. Plan to fit. Dying once to learn is acceptable; the baseline already includes such retries. |
| R3 | **The goal is drawn on screen.** Examples: a target image (cd82), a reference arm (sk48), outline frames (ka59), target dots (re86), target cells (ar25), wall markers (vc33), cups (sp80), a recipe panel (su15), clue glyphs (ft09), a target outline (tn36), a target color row (sb26), a gem or pad (bp35, g50t). | **Goal-by-example.** Find static, non-interactive regions whose colors and shapes resemble something the agent can manipulate. Define distance-to-goal as the mismatch, and plan to reduce it. |
| R4 | **A few control schemes recur.** (a) A1–4 move an avatar or the selected object. (b) A click selects, then arrows move (ka59, sp80, m0r0, ar25). (c) A5 cycles the selection (re86, ar25). (d) Pure click puzzles: toggle, cycle or pump (ft09, tn36, vc33), select-then-target (lf52, sb26, r11l), attract (su15). (e) Edit, then execute: a costly run or pour shows the result (sp80, sb26, tn36, cd82). | **Probe library.** Each arrow once. Each object class clicked once. A5 once. After a click that highlights something, try one arrow, since selection usually means move. Treat run-type actions as expensive tests. |
| R5 | **Rules are taught inside the level.** tn36 runs demos from its tabs, ft09 has tutorial graphics, ar25 flashes a hint on L2, and sc25 shows spellbook hints. | Click panel-like UI early and watch the animation frames: rules can be learned cheaply. |
| R6 | **Short exact solutions exist.** Level 1 takes 3–20 actions and is often well under the human baseline (ft09 4 vs 43, sp80 4 vs 39, tn36 7 vs 32). | Even planned play scores at most 1.15 per level, so speed on L1 matters less than *reaching* L2+. |
| R7 | **Actions animate** (up to 118 frames in an sb26 run). The intermediate frames show the dynamics: liquid flow, program execution, slides, kicks. | Diff frame by frame on run-type actions to find the first divergence from the goal. |
| R8 | **The display is not always fixed.** sp80 rotates the whole frame 180° on some levels (inputs follow the screen). vc33 changes the gravity axis per level. bp35 scrolls; lf52 pans. | Never hard-code "A1 = up on screen". Re-verify the direction mapping with one probe at each level start. Key states in world coordinates when the view scrolls. |
| R9 | **Hazards and enemies exist, but the budget dominates.** m0r0 resets positions; r11l allows 5 hazard hits; bp35 has spikes and a rising flood; g50t has doors and an enemy; su15 has chasers. tu93 has turn-based enemies that kill. ls20 gives 3 lives of 42 moves each, and losing a life resets the level state. dc22 falls cost 20 steps. wa30 has thief NPCs. | Learn death causes from the last contact event before GAME_OVER (or before a life loss), and avoid them afterward. Model turn-based NPCs as moving once per action. |
| R10 | **Unavailable actions are not safe.** They drain the budget in 6 games (r11l, ft09, bp35, g50t, cn04, dc22), and some still act (bp35 A1/2 walk left; lf52 A5 works). | Send only `available_actions`. |
| R11 | **Levels add mechanics progressively.** bp35 adds new block types at L3, L4 and L8. re86 adds paint pads at L4. sb26 adds call tokens at L4. sk48 adds head switching at L6. | Carry the model into the next level, but re-probe *new* object classes at every level start. |

### 3.4 Goal families: the hypothesis library for §5.3-E

| Family | Public examples | Distance-to-goal detector |
| --- | --- | --- |
| Reach | bp35 gem; tu93 exit; ls20 goal, which also needs a matching key (and holdout g50t, dc22) | Path length from the avatar to a unique static target. For ls20, also the mismatch between the avatar's state icon and the target's icon |
| Overlap / cover / fit | re86, ka59, r11l, ar25, wa30 boxes, cn04 connectors (and holdout s5i5, lp85) | Sum of displacements from each movable object to its best-matching target, by color and shape |
| Copy a reference | cd82, ft09 (and holdout sk48) | Cell mismatches between the editable region and the aligned reference |
| Equalize / merge / count | vc33 levels, m0r0 merge, su15 counts | Level gap, pairwise distance, or count deficit |
| Edit-then-execute | sp80, sb26, tn36 | Unknown until the run; read the run's animation |
| Rule / program induction | tn36 opcodes, sb26 calls (and holdout tr87) | Hardest family; deprioritize until the others work |

### 3.5 Open questions about mechanics

1. How far do the hidden games stray from these families? R1–R11 are a prior,
   not a guarantee. When no goal detector fires, the agent needs a
   novelty-driven fallback.
2. The dev set is small: 18 games, 6 families. Track performance by family to
   avoid tuning toward any single game.

---

## 4. Analyzing frames

### 4.1 Measured facts

**Format and colors.**

- One action usually returns 1 frame. Heavy animators:
  - sb26 program runs: 42–118;
  - lf52 clicks: 16–27;
  - sc25 moves: 22;
  - sp80 ACTION5: 22;
  - su15 clicks: 5–16;
  - cd82 ACTION5: 15;
  - tu93: 8;
  - g50t and ka59 kicks: 7;
  - bp35: 5.
- `frame[-1]` is the state. The intermediate frames show motion paths: sliding,
  falling, projectiles.
- Colors are the integers 0–15.
- The background is the mode color, and it differs per game. Seen so far: 0, 1,
  2, 3, 4, 5, 9, 10, 12.

**Viewports and scale.**

- Camera viewports run from 11×11 to 64×64, integer-upscaled:
  - ×5: m0r0;
  - ×4: sp80;
  - ×3: ar25, cn04;
  - ×2: ft09, lp85, vc33;
  - ×1: the rest.
- **Dummy cameras.** bp35 and lf52 declare an 8×8 camera but draw a native
  64×64 world with 6-px cells through an overlay.
- **Letterboxing.** Smaller viewports get a border: 9 px around a 45×45
  viewport, letterbox rows around 64×44.
- **Viewport sizes change between levels** (ka59, lp85, vc33, sp80, m0r0, dc22).
- **The movement step is not the camera scale.** Measured steps:
  - 2 px: dc22 avatar;
  - 3 px: ka59, re86;
  - 4 px: wa30;
  - 5 px: ls20;
  - 6 px: g50t, tu93, sk48.
- **Naive scale detection fails.** "Find the largest k with uniform k×k blocks"
  returned 1 for every scaled game except cn04, because HUD and border pixels
  are drawn at native resolution. Infer the cell from motion instead.

**The HUD: budget bars and counters.**

- Budget bars and counters sit along screen edges and change as budget is
  spent, whether or not the action did anything.
- In cd82, dc22, ka59, lf52, r11l, s5i5, sb26, sp80, tn36, tu93 and vc33,
  some repeated action changed only 60–64 pixels in total before GAME_OVER.
  That is exactly a 64-pixel bar draining while nothing else on screen
  changes.
- On the first action of a level, a single corner pixel often flips first:
  (63,63), (0,63), (63,0) or (0,0).
- **The bar is not always a faithful budget meter.** Decoded variations:
  - **Location:** row 0, row 1, row 53, row 63, column 0, column 63, or
    columns 62–63. m0r0 uses both row 0 and row 63, filling from opposite
    ends.
  - **Rate:** some bars move only on "costed" actions. Others move every
    second action (g50t, tr87), or 2 px per action (ft09).
  - **Direction:** bars can grow instead of shrinking (bp35, lf52).
  - **lf52:** arrow presses drain a hidden counter, and undo refunds it.
    The row-0 bar is redrawn only on clicks, so it jumps by the arrow presses
    made since the last click (measured: 6 presses, then one click, gave a
    6-pixel jump). Until then the frame shows no change.
  - **cd82:** 99 actions are spread over 64 pixels, so the bar loses a pixel
    on about 2 actions in 3.
  - **sp80:** the whole frame, HUD included, is rotated 180° on some levels.
  Learn the bar's direction and rate per game. Treat the reading as an
  estimate.

**No-ops are common.** These actions changed 0 pixels from the start state:

- ar25 ACTION7;
- g50t ACTION1, ACTION3 and ACTION5;
- m0r0 ACTION5;
- sk48 ACTION2 and ACTION7;
- wa30 ACTION5.

In sc25, moves animate for 22 frames and then return to the same final frame.
"Frame changed" is therefore not the same as "world changed" in either
direction.

### 4.2 Perception pipeline (target: under 2 ms per step in numpy)

1. **Ingest.** Read `g = frame[-1]` as an int8 64×64 array. Keep the
   intermediate frames only when a diff needs motion evidence.
2. **Layout, at each level start.**
   - Find the letterbox: constant border rows and columns.
   - Background is the mode color inside the play area.
   - Estimate the cell size from the GCD of color-run lengths inside the play
     area.
   - Confirm the cell size later: one move of the controllable object should
     displace it by exactly one cell.
3. **HUD mask.** Flag pixels that:
   - change under every kind of action, including known no-ops;
   - change monotonically;
   - sit on edge rows or columns, or in a fixed panel.

   Learn the mask in the first ~5 actions and refine it online. Two side
   benefits:
   - The bar refills on a level reset or level-up, which confirms the mask.
   - Counting bar pixels gives the remaining attempt budget, an input to the
     planner.
4. **Objects.** Take connected components per color (4-connectivity) over the
   masked play area, at the logical cell scale. For each, record:
   - color;
   - bbox and area;
   - centroid;
   - a normalized shape hash.

   Group multi-color sprites by co-motion over time.
5. **State key for the exploration graph.** Hash the level index together with
   the masked play area at the logical scale. It is exact inside the play area
   and blind to the HUD. Optionally add an abstract key built from the object
   multiset.
6. **Diff into events.** Match components before and after an action by color,
   shape and nearest centroid. Emit:
   - `moved(obj, dx, dy)`, `appeared`, `disappeared`, `recolored`, `reshaped`;
   - global events: `level_up`, `game_over`, `bar_refill`, `noop`,
     `hud_only`.
7. **Roles.**
   - The **controllable object** is the component whose displacement tracks
     ACTION1–4 (expect up/down/left/right).
   - **Click affordances** are the object classes whose state changes when
     clicked.
   - **Hazards** are whatever was touched just before a GAME_OVER that was not
     caused by the budget.

### 4.3 Pitfalls

| Pitfall | Example | Mitigation |
| --- | --- | --- |
| HUD changes counted as progress | Budget bars; corner pixels | HUD mask (step 3) |
| Animation hides the net effect | sc25: 22 frames, net zero | Diff `frame[-1]` against the previous `frame[-1]` |
| Pixel coordinates ≠ logical cells | ×8 games: one cell is 8×8 px | Click cell centers; work at logical scale |
| Letterbox counted as an object | 45×45, 39×39, 64×44 viewports | Crop the play area first |
| Camera scrolls | bp35 follows the player vertically; lf52 pans on L3–9 | Detect a global translation; key states by world coordinates |
| Whole display rotated | sp80 is rotated 180° on some levels; vc33 changes the gravity axis | Re-learn the action-to-direction mapping at each level start |
| Visually identical states, different hidden state | Collected keys, counters | Include recent event history in the key when conflicts are detected (§5.3-C) |

### 4.4 Tooling (outside the agent)

- **`scripts/frame_dump.py`.** Runs any policy offline and writes, per step:
  - an upscaled palette PNG;
  - a 64-line hex ASCII grid (about 4 KB, pasteable into Claude);
  - a JSON event list.
- **Diff triptych.** A before / after / diff image for any step.
- **Ground-truth checker (offline only).** Reads `env._game` sprites and the
  level index to score perception accuracy, such as controllable-object
  precision and recall. It is never imported by the agent.
- **`game_gifs/`.** Already holds one GIF per public game, useful for human
  intuition.

---

## 5. Planning the agent

### 5.1 Principles

- **P1. Game-agnostic.** No game ids and no per-game constants. A test enforces
  this.
- **P2. Every action buys information or progress.** Never repeat an action
  whose outcome is already known to be useless.
- **P3. Learn once, reuse.** The per-game model persists across level resets and
  level-ups.
- **P4. Plan with the learned model, never with the engine.**
- **P5. Respect the attempt budget.** Plans must fit what's left of the bar.
- **P6. Fail safe.** Never crash and never send an illegal action. On timeout,
  degrade to a cheap policy.
- **P7. Stay cheap per step.** Measure under the simulator with 25 threads, then
  set the time limit.

### 5.2 Architecture

```
frame[-1] --> Perception --> ObjState ------------------+
               |   (HUD mask, budget, objects, events)  |
               v                                        v
           EffectModel  <---- transitions ----      StateGraph
     (action/target-class -> effect dist.)    (masked state keys, edges)
               |                                        |
     +---------+----------------+----------------------+
     v                          v                       v
  Explorer               GoalHypotheses            LevelMemory
 (info-gain probes)    (ranked win rules +     (what preceded level-ups,
                        distance-to-goal)        death causes, roles)
     +-------------> Planner (BFS/A* on learned model) <------+
                              |
                         SafetyShell ---> GameAction
```

### 5.3 Modules

**A. Safety shell (v0).**

- **Legal actions only.** The action set is exactly
  `latest_frame.available_actions`.
- **RESET rules.**
  - Send RESET if and only if the state is NOT_PLAYED or GAME_OVER.
  - On WIN, `is_done` returns True.
  - Never send any other action in GAME_OVER or WIN.
- **Crash-proof stepping.** Override `take_action` with try/except and return
  `None` on failure. After 3 consecutive failures, send RESET. If that fails
  too, stop the game.
- **Stable seeding.** Give each agent `random.Random(zlib.crc32(...))`. No
  global seeding.
- **Budgets.**
  - MAX_ACTIONS between 2,000 and 5,000, to be set by E1.
  - A per-game wall-clock budget: `is_done` returns True when it runs out.
- **Bounded memory.**
  - Override `append_frame` to keep only the last ~8 frames.
  - Override `start_recording` to skip agent-side JSON recording. The gateway
    records actions server-side anyway.
- **Hygiene.** Call `set_data` on ACTION6 immediately before returning it. No
  per-step prints of frames.

**B. Perception.** As specified in §4.2.

**C. Effect model.** Online; it persists across levels.

- **Key.** `(action, target_class)`, plus a small context: is the controllable
  object blocked in the action's direction? `target_class` is the clicked
  object's (color, shape), or `none` for simple actions.
- **Value.** Counts over effect types:
  - `noop`, `hud_only`;
  - `move(ctrl, dx, dy)`, `move(other)`;
  - `recolor`, `appear`, `disappear`;
  - `level_up`, `game_over`.
- **Determinism.** Assume it, and detect conflicts: if the same
  `(state_key, action)` gives two different outcomes, flag hidden state for
  that key.
- **Prediction.** Small local rules are enough for planning. The controllable
  object translates by one cell when the target cell is free; otherwise it is
  blocked or pushes.

**D. Explorer.** Information first.

1. **Probe** (≤ 6 + #classes actions):
   - Try each legal simple action once.
   - Click one representative cell of each object class, deduplicated by color
     and shape, rarest first.
   - Give up on a class after one no-op.
2. **Navigate and touch.** Once a controllable object is known, run BFS over
   the inferred free cells. Visit, cheapest first:
   - cells next to each object class, to touch or push them;
   - unexplored areas.

   This is Go-Explore's "return then explore", but done by moving along the
   learned map instead of resetting to a saved state.
3. **Interact.** Use ACTION5 or clicks on objects next to the controllable
   object. Try two-step sequences (select, then place) when a click leaves a
   highlight.
4. **Undo.** Use ACTION7 to revert harmful changes when that's cheaper than a
   level reset. Undo always costs a scored action. It refunds the attempt
   budget in only some games (§6.3, E9), so the agent must detect that online
   from the bar.
5. **Novelty.** Add a count bonus in masked-state space only, never on raw
   frames.

**E. Goal hypotheses.** One entry per goal family in §3.4: reach,
overlap/fit, copy a reference, equalize/merge/count, edit-then-execute, and
rule induction. Each entry has:

- a detector, giving a distance-to-goal heuristic on the current ObjState;
- a cheap test plan.

Most detectors start by finding the goal-by-example region (R3): static,
non-interactive content that resembles something the agent can manipulate.

Rank the families by their frequency in the dev games times the detector's
confidence, such as a reference panel being present. If no detector fires, fall
back to novelty-driven exploration (§5.3-D).

**F. Planner.**

- **Navigation and push puzzles.** BFS/A* over predicted states: the
  controllable object's position plus a few toggles. Use goal distance as the
  heuristic, and replan whenever a prediction is wrong.
- **Click-toggle puzzles.** Search over click sequences on object classes. If
  the effect model shows local XOR-style toggles, solve exactly with Gaussian
  elimination over GF(2).
- **Plan length.** A plan must fit the remaining attempt budget read from the
  bar.

**G. Level memory and transfer.**

- **On `level_up`, store:**
  - the event signature of the last k steps (who touched what, which pattern
    matched);
  - the effect model;
  - the role map, from (color, shape) to role.
- **On the next level:**
  - skip the probes for actions already known;
  - try the hypothesis that solved the previous level first;
  - probe only object classes not seen before.
- **On GAME_OVER:**
  - record the death cause: the budget, or the last contact event;
  - treat that cause as a hazard from then on;
  - keep everything else.

**H. Budget and time manager.**

- **Per attempt.** Track the remaining bar.
- **Per game.** Enforce a wall-clock cap. Reduce planner depth as decisions
  slow down.

### 5.4 Decision loop

```
choose_action(frames, latest):
    if latest.state in (NOT_PLAYED, GAME_OVER): return RESET
    obs = perceive(latest.frame[-1])                 # HUD-masked objects + events
    model.update(prev_obs, prev_action, obs, latest)  # effect model, graph, roles
    if latest.levels_completed > prev_levels: memory.on_level_up(trace)
    if plan and plan.still_valid(obs): return plan.pop()
    for h in goals.ranked(obs, memory):              # cheapest confident goal first
        p = planner.search(obs, h, budget=obs.bar_left)
        if p: plan = p; return plan.pop()
    return explorer.next_probe(obs, model)           # info-gain probe / frontier move
```

### 5.5 Staged build (exit criteria on the 18 dev games, 3 seeds, simulator)

| Version | Adds | Exit criterion |
| --- | --- | --- |
| v0 | Safety shell, legal-only random, budget ≥ 2,000 | 0 crashes; runtime measured; L1 cleared in ≥ 3/18 |
| v1 | Perception, HUD mask, probe explorer, object clicks, graph novelty | L1 ≥ 8/18; median L1 cost ≤ 10× human |
| v2 | Effect model, controllable object, BFS navigation, death causes | L1 ≥ 12/18; L2 in ≥ 2 games |
| v3 | Goal hypotheses, level transfer | L2+ in ≥ 6/18; simulator ARC ≥ 1.0 on dev |
| v4 | Freeze; run on the sealed holdout; harden runtime | Holdout ≥ 50% of dev score; Kaggle LB > 0 |
| v5 (optional) | Small offline VLM/LLM proposing goal hypotheses at level starts | Only if v3 plateaus and E11 shows runtime headroom |

These targets are educated guesses. Recalibrate them after v1.

### 5.6 Anti-patterns (each tried and measured before)

- engine cloning;
- game-id branches and opening books;
- disk memory across runs;
- exact full-frame hashing;
- coarse 16×16 hashing;
- hard action bans;
- best-of-trials reporting.

### 5.7 Packaging constraint

`scripts/build_notebook.py` splices only `agent/my_agent.py` into the notebook.
Either keep the agent in one file, or extend the builder to `%%writefile` each
module.

Dependencies:

- **numpy:** guaranteed, since `arcengine` depends on it.
- **scipy:** the Kaggle image has it, but confirm in the Phase A run before
  relying on `scipy.ndimage.label`. A numpy flood fill is a safe fallback.

---

## 6. Designing experiments

### 6.1 Infrastructure

1. **`scripts/kaggle_sim.py`** (Appendix A). Runs the competition-mode server
   plus the framework `Swarm` over HTTP. Start a fresh server for each run,
   because competition mode allows one scorecard per server.
2. **`scripts/eval.py`.** Runs N seeds × games through the simulator. The agent
   reads a `SEED_OFFSET` environment variable. Output JSON, per game:
   - levels and actions per level;
   - efficiency ratio (agent / human);
   - ARC score;
   - GAME_OVERs, crashes;
   - wall time and p95 decision time.

   Summary: mean ± SD over seeds.
3. **Offline ground truth.** Attempt budgets (Appendix B) and engine sprite
   roles. Used only to score perception and effect-model accuracy.
4. **Regression suite.** Every level ever cleared on dev must stay cleared.
   Re-run before each submission.

### 6.2 Protocol

- **Split.** 18 dev games and 7 sealed holdout games (§3.1). Tune only on dev.
  Run holdout only at M3/M4, and log every holdout run.
- **Statistics.**
  - Use ≥ 3 seeds per arm, with identical MAX_ACTIONS and wall-clock limits.
  - Paired per-game comparison with a Wilcoxon signed-rank test over games.
  - Adopt a change only if it wins on the seed mean **and** does not break the
    regression suite.
- **Pre-registration.** Write the hypothesis, primary metric and expected
  direction in the log *before* running.
- **Metrics.**
  - Primary: simulator ARC score, mean over seeds.
  - Secondary: levels cleared, L1 clear rate, median efficiency ratio,
    GAME_OVERs per level, p95 decision time.

### 6.3 Experiment backlog

| ID | Question | Arms | Primary metric | Decision rule |
| --- | --- | --- | --- | --- |
| E0 | Does the simulator predict Kaggle? | Simulator (dev+holdout) vs LB for every submission | Rank agreement, crash and timeout parity | If any Kaggle-only failure appears, fix the simulator first |
| E1 | How many actions can we afford? | MAX_ACTIONS 80 / 500 / 2,000 / 5,000 | Levels; wall time; RAM; disk | Largest value with p95 runtime inside the limit |
| E2 | Legal-action filtering (confirm in the simulator) | Starter vs legal-only | L1 clears at fixed budget | Adopt (expected) |
| E3 | HUD masking | Off / on | Distinct masked states per 100 actions; L1 clears | Adopt if it helps on ≥ 3 games and hurts none |
| E4 | Click targeting | Pixel-random / one per component / one per class | Click-game L1 clears; actions to first world change | Best arm |
| E5 | Probe first | No probe / probe phase | Actions to the first level-up | Adopt if the median falls |
| E6 | Controllable-object inference | Detector vs engine ground truth | Precision and recall on movement games | ≥ 0.9 before building v2 navigation |
| E7 | Navigation planner | Novelty walk / BFS on learned map | L1 efficiency ratio | Adopt if the ratio falls ≥ 30% |
| E8 | Cross-level transfer | Reset model at level-up / keep model + hypotheses | L2+ clears; L2+ ratio | Adopt if L2+ improves |
| E9 | Undo as an online probe | Move+undo; check that state is restored and whether bar pixels return | The agent detects "undo refunds budget" correctly | Decoding already answers this for the public games (below). The agent still has to detect it online for hidden games. |
| E10 | Death avoidance | Off / learned death causes | GAME_OVERs per cleared level | Adopt if lower with no clear loss |
| E11 | Runtime stress | 25 threads at the chosen MAX_ACTIONS | p95 decision ms; total wall time; RAM | Must fit the Kaggle runtime limit with ≥ 2× margin |
| E12 | Generalization | v3 vs v4 on the sealed holdout | Holdout ARC / dev ARC | Report honestly; it goes in the paper either way |

What decoding found about undo (ACTION7) in the six public games that offer
it:

- **State:** undo restores it in all six.
- **Budget refund:** only lf52 refunds the budget. ar25 undo is free.
  bp35 charges a step. In sb26, sk48 and su15, undo does not refund the budget.
- **Score:** undo always counts as a scored action.

### 6.4 Kaggle submission policy (5 per day)

- **Before the first serious submission.** Submit v0 once to confirm it runs
  end to end on hidden games without crashing. Expect about 0.
- **After that.** Submit only when both hold:
  - the simulator dev score beats the last submitted version by more than the
    seed SD;
  - E11 passes.
- **Log every submission:** commit hash, simulator dev/holdout score, LB score,
  runtime and notes.

### 6.5 Research log entry template

Keep one file, for example `RESEARCH_LOG.md`, and append an entry per run:

```
## YYYY-MM-DD  E<id>  <short title>
Hypothesis:      ...
Change:          commit <hash>, files ...
Command:         .venv/bin/python scripts/eval.py --seeds 3 --games dev ...
Result (mean±sd over seeds): ARC x.xx±y.yy | levels a/b | L1 c/18 | ratio r
Per-game deltas: ...
Decision:        adopt / reject / follow-up E<id>
```

---

## 7. Research-paper reasoning

ARC Prize 2026 also runs a separate Paper Track:
kaggle.com/competitions/arc-prize-2026-paper-track. Check its deadline and
format. Prize-eligible code must be open source.

### 7.1 Candidate thesis and contributions

**Thesis.** On ARC-AGI-3, what decides a score is sample-efficient model
learning under an action-efficiency metric. Search power doesn't decide it.

**C1. Failure analysis.** Quantify why uninformed agents score 0:

- the budget-to-baseline gap;
- the illegal-action share;
- per-attempt budgets;
- HUD-induced state aliasing.

§2 is already a draft of this.

**C2. Method.** Combine:

- an object-centric, HUD-invariant state abstraction;
- an online effect model with information-first probing;
- goal-hypothesis testing;
- cross-level transfer.

Ablate each part (E3–E8).

**C3. Evaluation methodology.** A competition-mode simulator, a sealed holdout
of games, seed statistics, and a no-game-id rule. This is reusable by others.

### 7.2 Evidence map

| Claim | Experiments | Artifact |
| --- | --- | --- |
| Random agents fail because of budget and action-space mismatch, not bad luck | §2.1 numbers, E1, E2 | Table: budget vs baseline; random L1 rate vs budget |
| HUD aliasing breaks novelty search | E3; old exact-hash 0/183 result | Plot: distinct states per 100 actions, masked vs raw |
| Probing plus the effect model cuts the cost of discovery | E4, E5, E7 | L1 efficiency-ratio CDF per arm |
| Transfer is where the score comes from | E8; ceiling table §1.4 | Per-level ratio, L1 vs L2+ |
| The method generalizes | E12 | Dev vs holdout bars, with seeds |

### 7.3 Outline

1. Introduction: interactive abstraction; an efficiency-scored benchmark.
2. Problem setup: POMDP with unknown dynamics, per-attempt budgets, the scoring
   formula, competition-mode constraints.
3. Why naive agents fail: C1.
4. Method: perception → effect model → hypotheses → planner → transfer.
5. Experimental protocol: simulator, split, seeds, statistics.
6. Results and ablations.
7. Limitations: hidden state, mechanics outside the hypothesis library,
   holdout leakage, the gap between simulator and gateway.
8. Reproducibility: commands, seeds, commit hashes.

### 7.4 Related work to position against

Verify every bibliographic detail before submitting.

- **Chollet (2019)**, *On the Measure of Intelligence*: skill-acquisition
  efficiency and ARC.
- **Ecoffet et al. (2021, Nature)**, *First return, then explore*: Go-Explore.
- **Count-based and curiosity exploration:**
  - Bellemare et al. (2016), pseudo-counts;
  - Pathak et al. (2017), ICM;
  - Burda et al. (2019), RND;
  - Houthooft et al. (2016), VIME, for information gain.
- **Lipovetzky & Geffner (2012)**: width-based planning (IW) and novelty
  pruning.
- **Diuk, Cohen & Littman (2008)**: object-oriented MDPs.
- **Tsividis et al.:**
  - *Human learning in Atari* (2017);
  - *Theory-based modeling, exploration and planning* (EMPA, 2021). This is the
    closest analogue to this plan.
- **World models:** Hafner et al. (2023), DreamerV3. Also code and program
  world models: Das et al. (2023), AutumnSynth; Tang et al. (2024), WorldCoder.
- **Kocsis & Szepesvári (2006)**: UCT, for the search baselines.
- **ARC-AGI-3 preview-competition write-ups (2025).** These include
  StochasticGoose, which the Kaggle sample is named after, and the
  graph-exploration entries. Collect and read them; they are the most direct
  baselines.

### 7.5 Threats to validity

- The 25 public games may not represent the hidden set.
- Holdout leakage: anyone who reads the sealed cards can no longer treat those
  games as held out.
- Human baselines measure first-time players. Agent "first time" includes
  probing with no instructions.
- The simulator is `arc_agi` 0.9.9's server, not necessarily the gateway's exact
  build. E0 tracks this.

---

## 8. Using Opus 5.5 at High/Max effort

| Workstream | What to give Claude | What to ask | Output |
| --- | --- | --- | --- |
| Planning the agent | This plan, current `my_agent.py`, latest eval JSON | "Pick the next module; design it with failure modes, tests, and a time budget" | Design note, then diff |
| Game mechanics | One **dev** game's source and its probe JSON | "Deobfuscate the step logic; state win/loss exactly; what *general* prior does it suggest?" | Mechanics card (§3 format) |
| Analyzing frames | ASCII before/after frames, the action, the event list | "Which object is controllable, and what changed? Propose 3 probes that tell the hypotheses apart" | Ranked hypotheses and probes |
| Why 0 levels | One game's trace: actions, events, bar level, GAME_OVERs | "Find the first wasted action and the root cause" | Ranked causes, then a fix |
| Designing experiments | Hypothesis plus harness description | "Design the A/B: arms, seeds, metric, stop rule, confounds" | Entry in §6.3 format |
| Paper reasoning | Evidence map plus results | "Which claims are supported? What's missing? Red-team the method" | Claims list, outline edits |

Practical notes:

- Use High/Max effort for diagnosis, design and red-teaming, and default effort
  for mechanical edits.
- Analyze per game in parallel subagents. §3 was produced this way: 5 agents,
  5 games each.
- Never give Claude a holdout game's source or traces before the design freeze.

---

## 9. Timeline

The final submission deadline is **2026-11-02**, per a third-party summary.
Confirm it on Kaggle. Milestone #2 passed on 2026-09-30.

| Week | Dates | Deliverables |
| --- | --- | --- |
| 1 | Oct 3–9 | Simulator and eval harness (M0); v0 safety shell plus one sanity submission; v1 perception and probe explorer; E1–E5 |
| 2 | Oct 10–16 | v2 effect model and navigation; E6, E7, E9, E10 |
| 3 | Oct 17–23 | v3 goals and transfer (E8); design freeze; holdout run (E12); serious Kaggle submissions |
| 4 | Oct 24–Nov 2 | Hardening (E11); final submissions by Oct 31 with 2 days of slack; paper draft |

Open questions to resolve early:

1. What is the Kaggle runtime limit for the rerun? Check the Kaggle overview's
   code requirements and the discussion forum.
2. Does the gateway cap actions per game?
3. How many hidden games are there?

---

## Appendix A — Competition-mode simulator (tested 2026-10-02)

Two processes. Start a new server for every run, because competition mode
allows only one scorecard per server.

`scripts/kaggle_sim_server.py`

```python
"""Local stand-in for the Kaggle gateway: competition-mode REST server over cached games."""
import logging
import sys

import arc_agi
from arc_agi import OperationMode

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
logging.basicConfig(level=logging.WARNING)
arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
arc.listen_and_serve(host="127.0.0.1", port=port, competition_mode=True)
```

`scripts/kaggle_sim.py`

```python
"""Run agent/my_agent.py through the real framework Swarm against the local gateway,
mirroring the Kaggle rerun (HTTP, online mode, one thread per game)."""
import importlib.util
import logging
import os
import sys
import time

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
os.environ.update({"OPERATION_MODE": "online", "ARC_BASE_URL": f"http://127.0.0.1:{port}/",
                   "ARC_API_KEY": "test-key-123", "ENVIRONMENTS_DIR": "",
                   # Swarm records every frame; without this it writes into the repo root.
                   "RECORDINGS_DIR": "recordings"})
sys.path.insert(0, "vendor/ARC-AGI-3-Agents")
logging.basicConfig(level=logging.ERROR)

import requests  # noqa: E402

import agents  # noqa: E402
from agents import Swarm  # noqa: E402

spec = importlib.util.spec_from_file_location("user_agent", "agent/my_agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
agents.AVAILABLE_AGENTS["myagent"] = mod.MyAgent

games = [g["game_id"] for g in requests.get(f"http://127.0.0.1:{port}/api/games",
                                            headers={"X-API-Key": "test-key-123"}, timeout=10).json()]
if len(sys.argv) > 2:
    games = [g for g in games if any(g.startswith(k) for k in sys.argv[2].split(","))]

t0 = time.time()
swarm = Swarm("myagent", f"http://127.0.0.1:{port}", games)
card = swarm.main()
for a in sorted(swarm.agents, key=lambda a: a.game_id):
    f = a.frames[-1]
    print(f"{a.game_id[:4]:6} levels {f.levels_completed}/{f.win_levels}  "
          f"actions {a.action_counter}  final {f.state.name}")
if card is not None:
    print("ARC SCORE", card.model_dump().get("score"))
print(f"elapsed {time.time() - t0:.1f}s")
```

Usage, from the repo root:

```sh
.venv/bin/python scripts/kaggle_sim_server.py 8765 &
.venv/bin/python scripts/kaggle_sim.py 8765            # all games
.venv/bin/python scripts/kaggle_sim.py 8765 ls20,vc33  # subset (needs a fresh server)
```

Starter results:

- **Run 1:** 25/25 games at 0 levels, 81 actions each, ARC score 0.0, 16 s.
- **Run 2:** sp80 cleared level 1, ARC score 0.096, 17 s.

The starter seeds from the clock, so results vary from run to run.

---

## Appendix B — Per-game measurements (offline engine, 2026-10-02)

Column key:

- **GAME_OVER after (probe):** how many actions passed before GAME_OVER while
  one action was repeated from the level start. The probe action is named in
  brackets; a click probe always clicked (0,0). "none" means no GAME_OVER
  within 400 actions, which happens when that action costs no budget (§3.2
  gives the real budgets).
- **Random L1:** how many of 3 runs (3,000 actions each) cleared level 1.
  "Legal-only" picks uniformly among `available_actions`. "Object clicks" is
  the same, except clicks land on a random non-background pixel.
- **Viewport:** sizes can change on later levels.
- **(H):** sealed holdout game. Only the stats the agent would observe anyway
  are shown.

| Game | Levels | L1 human | Legal | Viewport → scale | GAME_OVER after (probe) | Random L1, legal-only | Random L1, object clicks |
| --- | ---: | ---: | --- | --- | --- | :---: | :---: |
| ar25 | 8 | 32 | 1–7 | 21² → ×3 | none (blocked moves are free) | 0/3 | 0/3 |
| bp35 | 9 | 21 | 3,4,6,7 | 8² dummy; native 64² render, 6-px cells, scrolls | 16 (A3), 24 (A4): flood; 64 (click) | 0/3 | 0/3 |
| cd82 | 6 | 55 | 1–6 | 64² → ×1 | 100 | 1/3 | 1/3 |
| cn04 | 6 | 29 | 1–6 | 20² → ×3 | 75 | 0/3 | 0/3 |
| dc22 (H) | 6 | 59 | 1–4,6 | 64×44 → ×1 | 128 | 0/3 | 0/3 |
| ft09 | 6 | 43 | 6 | 32² → ×2 | none (missed clicks are free) | 2/3 | 2/3 |
| g50t (H) | 7 | 78 | 1–5 | 64² → ×1 | 130 | 0/3 | 0/3 |
| ka59 | 7 | 28 | 1–4,6 | 45² → ×1, letterboxed | 100 | 0/3 | 0/3 |
| lf52 | 10 | 32 | 1–4,6,7 | 8² dummy; native 64² render, 6-px cells | 64 | 0/3 | 0/3 |
| lp85 (H) | 8 | 17 | 6 | 32×19 → ×2 | none (missed clicks are free) | 2/3 | 3/3 |
| ls20 | 7 | 22 | 1–4 | 64² → ×1 | 129 (A2); none (A1: bumping the top wall costs nothing) | 0/3 | 0/3 |
| m0r0 | 6 | 30 | 1–6 | 11² → ×5 | 151 | 2/3 | 2/3 |
| r11l | 6 | 22 | 6 | 64² → ×1 | 60 | 3/3 | 3/3 |
| re86 | 8 | 26 | 1–5 | 64² → ×1 | 100 | 0/3 | 0/3 |
| s5i5 (H) | 8 | 20 | 6 | 64² → ×1 | 50 | 0/3 | 0/3 |
| sb26 | 8 | 18 | 5,6,7 | 64² → ×1 | 64 (A5); none (A7, click) | 0/3 | 0/3 |
| sc25 (H) | 6 | 36 | 1–4,6 | 64² → ×1 | 52 (moves); none (click) | 0/3 | 0/3 |
| sk48 (H) | 8 | 61 | 1–4,6,7 | 64² → ×1 | 196 | 0/3 | 0/3 |
| sp80 | 6 | 39 | 1–6 | 16² → ×4 | 30 | 3/3 | 3/3 |
| su15 | 9 | 22 | 6,7 | 64² → ×1 | none (panel clicks are free) | 0/3 | 0/3 |
| tn36 | 7 | 32 | 6 | 64² → ×1 | 61 | 0/3 | 0/3 |
| tr87 (H) | 6 | 54 | 1–4 | 64² → ×1 | 128 | 0/3 | 0/3 |
| tu93 | 9 | 19 | 1–4 | 39² → ×1, letterboxed | 50 | 0/3 | 0/3 |
| vc33 | 7 | 7 | 6 | 32² → ×2 | 50 | 0/3 | 2/3 |
| wa30 | 9 | 71 | 1–5 | 64² → ×1 | 200 | 0/3 | 0/3 |

Totals:

- **Levels:** 183.
- **Human baseline actions:** 17,135 over all levels, a mean of 685 per game.
- **Level-1 baselines:** median 30, mean 34.9, range 7–78.

How the random runs went:

- **Legal-only random:** cleared level 1 in 6 games. The median first clear
  came at action 884, 35× the human count.
- **With object clicks:** 7 games, median at action 1,170 (56×).
- **The starter policy given 3,000 actions:** 5 games, median at action 1,780
  (48×).
- **Depth:** no random run of any kind cleared more than 2 levels.

---

## Appendix C — Housekeeping

- **Stale scripts.** `scripts/collect_scores.py`, `scripts/replay_json.py` and
  `scripts/test_agent_state.py` set attributes of the deleted agents
  (`OPENING_BOOK`, `USE_ENGINE_PLANNER`, `EXACT_STATE_HASH`, …). Replace them
  with `scripts/eval.py` (§6.1). `scripts/visualize_games.py` is generic and
  still useful.
- **`.gitignore`.** Add `local_scores_*.json`, `submission.parquet` and any
  frame-dump output directory.
- **`notebooks/kernel-metadata.json`.** Still contains
  `REPLACE_WITH_YOUR_USERNAME`, so `make submit` will refuse to push.

---

## Sealed holdout: mechanics of dc22, g50t, lp85, s5i5, sc25, sk48, tr87

<details>
<summary>Don't open before the design freeze (M3). Reading these turns the holdout into dev data.</summary>

| Game | Legal | Inputs | Win | Budget (L1) and other losses | L1 solve (human) |
| --- | --- | --- | --- | --- | --- |
| dc22 | 1–4,6 | A1–4 move a 2×2 avatar 2 px. A6 on control-panel buttons cycles hidden, color-coded groups of tiles (walkable or blocking, pivoting bridges). Later levels add keys, carrying platforms, a crane and pressure plates. | The avatar reaches the goal cell | 128; every action costs, including unavailable ones and off-target clicks; a button costs 2; removing your own floor is a fall (−20 steps) | — (59) |
| g50t | 1–5 | A1–4 move one 6-px cell. A5 rewinds the avatar to the start, and the recorded path becomes a ghost that replays alongside your next moves, so it can hold plates and open doors. | The avatar is on the goal pad | 130 actions of any kind; crushing doors; a patrolling enemy on L6–7 | — (78) |
| lp85 | 6 | A6 on an arrow button rotates every tile on that ring one step forward or backward | Every bracket holds a goal tile of its color | 13 button presses; missed clicks are free | 5 (17) |
| s5i5 | 6 | A6 on a slider's far or near half lengthens or shortens every arm segment of that color. Rotate buttons appear on L6+. | Every target ring has a hand on it | 50 clicks; misses cost too | 13 (20) |
| sc25 | 1–4,6 | A1–4 move a wizard. A6 toggles cells of a 3×3 rune grid; a matching pattern casts a spell (plus = resize, L = teleport, line = fireball). Spellbook icons show hints. | The wizard bumps into the exit | 50 (+1 for the tutorial); moves, toggles and casts each cost 1 | — (36) |
| sk48 | 1–4,6,7 | A1–4 extend, retract or slide a telescoping arm, which pushes or threads blocks. A6 switches heads (L6+). A7 undo (no refund). | Each arm's block sequence matches the reference strip | 196 moves | 14 by BFS (61) |
| tr87 | 1–4 | A3/A4 move a cursor; A1/A2 cycle the selected glyph (levels 5–6 edit the rules instead) | The answer row equals the source row rewritten by the rule legend | 128 presses | 15 (54) |

What these mean for the priors:

- **Already covered** by the dev families: reach (dc22, g50t, sc25), overlap/fit
  (lp85, s5i5), copy a reference (sk48), rule translation (tr87).
- **New patterns**, worth noting only after the freeze:
  - **Time-loop clones (g50t):** cooperate with your own recorded replay.
  - **Spell grammar (sc25):** a click pattern cast as a command.
  - **Ring permutations (lp85).**

</details>
