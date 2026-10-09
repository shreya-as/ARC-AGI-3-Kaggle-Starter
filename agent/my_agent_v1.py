"""v1 agent: perception + probe explorer (AGENT_PLAN.md §4.2, §5.3-B/D, §5.5 v1).

Built on the v0 safety shell (legal actions only, crash-proof stepping, stable
seeding, bounded memory, wall-clock limit), which is unchanged below. New in v1:

  Perception (numpy only, no scipy)
    - HUD mask: budget bars drain 1-2 px per action along an edge line. A small
      thin diff blob near the frame edge is a "tick"; two ticks on the same row
      or column, a few px apart, confirm a bar line, and the bar's whole run on
      that line is masked. Masked pixels never count as change or as state.
    - Objects: 4-connected same-colour components over the non-background,
      non-HUD pixels. Class = (colour, shape, surrounding colour), so identical
      sprites in the same context share stats.
    - State key: hash of (level, HUD-masked grid). Blind to the budget bar, so
      revisiting a position after a reset or an undo is recognised.

  Explorer
    - Every action is an "action key": a simple action id, or a click on one
      specific object (its pixel coordinates are part of the key, so a click
      replays exactly).
    - Each state remembers which keys were tried there and where they led.
    - Untried keys are ranked by a prior learned across all states and levels:
      how often that simple action, or that object class, changed the world.
      Unknown actions start optimistic, so every legal action and every object
      class is probed early; a class that did nothing once drops to a low prior.
    - When the current state has nothing promising left, BFS over the learned
      transition graph to the nearest state that does, and walk there.
    - Otherwise prefer moves into the least-visited known states (novelty).

Knobs (environment variables, read once at import):
  AGENT_MAX_ACTIONS   actions per game              (default 2000)
  AGENT_WALL_SECONDS  wall-clock limit per game     (default 1800)
  SEED_OFFSET         added to every per-game seed  (default 0)

To submit it, copy this file over `agent/my_agent.py` (the notebook builder only
splices that file). The class is named `MyAgent` so it registers unchanged.
"""
from __future__ import annotations

import logging
import os
import random
import time
import zlib
from collections import deque
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

logger = logging.getLogger()

MAX_ACTIONS_DEFAULT = int(os.getenv("AGENT_MAX_ACTIONS", "2000"))
WALL_SECONDS_DEFAULT = float(os.getenv("AGENT_WALL_SECONDS", "1800"))
SEED_OFFSET = int(os.getenv("SEED_OFFSET", "0"))

FRAMES_KEPT = 8          # frames held in self.frames
MAX_FAIL_STREAK = 3      # consecutive failed steps before forcing a RESET
DECISION_SAMPLES = 4096  # decision-time samples kept for the p95 report

SIZE = 64
BAND = 3                 # ticks must touch the outer BAND px of the frame
TICK_MAX_AREA = 4        # a bar tick changes at most this many pixels
TICK_MAX_GAP = 4         # max distance between two ticks that confirm a bar line
MAX_OBJECTS = 300        # components considered per state (largest first)
MAX_CLICKS = 128         # click candidates kept per state
BIG_OBJECT = 600         # px; walls, panels and letterbox are rarely buttons
LOCAL_MIN_PRIOR = 0.15   # below this, walk to a better frontier state instead
BFS_MAX_NODES = 4000


# ── Perception helpers ──────────────────────────────────────────────────────

def label_components(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """4-connected components of equal `values` among `valid` pixels.

    Returns an int array where each valid pixel holds the flat index of one
    pixel of its component (a component id), and invalid pixels hold H*W.
    Min-label propagation with pointer jumping; converges in a few dozen
    vectorised passes even for long snaking walls.
    """
    h, w = values.shape
    big = h * w
    lab = np.where(valid, np.arange(big).reshape(h, w), big)
    eq_r = valid[:, :-1] & valid[:, 1:] & (values[:, :-1] == values[:, 1:])
    eq_d = valid[:-1, :] & valid[1:, :] & (values[:-1, :] == values[1:, :])
    while True:
        new = lab.copy()
        np.minimum(new[:, :-1], np.where(eq_r, lab[:, 1:], big), out=new[:, :-1])
        np.minimum(new[:, 1:], np.where(eq_r, lab[:, :-1], big), out=new[:, 1:])
        np.minimum(new[:-1, :], np.where(eq_d, lab[1:, :], big), out=new[:-1, :])
        np.minimum(new[1:, :], np.where(eq_d, lab[:-1, :], big), out=new[1:, :])
        flat = new.ravel()
        inside = flat < big
        for _ in range(4):
            flat[inside] = flat[flat[inside]]
        if np.array_equal(new, lab):
            return new
        lab = new


def components(lab: np.ndarray) -> list[np.ndarray]:
    """Flat pixel indices of each component in a label array, largest first."""
    flat = lab.ravel()
    pix = np.nonzero(flat < flat.size)[0]
    if pix.size == 0:
        return []
    labs = flat[pix]
    order = np.argsort(labs, kind="stable")
    pix, labs = pix[order], labs[order]
    _, starts, counts = np.unique(labs, return_index=True, return_counts=True)
    comps = [pix[s:s + c] for s, c in zip(starts, counts)]
    comps.sort(key=len, reverse=True)
    return comps


class Perception:
    """HUD mask, background, objects and state keys for one game."""

    def __init__(self) -> None:
        self.hud = np.zeros((SIZE, SIZE), dtype=bool)
        self.band = np.zeros((SIZE, SIZE), dtype=bool)
        self.band[:BAND, :] = self.band[-BAND:, :] = True
        self.band[:, :BAND] = self.band[:, -BAND:] = True
        # candidate bar lines: ('r', row) / ('c', col) -> [(step, pos), ...]
        self.ticks: dict[tuple[str, int], list[tuple[int, int]]] = {}
        self.bar_lines: set[tuple[str, int]] = set()

    def observe_step(self, prev: np.ndarray, cur: np.ndarray, step: int) -> bool:
        """Learn the HUD from one in-level transition; return True if the world changed."""
        diff = prev != cur
        if not diff.any():
            return False
        world = diff & ~self.hud
        if not world.any():
            return False
        if world.sum() > 1500:  # whole-screen change: not a tick, clearly effective
            return True
        changed = False
        for comp in components(label_components(world.astype(np.int8), world)):
            ys, xs = comp // SIZE, comp % SIZE
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            on_bar = any(("r", int(y)) in self.bar_lines for y in ys) or any(
                ("c", int(x)) in self.bar_lines for x in xs)
            is_tick = comp.size <= TICK_MAX_AREA and min(h, w) <= 2
            if is_tick and (on_bar or self.band[ys, xs].any()):
                self.hud[ys, xs] = True
                for y, x in zip(ys.tolist(), xs.tolist()):
                    self._add_tick(("r", y), x, step, prev, cur)
                    self._add_tick(("c", x), y, step, prev, cur)
            elif on_bar and min(h, w) <= 2:
                self.hud[ys, xs] = True  # bar refill or multi-px jump
            else:
                changed = True
        return changed

    def _add_tick(self, line: tuple[str, int], pos: int, step: int,
                  prev: np.ndarray, cur: np.ndarray) -> None:
        seen = self.ticks.setdefault(line, [])
        confirm = line in self.bar_lines or any(
            s != step and abs(p - pos) <= TICK_MAX_GAP for s, p in seen)
        seen.append((step, pos))
        if len(seen) > 16:
            del seen[:-16]
        if confirm:
            self.bar_lines.add(line)
            self._mask_bar_run(line, pos, prev, cur)

    def _mask_bar_run(self, line: tuple[str, int], pos: int,
                      prev: np.ndarray, cur: np.ndarray) -> None:
        """Mask the contiguous run of bar colours along `line` through `pos`."""
        kind, idx = line
        before = prev[idx, :] if kind == "r" else prev[:, idx]
        after = cur[idx, :] if kind == "r" else cur[:, idx]
        colours = {int(before[pos]), int(after[pos])}
        lo = pos
        while lo > 0 and int(after[lo - 1]) in colours:
            lo -= 1
        hi = pos
        while hi < SIZE - 1 and int(after[hi + 1]) in colours:
            hi += 1
        if kind == "r":
            self.hud[idx, lo:hi + 1] = True
        else:
            self.hud[lo:hi + 1, idx] = True

    def key(self, grid: np.ndarray, level: int) -> int:
        masked = np.where(self.hud, -1, grid).astype(np.int8)
        return hash((level, masked.tobytes()))

    def background(self, grid: np.ndarray) -> int:
        vals = grid[~self.hud]
        if vals.size == 0:
            vals = grid.ravel()
        return int(np.bincount(vals.astype(np.int64) & 15, minlength=16).argmax())

    def click_targets(self, grid: np.ndarray, rng: random.Random) -> list[tuple]:
        """One click per object: (x, y, class_key, area, n_same_class)."""
        bg = self.background(grid)
        valid = (grid != bg) & ~self.hud
        comps = components(label_components(grid, valid))[:MAX_OBJECTS]
        targets = []
        class_count: dict[tuple, int] = {}
        for comp in comps:
            ys, xs = comp // SIZE, comp % SIZE
            y0, x0 = int(ys.min()), int(xs.min())
            h, w = int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1
            shape = np.zeros((h, w), dtype=bool)
            shape[ys - y0, xs - x0] = True
            colour = int(grid[ys[0], xs[0]])
            # The surrounding colour separates look-alikes in different
            # contexts, e.g. a clickable tile inside a panel vs. a decorative
            # copy of it on the background.
            ring = grid[max(y0 - 1, 0):y0 + h + 1, max(x0 - 1, 0):x0 + w + 1].ravel()
            ring = ring[ring != colour]
            surround = int(np.bincount(ring.astype(np.int64) & 15).argmax()) if ring.size else -1
            cls = ("obj", colour, h, w, hash(shape.tobytes()), surround)
            # click the component pixel closest to its centroid
            i = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
            targets.append([int(xs[i]), int(ys[i]), cls, int(comp.size)])
            class_count[cls] = class_count.get(cls, 0) + 1
        out = [(x, y, cls, area, class_count[cls]) for x, y, cls, area in targets]
        # one background click per state, for select-then-place games
        bg_pix = np.argwhere((grid == bg) & ~self.hud)
        if len(bg_pix):
            y, x = bg_pix[rng.randrange(len(bg_pix))]
            out.append((int(x), int(y), ("bg", bg), SIZE * SIZE, 1))
        return out


# ── Agent ───────────────────────────────────────────────────────────────────

class MyAgent(Agent):
    """Probe-first, graph-novelty explorer inside the v0 safety shell."""

    MAX_ACTIONS = MAX_ACTIONS_DEFAULT
    WALL_SECONDS = WALL_SECONDS_DEFAULT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = zlib.crc32(f"{self.game_id}:{SEED_OFFSET}".encode())
        self.rng = random.Random(seed)
        self._click: Optional[dict[str, int]] = None  # data for the pending ACTION6
        self._fail_streak = 0
        self._gave_up = False
        self._failures = 0
        self._decision_times: list[float] = []
        self._deadline = float("inf")  # set when main() starts
        self._raw_grid: Optional[np.ndarray] = None

        self.perc = Perception()
        # transition graph: state -> {action_key: next_state | DEAD}
        self.edges: dict[int, dict[tuple, Any]] = {}
        self.visits: dict[int, int] = {}
        self.deaths: dict[tuple[int, tuple], int] = {}
        self.candidates: dict[int, list[tuple]] = {}  # state -> [(action_key, class_key|None)]
        # priors learned across states and levels: key -> [tries, effective]
        self.simple_stats: dict[int, list[int]] = {}
        self.class_stats: dict[tuple, list[int]] = {}
        self.class_meta: dict[tuple, tuple[int, int]] = {}  # class -> (area, count)
        self.path: list[tuple[tuple, int]] = []  # [(action_key, expected_next_state)]
        self.expect: Optional[int] = None          # state the last path step should reach
        # previous step, for learning from the transition it produced
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_key: Optional[int] = None
        self.prev_level = 0
        self.prev_action: Optional[tuple] = None
        self.prev_class: Optional[tuple] = None
        self.step_no = 0
        self.level_ups: list[int] = []
        self.effective_steps = 0

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    # ── Main loop (v0 shell) ─────────────────────────────────────────────────
    # Same shape as Agent.main, but nothing in it can raise: the base loop
    # converts `observation_space` before calling choose_action, and a None
    # there (failed reset) raises ValueError and kills the thread.

    def main(self) -> None:
        self.timer = time.time()
        self._deadline = self.timer + self.WALL_SECONDS
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            latest = self._latest_frame()
            t0 = time.perf_counter()
            try:
                action = self.choose_action(self.frames, latest)
            except Exception:
                logger.exception(f"{self.game_id} - choose_action failed")
                self._forget_prev()
                action = self._fallback_action(latest)
            if len(self._decision_times) < DECISION_SAMPLES:
                self._decision_times.append(time.perf_counter() - t0)

            frame = self.take_action(action)
            if frame is not None:
                self._fail_streak = 0
                self.append_frame(frame)
            else:
                self._failures += 1
                self._fail_streak += 1
                self._forget_prev()
                if action is GameAction.RESET and self._fail_streak > MAX_FAIL_STREAK:
                    logger.error(f"{self.game_id} - RESET failed after {self._fail_streak - 1} failed steps; stopping")
                    self._gave_up = True
            self.action_counter += 1

        self.cleanup()

    def _latest_frame(self) -> FrameData:
        """Current observation; falls back to the last good frame if unavailable."""
        try:
            raw = self.arc_env.observation_space if self.arc_env else None
            frame = self._convert_raw_frame_data(raw)
            self._raw_grid = np.asarray(raw.frame[-1], dtype=np.int8) if raw.frame else None
            return frame
        except Exception:
            self._raw_grid = None
            return self.frames[-1]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stopping early never earns points (§1.4), so stop only on WIN, when
        # the shell has given up, or when the wall clock runs out.
        return (
            latest_frame.state is GameState.WIN
            or self._gave_up
            or time.time() >= self._deadline
        )

    # ── Policy ───────────────────────────────────────────────────────────────

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._learn(latest_frame, None)
            self._forget_prev()
            return GameAction.RESET
        if self._fail_streak >= MAX_FAIL_STREAK:
            self._forget_prev()
            return GameAction.RESET

        grid = self._raw_grid
        if grid is None or grid.shape != (SIZE, SIZE):
            if not latest_frame.frame:
                self._forget_prev()
                return self._random_legal_action(latest_frame)
            grid = np.asarray(latest_frame.frame[-1], dtype=np.int8)
            if grid.shape != (SIZE, SIZE):
                self._forget_prev()
                return self._random_legal_action(latest_frame)

        key = self._learn(latest_frame, grid)
        action_key, cls = self._pick(key, grid, latest_frame)
        self.prev_grid, self.prev_key = grid, key
        self.prev_action, self.prev_class = action_key, cls
        return self._to_game_action(action_key, latest_frame)

    def _forget_prev(self) -> None:
        self.prev_grid = self.prev_key = self.prev_action = self.prev_class = None
        self.path = []

    def _learn(self, latest: FrameData, grid: Optional[np.ndarray]) -> Optional[int]:
        """Fold the last transition into the HUD mask, stats and graph; return the state key."""
        self.step_no += 1
        level = latest.levels_completed
        if level > self.prev_level and self.prev_action is not None:
            self.level_ups.append(self.action_counter)
            logger.info(f"{self.game_id} - level {level} reached at action {self.action_counter}")
        same_level = level == self.prev_level and self.prev_grid is not None
        effective = False
        if grid is not None and same_level:
            effective = self.perc.observe_step(self.prev_grid, grid, self.step_no)
        key = self.perc.key(grid, level) if grid is not None else None

        if self.prev_action is not None and self.prev_key is not None:
            leveled = level > self.prev_level
            dead = latest.state is GameState.GAME_OVER
            effective = effective or leveled
            if not dead:  # deaths are mostly the budget, not the last action's fault
                self._count(self.prev_action, self.prev_class, effective)
            if effective:
                self.effective_steps += 1
            outcome = "DEAD" if dead else (key if not leveled else "UP")
            if not effective and not dead:
                outcome = self.prev_key  # no-op: stays in place
            if dead:
                # Most deaths are the attempt budget running out, so keep the
                # move open; only a move that kills twice is recorded as deadly.
                n = self.deaths.get((self.prev_key, self.prev_action), 0) + 1
                self.deaths[(self.prev_key, self.prev_action)] = n
                if n < 2:
                    outcome = None
            if outcome is not None:
                self.edges.setdefault(self.prev_key, {})[self.prev_action] = outcome
        if key is not None:
            self.visits[key] = self.visits.get(key, 0) + 1
        self.prev_level = level
        return key

    def _count(self, action_key: tuple, cls: Optional[tuple], effective: bool) -> None:
        stats = (self.class_stats.setdefault(cls, [0, 0]) if cls is not None
                 else self.simple_stats.setdefault(action_key[1], [0, 0]))
        stats[0] += 1
        stats[1] += int(effective)

    def _prior(self, action_key: tuple, cls: Optional[tuple]) -> float:
        if cls is None:
            tries, eff = self.simple_stats.get(action_key[1], (0, 0))
            if action_key[1] == 7:
                # Undo only returns to a known state: probe it once, then keep
                # it as a leftover rather than trying it in every new state.
                return 0.9 if tries == 0 else 0.1
            if tries == 0:
                return 1.0
            return (eff + 0.2) / (tries + 1)
        tries, eff = self.class_stats.get(cls, (0, 0))
        if tries == 0:
            area, count = self.class_meta.get(cls, (0, 1))
            p = 0.95 - 0.01 * min(count, 20)  # rarest classes first
            if cls[0] == "bg":
                p = 0.4
            elif area >= BIG_OBJECT:
                p = 0.3
            return p
        return (eff + 0.2) / (tries + 1)

    def _state_candidates(self, key: int, grid: np.ndarray, latest: FrameData) -> list[tuple]:
        cands = self.candidates.get(key)
        if cands is None:
            legal = self._legal_ids(latest)
            cands = [(("s", a), None) for a in legal if a != 6]
            if 6 in legal:
                targets = self.perc.click_targets(grid, self.rng)
                for x, y, cls, area, count in targets:
                    self.class_meta.setdefault(cls, (area, count))
                targets.sort(key=lambda t: -self._prior(("c",), t[2]))
                cands += [(("c", x, y, cls), cls) for x, y, cls, _, _ in targets[:MAX_CLICKS]]
            self.candidates[key] = cands
        return cands

    def _best_untried(self, key: int) -> tuple[float, Optional[tuple]]:
        tried = self.edges.get(key, {})
        best, best_c = -1.0, None
        for ak, cls in self.candidates.get(key, ()):
            if ak in tried:
                continue
            p = self._prior(ak, cls)
            if p > best:
                best, best_c = p, (ak, cls)
        return best, best_c

    def _pick(self, key: int, grid: np.ndarray, latest: FrameData) -> tuple[tuple, Optional[tuple]]:
        cands = self._state_candidates(key, grid, latest)
        cls_of = dict(cands)

        # 1. keep walking a planned path while the world behaves as predicted
        if self.path and key == self.expect and self.path[0][0] in cls_of:
            ak, self.expect = self.path.pop(0)
            return ak, cls_of[ak]
        self.path = []

        # 2. best untried action here
        best, choice = self._best_untried(key)
        if choice is not None and best >= LOCAL_MIN_PRIOR:
            ties = [(ak, c) for ak, c in cands
                    if ak not in self.edges.get(key, {}) and abs(self._prior(ak, c) - best) < 1e-9]
            return self.rng.choice(ties)

        # 3. walk to the nearest state that still has a promising untried action
        route = self._route_to_frontier(key)
        if route and route[0][0] in cls_of:
            (ak, self.expect), self.path = route[0], route[1:]
            return ak, cls_of[ak]

        # 4. leftovers: anything untried, else the move into the least-visited state
        if choice is not None:
            return choice
        moves = [(ak, nxt) for ak, nxt in self.edges.get(key, {}).items()
                 if ak in cls_of and nxt not in ("DEAD", key)]
        if moves:
            low = min(self.visits.get(n, 0) if isinstance(n, int) else 0 for _, n in moves)
            ak = self.rng.choice([a for a, n in moves
                                  if (self.visits.get(n, 0) if isinstance(n, int) else 0) == low])
            return ak, cls_of[ak]
        ak, cls = self.rng.choice(cands) if cands else (("s", 0), None)
        return ak, cls

    def _route_to_frontier(self, start: int) -> list[tuple[tuple, int]]:
        """BFS over known effective edges to the nearest state with a promising untried action."""
        parent: dict[int, tuple[int, tuple]] = {start: (start, ("s", 0))}
        queue = deque([start])
        while queue and len(parent) < BFS_MAX_NODES:
            s = queue.popleft()
            if s != start and self._best_untried(s)[0] >= LOCAL_MIN_PRIOR:
                route = []
                while s != start:
                    p, ak = parent[s]
                    route.append((ak, s))
                    s = p
                return route[::-1]
            for ak, nxt in self.edges.get(s, {}).items():
                if isinstance(nxt, int) and nxt != s and nxt not in parent:
                    parent[nxt] = (s, ak)
                    queue.append(nxt)
        return []

    # ── Actions ──────────────────────────────────────────────────────────────

    def _legal_ids(self, latest_frame: FrameData) -> list[int]:
        ids = []
        for action_id in latest_frame.available_actions or []:
            try:
                a = int(action_id)
            except (TypeError, ValueError):
                continue
            if 1 <= a <= 7:
                ids.append(a)
        return ids

    def _to_game_action(self, action_key: tuple, latest_frame: FrameData) -> GameAction:
        legal = self._legal_ids(latest_frame)
        if action_key[0] == "c" and 6 in legal:
            self._click = {"x": action_key[1], "y": action_key[2]}
            return GameAction.ACTION6
        if action_key[0] == "s" and action_key[1] in legal and action_key[1] != 6:
            return GameAction.from_id(action_key[1])
        self._forget_prev()
        return self._random_legal_action(latest_frame)

    def _random_legal_action(self, latest_frame: FrameData) -> GameAction:
        legal = self._legal_ids(latest_frame)
        if not legal:
            # No legal list to sample from; RESET is the only input that can't
            # be illegal while the game is running.
            return GameAction.RESET
        action = GameAction.from_id(self.rng.choice(legal))
        if action.is_complex():
            self._click = {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)}
        return action

    def _fallback_action(self, latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER, GameState.WIN):
            return GameAction.RESET
        try:
            return self._random_legal_action(latest_frame)
        except Exception:
            return GameAction.RESET

    # ── Stepping (v0 shell) ──────────────────────────────────────────────────

    def do_action_request(self, action: GameAction) -> FrameData:
        # Send click coordinates from this agent, not from the shared
        # GameAction.ACTION6 singleton that every game thread writes to.
        if action.is_complex():
            data = dict(self._click or {"x": 0, "y": 0})
            data["game_id"] = self.game_id
        else:
            data = {"game_id": self.game_id}
        raw = self.arc_env.step(action, data=data, reasoning=None)
        return self._convert_raw_frame_data(raw)

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        try:
            return super().take_action(action)
        except Exception as e:
            logger.warning(f"{self.game_id} - {action.name} failed: {e}")
            return None

    # ── Bounded memory (v0 shell) ────────────────────────────────────────────

    def start_recording(self) -> None:
        # The framework would write every frame as JSON (GBs at thousands of
        # actions). The gateway records actions server-side, so skip it.
        pass

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if len(self.frames) > FRAMES_KEPT:
            del self.frames[: len(self.frames) - FRAMES_KEPT]
        if frame.guid:
            self.guid = frame.guid

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup and self._decision_times:
            times = sorted(self._decision_times)
            p95 = times[int(0.95 * (len(times) - 1))] * 1000
            logger.info(
                f"{self.game_id} - actions {self.action_counter}, levels {self.levels_completed} "
                f"(at {self.level_ups}), states {len(self.visits)}, effective {self.effective_steps}, "
                f"hud px {int(self.perc.hud.sum())}, bar lines {sorted(self.perc.bar_lines)}, "
                f"failed steps {self._failures}, gave up {self._gave_up}, "
                f"p95 decision {p95:.2f} ms, {self.seconds}s"
            )
        super().cleanup(*args, **kwargs)
