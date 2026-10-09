"""v3 agent: goal hypotheses + level transfer (AGENT_PLAN.md §5.3-E/F/G, §5.5 v3).

Built on v2 (v0 safety shell, v1 HUD-masked perception and probe/graph
explorer, v2 move model / wall learning / death causes), unchanged except
where noted. New in v3:

  Level memory and transfer (§5.3-G)
    - On every level-up, the action that triggered it is remembered:
        * a move: the colours the avatar moved onto ("goal colours");
        * a click: the clicked object class and its colour;
        * any other action (ACTION5, ...): the action id.
      On later levels those are tried first: clicks on the winning class or
      colour and the winning action get a prior above every unknown action.
    - The previous level's move mapping is kept as a provisional model, so
      navigation can start on the first step of a new level; one move that
      disagrees with it replaces it (screens can rotate between levels, R8).

  Goal hypotheses and a navigation planner (§5.3-E/F)
    - "Reach" hypothesis: if a goal colour is known, BFS over avatar
      positions using the learned move model, through colours known to be
      passable, to the nearest box overlapping a goal colour; then walk the
      plan, replanning whenever a move doesn't go as predicted.
    - Before the first level-up no goal is known, so the hypothesis is "go
      touch what you haven't touched": the same planner targets small objects
      whose colour the avatar never stood on or bumped into (R3: the goal is
      drawn on screen). Each target is visited once per level.
    - Everything else falls through to the v2 explorer.

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
HIGH_PRIOR = 0.5         # walk a few steps for a move this good before settling
HIGH_ROUTE_MAX = 12      # max path length when walking to a high-value move
BFS_MAX_NODES = 4000
MOVES = (1, 2, 3, 4)     # actions that may move a controllable object
MOVE_MARGIN = 8          # px around the changed region searched for moved objects
CTRL_MIN_VOTES = 3       # moves seen before an object counts as controllable
HAZARD_MAX_USED = 0.6    # dying with less than this share of the bar used = hazard
PLAN_MAX_NODES = 4000    # avatar positions expanded per navigation plan
GOAL_MAX_SHARE = 0.25    # a colour covering more of the screen is floor, not a goal
TOUCH_MAX_PX = 400       # only objects up to this size are "go touch it" targets
WIN_PRIOR = 1.2          # prior of the action class that won the previous level


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


def shape_key(grid: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple[tuple, int, int]:
    """(colour, h, w, shape hash) of one component, plus its top-left corner."""
    y0, x0 = int(ys.min()), int(xs.min())
    h, w = int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1
    shape = np.zeros((h, w), dtype=bool)
    shape[ys - y0, xs - x0] = True
    return (int(grid[ys[0], xs[0]]), h, w, hash(shape.tobytes())), y0, x0


def objects_in(grid: np.ndarray, valid: np.ndarray,
               box: tuple[int, int, int, int]) -> dict[tuple, list[tuple[int, int]]]:
    """Components lying wholly inside `box` = (y0, y1, x0, x1): shape key -> [top-left]."""
    y0, y1, x0, x1 = box
    sub, sv = grid[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    hh, ww = sub.shape
    out: dict[tuple, list[tuple[int, int]]] = {}
    for comp in components(label_components(sub, sv)):
        ys, xs = comp // ww, comp % ww
        if ((ys.min() == 0 and y0 > 0) or (ys.max() == hh - 1 and y1 < SIZE)
                or (xs.min() == 0 and x0 > 0) or (xs.max() == ww - 1 and x1 < SIZE)):
            continue  # may extend past the crop; its shape is unknown
        key, a, b = shape_key(sub, ys, xs)
        out.setdefault(key, []).append((a + y0, b + x0))
    return out


def match_moves(prev: np.ndarray, cur: np.ndarray, ignore: np.ndarray,
                bg: int) -> list[tuple[tuple, int, int, tuple[int, int], int]]:
    """Objects that translated between two frames:
    [(shape key, dy, dx, new top-left, copies of that shape nearby)].

    Only the changed region (plus a margin) is labelled, so this stays cheap.
    Objects are matched by identical shape key, nearest first. The copy count
    matters because an avatar stepping between identical tiles uncovers one
    and covers another, which looks like a tile moving.
    """
    diff = (prev != cur) & ~ignore
    if not diff.any():
        return []
    ys, xs = np.nonzero(diff)
    box = (max(int(ys.min()) - MOVE_MARGIN, 0), min(int(ys.max()) + MOVE_MARGIN + 1, SIZE),
           max(int(xs.min()) - MOVE_MARGIN, 0), min(int(xs.max()) + MOVE_MARGIN + 1, SIZE))
    before = objects_in(prev, (prev != bg) & ~ignore, box)
    after = objects_in(cur, (cur != bg) & ~ignore, box)
    moves = []
    for key in before.keys() & after.keys():
        olds, news = before[key], list(after[key])
        if len(olds) != len(news):
            continue
        for o in olds:
            j = min(range(len(news)), key=lambda j: abs(news[j][0] - o[0]) + abs(news[j][1] - o[1]))
            n = news.pop(j)
            if n != o:
                moves.append((key, n[0] - o[0], n[1] - o[1], n, len(olds)))
    return moves


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
        self.bar = np.zeros((SIZE, SIZE), dtype=bool)  # budget-bar pixels only
        # pixels that changed while the controllable object was blocked:
        # enemies, NPCs, animations. Masked once seen twice.
        self.noise_count = np.zeros((SIZE, SIZE), dtype=np.int16)
        self.noisy = np.zeros((SIZE, SIZE), dtype=bool)

    @property
    def ignore(self) -> np.ndarray:
        """Pixels that never count as state or as an effect of our action."""
        return self.hud | self.noisy

    def add_noise(self, prev: np.ndarray, cur: np.ndarray) -> None:
        changed = (prev != cur) & ~self.hud
        self.noise_count[changed] += 1
        noisy = self.noise_count >= 2
        # If "noise" covers much of the screen it is the game itself reacting
        # (mirrors, linked pieces), not a few patrolling enemies: don't mask it.
        self.noisy = noisy if noisy.sum() <= 0.15 * SIZE * SIZE else np.zeros_like(noisy)

    def budget_used(self, start: Optional[np.ndarray], cur: np.ndarray) -> Optional[float]:
        """Share of the budget bar spent since `start`, or None if no bar is known."""
        n = int(self.bar.sum())
        if start is None or n < 4:
            return None
        return float((start[self.bar] != cur[self.bar]).sum()) / n

    def observe_step(self, prev: np.ndarray, cur: np.ndarray, step: int) -> bool:
        """Learn the HUD from one in-level transition; return True if the world changed."""
        diff = prev != cur
        if not diff.any():
            return False
        world = diff & ~self.hud
        if not world.any():
            return False
        if not (world & ~self.noisy).any():
            return False  # only enemies / animations moved
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
            elif not self.noisy[ys, xs].all():
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
            self.bar[idx, lo:hi + 1] = True
        else:
            self.hud[lo:hi + 1, idx] = True
            self.bar[lo:hi + 1, idx] = True

    def key(self, grid: np.ndarray, level: int) -> int:
        masked = np.where(self.ignore, -1, grid).astype(np.int8)
        return hash((level, masked.tobytes()))

    def locate(self, grid: np.ndarray, key: tuple) -> list[tuple[int, int]]:
        """Top-left corners of every object with this shape key in the full frame."""
        bg = self.background(grid)
        return objects_in(grid, (grid != bg) & ~self.hud, (0, SIZE, 0, SIZE)).get(key, [])

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
    """v2 explorer plus level transfer and a goal-directed navigation planner."""

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
        self.level = 0

        # v2: effect model for moves
        self.ctrl_votes: dict[tuple, float] = {}           # shape key -> decayed move count
        self.key_disp: dict[tuple, set[tuple[int, int, int]]] = {}  # key -> {(action, dy, dx)}
        self.ctrl_key: Optional[tuple] = None
        self.ctrl_box: Optional[tuple[int, int, int, int]] = None  # y0, x0, y1, x1
        self.ctrl_colours: set[int] = set()
        self.disp: dict[int, dict[tuple[int, int], int]] = {}  # action -> {(dy, dx): n}, this level
        self.block_cnt: dict[int, int] = {}   # colour -> times it stopped a move
        self.pass_cnt: dict[int, int] = {}    # colour -> times a move went onto it
        self.visited_pos: set[tuple[int, int, int]] = set()  # (level, y0, x0) of the ctrl box
        self.move_info: dict[int, dict[int, tuple]] = {}      # state -> {action: prediction}
        self.attempt_start: Optional[np.ndarray] = None       # first frame of this attempt
        self.hazard_deaths = 0
        self.blocked: set[tuple[int, int, int, int]] = set()  # (level, y, x, action) that didn't move

        # v3: what won earlier levels
        self.win_colours: set[int] = set()        # colours a winning move went onto
        self.win_classes: set[tuple] = set()      # object classes a winning click hit
        self.win_click_colours: set[int] = set()
        self.win_simple: set[int] = set()         # other winning action ids
        self.prev_disp: dict[int, dict[tuple[int, int], int]] = {}  # last level's move model
        # v3: navigation planner
        self.goal_plan: list[tuple[int, tuple[int, int]]] = []  # [(action, expected avatar pos)]
        self.goal_expect: Optional[tuple[int, int]] = None
        self.goal_tried: set[tuple[int, int]] = set()  # target positions reached this level
        self.touched: set[int] = set()             # target colours reached this level
        self.no_plan: dict[int, int] = {}          # state -> knowledge version with no plan
        self.knowledge = 0                         # bumps when a colour is first classified
        self.goal_steps = 0
        self.goal_stall = 0                        # planned moves in a row that didn't move us
        self.goal_replans = 0                      # replans since the last target was reached

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
        self.level = level
        if level > self.prev_level and self.prev_action is not None:
            self.level_ups.append(self.action_counter)
            logger.info(f"{self.game_id} - level {level} reached at action {self.action_counter}")
        if level > self.prev_level and self.prev_action is not None and self.prev_grid is not None:
            self._record_win(self.prev_action, self.prev_class, self.prev_grid)
        if level > self.prev_level:
            # Re-verify the action -> direction mapping on every level (R8),
            # but keep the old one as a provisional model until then.
            if self.disp:
                self.prev_disp = {a: dict(c) for a, c in self.disp.items()}
            self.goal_plan, self.goal_tried, self.touched = [], set(), set()
            self.disp.clear()
            self.ctrl_box = None
            for k in self.ctrl_votes:
                self.ctrl_votes[k] *= 0.25
        same_level = level == self.prev_level and self.prev_grid is not None
        dead = latest.state is GameState.GAME_OVER
        effective = False
        if grid is not None and same_level:
            effective = self.perc.observe_step(self.prev_grid, grid, self.step_no)
            act = self.prev_action
            if act is not None and act[0] == "s" and act[1] in MOVES:
                self._learn_move(act[1], self.prev_grid, grid)
                effective = effective and bool(((self.prev_grid != grid) & ~self.perc.ignore).any())
        hazard = False
        if dead and self.prev_grid is not None and self.prev_action is not None:
            hazard = self._death_cause(self.prev_grid)
        key = self.perc.key(grid, level) if grid is not None else None
        if grid is not None and (self.prev_grid is None or level > self.prev_level):
            self._start_attempt(grid)

        if self.prev_action is not None and self.prev_key is not None:
            leveled = level > self.prev_level
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
                # move open; only a move that kills twice, or once with most
                # of the budget left, is recorded as deadly.
                n = self.deaths.get((self.prev_key, self.prev_action), 0) + (2 if hazard else 1)
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

    # ── Effect model for moves (v2) ────────────────────────────────────────────

    def _start_attempt(self, grid: np.ndarray) -> None:
        """First frame after a reset or level-up: remember it and find the avatar."""
        self.attempt_start = grid
        self.ctrl_box = None
        self.goal_plan, self.no_plan = [], {}
        if self.ctrl_key is not None:
            found = self.perc.locate(grid, self.ctrl_key)
            if found:
                y, x = found[0]
                self.ctrl_box = (y, x, y + self.ctrl_key[1], x + self.ctrl_key[2])
                self.visited_pos.add((self.level, y, x))

    def _update_ctrl_key(self) -> None:
        best, best_v = None, 0.0
        for k, v in self.ctrl_votes.items():
            seen = self.key_disp.get(k, set())
            # controllable = moves often, and where it goes depends on the action
            if (v >= CTRL_MIN_VOTES and v > best_v
                    and len({(dy, dx) for _, dy, dx in seen}) >= 2
                    and len({a for a, _, _ in seen}) >= 2):
                best, best_v = k, v
        if best != self.ctrl_key:
            self.ctrl_key = best
            self.ctrl_colours = {best[0]} if best else set()

    def _disp_for(self, action: int) -> Optional[tuple[int, int]]:
        counts = self.disp.get(action) or self.prev_disp.get(action)
        if not counts:
            return None
        d = max(counts, key=counts.get)
        return None if d == (0, 0) else d

    def _region_colours(self, grid: np.ndarray, box: tuple[int, int, int, int]) -> Optional[frozenset]:
        """Colours under a box, minus background, HUD and the avatar's own; None if off-screen."""
        y0, x0, y1, x1 = box
        if y0 < 0 or x0 < 0 or y1 > SIZE or x1 > SIZE:
            return None
        region = grid[y0:y1, x0:x1][~self.perc.hud[y0:y1, x0:x1]]
        bg = self.perc.background(grid)
        return frozenset(int(c) for c in np.unique(region)) - {bg} - self.ctrl_colours

    def _learn_move(self, action: int, prev: np.ndarray, cur: np.ndarray) -> None:
        """Learn which object the arrows move, by how much, and what blocks it."""
        moves = match_moves(prev, cur, self.perc.ignore, self.perc.background(cur))
        for k in list(self.ctrl_votes):
            self.ctrl_votes[k] *= 0.97
            if self.ctrl_votes[k] < 0.05:
                del self.ctrl_votes[k]
        for k, dy, dx, _, copies in moves:
            # avatars are usually one of a kind; tiles and nodes come in crowds
            self.ctrl_votes[k] = self.ctrl_votes.get(k, 0.0) + 1.0 / copies
            seen = self.key_disp.setdefault(k, set())
            if len(seen) < 32:
                seen.add((action, dy, dx))
        self._update_ctrl_key()
        ck = self.ctrl_key
        if ck is None:
            return
        mine = [(dy, dx, pos) for k, dy, dx, pos, _ in moves if k == ck]
        if not mine and self.ctrl_box is not None:
            # The avatar may have changed shape or colour (changer tiles). Adopt
            # an object that started where it was and moved as predicted.
            d = self._disp_for(action)
            y0, x0 = self.ctrl_box[0], self.ctrl_box[1]
            for k, dy, dx, pos, _ in moves:
                if (d is not None and (dy, dx) == d
                        and abs(pos[0] - dy - y0) <= 3 and abs(pos[1] - dx - x0) <= 3):
                    self.ctrl_votes[k] = max(self.ctrl_votes.get(k, 0.0), self.ctrl_votes.get(ck, 0.0)) + 1.0
                    self.key_disp.setdefault(k, set()).update(self.key_disp.get(ck, set()))
                    self.ctrl_key, self.ctrl_colours, ck = k, {k[0]}, k
                    mine = [(dy, dx, pos)]
                    break
        if not mine and self.ctrl_box is not None:
            # Another object moved exactly as this arrow predicts: the arrows
            # now drive it (a new level, or a different piece is selected).
            d = self._disp_for(action)
            same = sorted((m for m in moves if d is not None and (m[1], m[2]) == d), key=lambda m: m[4])
            if same:
                k, dy, dx, pos, _ = same[0]
                self.ctrl_votes[k] = max(self.ctrl_votes.get(k, 0.0), self.ctrl_votes.get(ck, 0.0)) + 1.0
                self.key_disp.setdefault(k, set()).update(self.key_disp.get(ck, set()))
                self.ctrl_key, self.ctrl_colours, ck = k, {k[0]}, k
                mine = [(dy, dx, pos)]
        if mine:
            self.goal_stall = 0
            if self.ctrl_box is not None and len(mine) > 1:
                y0, x0 = self.ctrl_box[0], self.ctrl_box[1]
                mine.sort(key=lambda m: abs(m[2][0] - m[0] - y0) + abs(m[2][1] - m[1] - x0))
            dy, dx, (y, x) = mine[0]
            counts = self.disp.setdefault(action, {})
            counts[(dy, dx)] = counts.get((dy, dx), 0) + 1
            # The avatar box is the main piece only, so one spot always has one
            # coordinate (a union of whichever pieces matched would wobble).
            # Pieces moving with it are part of the avatar, never obstacles.
            y0, x0, y1, x1 = y, x, y + ck[1], x + ck[2]
            for k, ddy, ddx, (yy, xx), _ in moves:
                if (ddy, ddx) == (dy, dx) and abs(yy - y) <= 12 and abs(xx - x) <= 12:
                    self.ctrl_colours.add(k[0])
            if action in self.prev_disp and (dy, dx) not in self.prev_disp[action]:
                self.prev_disp = {}  # the old level's mapping no longer holds
            for c in self._region_colours(prev, (y0, x0, y1, x1)) or ():
                if c not in self.pass_cnt:
                    self.knowledge += 1
                self.pass_cnt[c] = self.pass_cnt.get(c, 0) + 1
            self.ctrl_box = (y0, x0, y1, x1)
            self.visited_pos.add((self.level, y0, x0))
        elif self.ctrl_box is not None:
            d = self._disp_for(action)
            if d is not None:
                y0, x0, y1, x1 = self.ctrl_box
                self.blocked.add((self.level, y0, x0, action))
                for c in self._region_colours(prev, (y0 + d[0], x0 + d[1], y1 + d[0], x1 + d[1])) or ():
                    if c not in self.block_cnt:
                        self.knowledge += 1
                    self.block_cnt[c] = self.block_cnt.get(c, 0) + 1
            # whatever changed while we stood still is someone else's doing
            keep = np.zeros((SIZE, SIZE), dtype=bool)
            y0, x0, y1, x1 = self.ctrl_box
            keep[max(y0 - 8, 0):y1 + 8, max(x0 - 8, 0):x1 + 8] = True
            self.perc.add_noise(np.where(keep, cur, prev), cur)

    def _death_cause(self, prev_grid: np.ndarray) -> bool:
        """True if the last death was a hazard rather than the budget running out."""
        used = self.perc.budget_used(self.attempt_start, prev_grid)
        if used is None or used >= HAZARD_MAX_USED:
            return False
        self.hazard_deaths += 1
        # Only the move itself is blamed. Blaming the colours it moved into
        # was tried and hurt: enemies kill on paths whose colour is harmless.
        logger.info(f"{self.game_id} - hazard death ({used:.0%} of budget used)")
        return True

    # ── Level transfer and goal planner (v3) ─────────────────────────────────

    def _record_win(self, act: tuple, cls: Optional[tuple], prev: np.ndarray) -> None:
        """Remember what the level-winning action was, to try it first next level."""
        if act[0] == "c":
            if cls is not None:
                self.win_classes.add(cls)
                if cls[0] == "obj":
                    self.win_click_colours.add(cls[1])
        elif act[1] in MOVES and self.ctrl_box is not None and self._disp_for(act[1]):
            dy, dx = self._disp_for(act[1])
            y0, x0, y1, x1 = self.ctrl_box
            cols = self._region_colours(prev, (y0 + dy, x0 + dx, y1 + dy, x1 + dx)) or frozenset()
            counts = np.bincount(prev.ravel().astype(np.int64) & 15, minlength=16)
            self.win_colours |= {c for c in cols if counts[c] < GOAL_MAX_SHARE * SIZE * SIZE}
        else:
            self.win_simple.add(act[1])
        logger.info(f"{self.game_id} - level won by {act[:2]}; goal colours {sorted(self.win_colours)}, "
                    f"click colours {sorted(self.win_click_colours)}, actions {sorted(self.win_simple)}")

    def _goal_targets(self, grid: np.ndarray) -> int:
        """Bitmask of target colours: known goal colours, else untouched small objects."""
        counts = np.bincount(grid[~self.perc.hud].astype(np.int64) & 15, minlength=16)
        bg = self.perc.background(grid)
        present = {c for c in range(16) if counts[c] and c != bg and c not in self.ctrl_colours}
        goals = self.win_colours & present
        if not goals:
            goals = {c for c in present if counts[c] <= TOUCH_MAX_PX and c not in self.touched
                     and c not in self.pass_cnt and c not in self.block_cnt}
        bits = 0
        for c in goals:
            bits |= 1 << c
        return bits

    def _box_colour_bits(self, grid: np.ndarray) -> np.ndarray:
        """For every avatar top-left (y, x): bitmask of colours under the avatar's box there."""
        y0, x0, y1, x1 = self.ctrl_box
        h, w = y1 - y0, x1 - x0
        bg = self.perc.background(grid)
        g = grid.copy()
        g[self.perc.hud] = bg
        here = g[y0:y1, x0:x1]
        here[np.isin(here, list(self.ctrl_colours))] = bg  # the avatar itself is not an obstacle
        bits = np.zeros((SIZE - h + 1, SIZE - w + 1), dtype=np.int32)
        for c in np.unique(g):
            if c == bg:
                continue
            sat = np.zeros((SIZE + 1, SIZE + 1), dtype=np.int32)
            sat[1:, 1:] = (g == c).astype(np.int32).cumsum(0).cumsum(1)
            cnt = sat[h:, w:] - sat[:-h, w:] - sat[h:, :-w] + sat[:-h, :-w]
            bits |= (cnt > 0).astype(np.int32) << int(c)
        return bits

    def _plan_to(self, grid: np.ndarray, targets: int, legal: list[int]) -> Optional[list]:
        """BFS over avatar positions to the nearest untried box overlapping a target colour."""
        moves = [(a, self._disp_for(a)) for a in MOVES if a in legal and self._disp_for(a)]
        if len(moves) < 2:
            return None
        bits = self._box_colour_bits(grid)
        hh, ww = bits.shape
        passable = 0
        for c in set(self.pass_cnt) | self.ctrl_colours:
            if self.pass_cnt.get(c, 0) >= self.block_cnt.get(c, 0):
                passable |= 1 << c
        start = (self.ctrl_box[0], self.ctrl_box[1])
        parent: dict[tuple[int, int], Optional[tuple]] = {start: None}
        queue = deque([start])
        while queue and len(parent) < PLAN_MAX_NODES:
            pos = queue.popleft()
            for a, (dy, dx) in moves:
                nxt = (pos[0] + dy, pos[1] + dx)
                if nxt in parent or not (0 <= nxt[0] < hh and 0 <= nxt[1] < ww):
                    continue
                if (self.level, pos[0], pos[1], a) in self.blocked:
                    continue
                b = int(bits[nxt])
                if b & targets and nxt not in self.goal_tried:
                    parent[nxt] = (pos, a)
                    plan, node = [], nxt
                    while parent[node] is not None:
                        prev_pos, act = parent[node]
                        plan.append((act, node))
                        node = prev_pos
                    return plan[::-1]
                if b & ~passable:
                    continue  # wall, or a colour never stood on
                parent[nxt] = (pos, a)
                queue.append(nxt)
        return None

    def _goal_step(self, key: int, grid: np.ndarray, latest: FrameData) -> Optional[int]:
        """Next move of the current goal plan, planning a new one if needed."""
        if self.ctrl_box is None or self.goal_stall >= 3:
            return None  # no avatar, or the model is wrong until a move succeeds again
        legal = self._legal_ids(latest)
        pos = (self.ctrl_box[0], self.ctrl_box[1])
        on_track = (self.goal_expect is not None and abs(pos[0] - self.goal_expect[0]) <= 2
                    and abs(pos[1] - self.goal_expect[1]) <= 2)  # sprites jitter as they animate
        if self.goal_plan and on_track and self.goal_plan[0][0] in legal:
            act, self.goal_expect = self.goal_plan.pop(0)
        else:
            if self.goal_plan:
                self.goal_replans += 1
                if self.goal_replans > 6:  # this target can't be reached as modelled
                    self.goal_tried.add(self.goal_plan[-1][1])
                    self.goal_replans = 0
            self.goal_plan = []
            if self.no_plan.get(key) == self.knowledge:
                return None
            targets = self._goal_targets(grid)
            plan = self._plan_to(grid, targets, legal) if targets else None
            if not plan:
                self.no_plan[key] = self.knowledge
                return None
            (act, self.goal_expect), self.goal_plan = plan[0], plan[1:]
        if not self.goal_plan:
            self.goal_tried.add(self.goal_expect)  # this move reaches the target
            self.goal_replans = 0
            y, x = self.goal_expect
            h, w = self.ctrl_box[2] - self.ctrl_box[0], self.ctrl_box[3] - self.ctrl_box[1]
            self.touched |= set(self._region_colours(grid, (y, x, y + h, x + w)) or ())
        self.goal_steps += 1
        self.goal_stall += 1  # reset by _learn_move when the avatar actually moves
        return act

    def _note_moves(self, key: int, grid: np.ndarray, legal: list[int]) -> None:
        """Predict where each move would take the avatar from this state."""
        if self.ctrl_box is None:
            return
        info = {}
        y0, x0, y1, x1 = self.ctrl_box
        for a in MOVES:
            d = self._disp_for(a) if a in legal else None
            if d is None:
                continue
            dest = (y0 + d[0], x0 + d[1], y1 + d[0], x1 + d[1])
            cols = self._region_colours(grid, dest)
            info[a] = ("off",) if cols is None else (cols, (self.level, dest[0], dest[1]))
        self.move_info[key] = info

    def _move_prior(self, key: int, action: int) -> Optional[float]:
        info = self.move_info.get(key, {}).get(action)
        if info is None:
            return None
        if info[0] == "off":
            return 0.05
        cols, dest = info
        walls = {c for c in cols if self.block_cnt.get(c, 0) > self.pass_cnt.get(c, 0)}
        if cols and walls == cols:
            return 0.05
        if any(c not in walls and self.pass_cnt.get(c, 0) == 0 for c in cols):
            return 1.0  # something never touched: go touch it
        return 0.3 if dest in self.visited_pos else 0.8

    def _prior(self, action_key: tuple, cls: Optional[tuple], key: Optional[int] = None) -> float:
        # Transfer: whatever won an earlier level is tried first (§5.3-G).
        if cls is not None and cls in self.win_classes:
            return WIN_PRIOR
        if cls is not None and cls[0] == "obj" and cls[1] in self.win_click_colours:
            return WIN_PRIOR - 0.1
        if cls is None and action_key[1] in self.win_simple:
            return WIN_PRIOR - 0.1
        if cls is None:
            if key is not None and action_key[1] in MOVES:
                mp = self._move_prior(key, action_key[1])
                if mp is not None:
                    return mp
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
            p = self._prior(ak, cls, key)
            if p > best:
                best, best_c = p, (ak, cls)
        return best, best_c

    def _pick(self, key: int, grid: np.ndarray, latest: FrameData) -> tuple[tuple, Optional[tuple]]:
        cands = self._state_candidates(key, grid, latest)
        cls_of = dict(cands)
        self._note_moves(key, grid, self._legal_ids(latest))

        # 0. goal hypothesis: navigate to a goal colour or an untouched object
        act = self._goal_step(key, grid, latest)
        if act is not None and ("s", act) in cls_of:
            self.path = []
            return ("s", act), None

        # 1. keep walking a planned path while the world behaves as predicted
        if self.path and key == self.expect and self.path[0][0] in cls_of:
            ak, self.expect = self.path.pop(0)
            return ak, cls_of[ak]
        self.path = []

        # 2. a high-value move here, else walk a few steps to one, else settle
        #    for a low-value move here, else walk any distance to one
        best, choice = self._best_untried(key)
        for threshold, max_depth in ((HIGH_PRIOR, HIGH_ROUTE_MAX), (LOCAL_MIN_PRIOR, None)):
            if choice is not None and best >= threshold:
                tried = self.edges.get(key, {})
                ties = [(ak, c) for ak, c in cands
                        if ak not in tried and abs(self._prior(ak, c, key) - best) < 1e-9]
                return self.rng.choice(ties)
            route = self._route_to_frontier(key, threshold, max_depth)
            if route and route[0][0] in cls_of:
                (ak, self.expect), self.path = route[0], route[1:]
                return ak, cls_of[ak]

        # 3. leftovers: anything untried, else the move into the least-visited state
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

    def _route_to_frontier(self, start: int, threshold: float,
                           max_depth: Optional[int]) -> list[tuple[tuple, int]]:
        """BFS over known effective edges to the nearest state with an untried action >= threshold."""
        parent: dict[int, tuple[int, tuple]] = {start: (start, ("s", 0))}
        depth = {start: 0}
        queue = deque([start])
        while queue and len(parent) < BFS_MAX_NODES:
            s = queue.popleft()
            if s != start and self._best_untried(s)[0] >= threshold:
                route = []
                while s != start:
                    p, ak = parent[s]
                    route.append((ak, s))
                    s = p
                return route[::-1]
            if max_depth is not None and depth[s] >= max_depth:
                continue
            for ak, nxt in self.edges.get(s, {}).items():
                if isinstance(nxt, int) and nxt != s and nxt not in parent:
                    parent[nxt] = (s, ak)
                    depth[nxt] = depth[s] + 1
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
                f"ctrl {self.ctrl_key[:3] if self.ctrl_key else None}, step {self.disp}, "
                f"noisy px {int(self.perc.noisy.sum())}, hazard deaths {self.hazard_deaths}, "
                f"goal colours {sorted(self.win_colours)}, win classes {len(self.win_classes)}, "
                f"win actions {sorted(self.win_simple)}, planner steps {self.goal_steps}, "
                f"failed steps {self._failures}, gave up {self._gave_up}, "
                f"p95 decision {p95:.2f} ms, {self.seconds}s"
            )
        super().cleanup(*args, **kwargs)
