"""v0 agent: the safety shell from AGENT_PLAN.md §5.3-A (Phase A).

Policy is still uniform random, but only over the game's legal actions. Everything
else here exists so that later versions can run thousands of actions per game on
Kaggle without crashing, sending illegal input, or exhausting RAM/disk:

  - Legal actions only: sample from `latest_frame.available_actions`.
  - RESET iff state is NOT_PLAYED or GAME_OVER; stop on WIN; never send any
    other action in GAME_OVER/WIN (the gateway answers HTTP 400).
  - Crash-proof stepping: a failed step returns None instead of killing the
    game's thread. After 3 consecutive failures send RESET; if that fails too,
    stop the game.
  - Stable per-game seeding: `random.Random(crc32(game_id, SEED_OFFSET))`, no
    global RNG (shared across threads) and no `hash()` (salted per process).
  - Budgets: MAX_ACTIONS in the thousands, plus a per-game wall-clock limit.
  - Bounded memory: keep only the last few frames; no agent-side JSON recording.
  - ACTION6 coordinates are kept per agent and sent from there, so the shared
    `GameAction.ACTION6` singleton can't leak clicks between game threads.

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
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

logger = logging.getLogger()

MAX_ACTIONS_DEFAULT = int(os.getenv("AGENT_MAX_ACTIONS", "2000"))
WALL_SECONDS_DEFAULT = float(os.getenv("AGENT_WALL_SECONDS", "1800"))
SEED_OFFSET = int(os.getenv("SEED_OFFSET", "0"))

FRAMES_KEPT = 8          # frames held in self.frames
MAX_FAIL_STREAK = 3      # consecutive failed steps before forcing a RESET
DECISION_SAMPLES = 4096  # decision-time samples kept for the p95 report


class MyAgent(Agent):
    """Legal-only random play inside a crash-proof, bounded shell."""

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

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    # ── Main loop ────────────────────────────────────────────────────────────
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
                if action is GameAction.RESET and self._fail_streak > MAX_FAIL_STREAK:
                    logger.error(f"{self.game_id} - RESET failed after {self._fail_streak - 1} failed steps; stopping")
                    self._gave_up = True
            self.action_counter += 1

        self.cleanup()

    def _latest_frame(self) -> FrameData:
        """Current observation; falls back to the last good frame if unavailable."""
        try:
            return self._convert_raw_frame_data(self.arc_env.observation_space if self.arc_env else None)
        except Exception:
            return self.frames[-1]

    # ── Policy ───────────────────────────────────────────────────────────────

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stopping early never earns points (§1.4), so stop only on WIN, when
        # the shell has given up, or when the wall clock runs out.
        return (
            latest_frame.state is GameState.WIN
            or self._gave_up
            or time.time() >= self._deadline
        )

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        if self._fail_streak >= MAX_FAIL_STREAK:
            return GameAction.RESET
        return self._random_legal_action(latest_frame)

    def _legal_actions(self, latest_frame: FrameData) -> list[GameAction]:
        legal = []
        for action_id in latest_frame.available_actions or []:
            try:
                action = GameAction.from_id(int(action_id))
            except ValueError:
                continue
            if action is not GameAction.RESET:
                legal.append(action)
        return legal

    def _random_legal_action(self, latest_frame: FrameData) -> GameAction:
        legal = self._legal_actions(latest_frame)
        if not legal:
            # No legal list to sample from; RESET is the only input that can't
            # be illegal while the game is running.
            return GameAction.RESET
        action = self.rng.choice(legal)
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

    # ── Stepping ─────────────────────────────────────────────────────────────

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

    # ── Bounded memory ───────────────────────────────────────────────────────

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
                f"{self.game_id} - actions {self.action_counter}, levels {self.levels_completed}, "
                f"failed steps {self._failures}, gave up {self._gave_up}, "
                f"p95 decision {p95:.2f} ms, {self.seconds}s"
            )
        super().cleanup(*args, **kwargs)
