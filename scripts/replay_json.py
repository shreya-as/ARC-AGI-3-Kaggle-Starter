"""Run local ARC-AGI-3 games and emit replay data as JSON.

This is intentionally quiet and machine-facing; the React observer consumes
the final JSON payload from stdout while stderr keeps ARC engine logs separate.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode


def load_my_agent_class():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit("agent/my_agent.py must define a class named `MyAgent`")
    return module.MyAgent


def enum_value(value: Any) -> Any:
    return getattr(value, "name", value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return enum_value(value)


def action_payload(action: Any) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "id": enum_value(getattr(action, "id", None)),
        "data": json_safe(getattr(action, "data", {}) or {}),
        "reasoning": json_safe(getattr(action, "reasoning", None)),
    }


def get_scorecard(arcade: Any) -> Any | None:
    try:
        return arcade.get_scorecard()
    except Exception:
        return None


def scorecard_score(scorecard: Any) -> float | None:
    if scorecard is None:
        return None
    raw_score = getattr(scorecard, "score", scorecard)
    try:
        return round(float(raw_score), 6)
    except (TypeError, ValueError):
        return None


def scorecard_environment_scores(scorecard: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    for environment in getattr(scorecard, "environments", []) or []:
        environment_id = str(getattr(environment, "id", ""))
        if not environment_id:
            continue
        try:
            scores[environment_id.split("-")[0]] = round(float(environment.score), 6)
        except (TypeError, ValueError):
            continue
    return scores


def serialize_frame(frame: Any, index: int) -> dict[str, Any]:
    grids = [
        [[int(cell) for cell in row] for row in grid]
        for grid in (getattr(frame, "frame", None) or [])
    ]
    return {
        "index": index,
        "state": enum_value(getattr(frame, "state", "")),
        "levelsCompleted": int(getattr(frame, "levels_completed", 0) or 0),
        "winLevels": int(getattr(frame, "win_levels", 0) or 0),
        "action": action_payload(getattr(frame, "action_input", None)),
        "telemetry": json_safe(
            getattr(getattr(frame, "action_input", None), "reasoning", None)
        ),
        "availableActions": [
            enum_value(action) for action in (getattr(frame, "available_actions", None) or [])
        ],
        "grids": grids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default=None, help="Short game id, comma list, or omitted for all.")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--generalization", action="store_true", help="Disable openings, disk memory, and engine planning.")
    parser.add_argument("--coarse-hash", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--no-observed-model", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    with contextlib.redirect_stdout(sys.stderr):
        arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        environments = arcade.get_environments()

    games = [
        {
            "id": env.game_id.split("-")[0],
            "fullId": env.game_id,
            "title": getattr(env, "title", None) or env.game_id.split("-")[0].upper(),
            "defaultFps": getattr(env, "default_fps", None) or 8,
            "tags": getattr(env, "tags", None) or [],
            "baselineActions": getattr(env, "baseline_actions", None) or [],
        }
        for env in environments
    ]

    if args.list:
        print(json.dumps({"games": games}))
        return

    if args.game:
        wanted = {game.strip().split("-")[0] for game in args.game.split(",") if game.strip()}
        game_ids = [game["id"] for game in games if game["id"] in wanted]
        missing = sorted(wanted - set(game_ids))
        if missing:
            raise SystemExit(f"Unknown game id(s): {missing}")
    else:
        game_ids = [game["id"] for game in games]

    MyAgent = load_my_agent_class()
    MyAgent.EXACT_STATE_HASH = False
    MyAgent.EXACT_CHANGE_DETECTION = not args.coarse_hash
    MyAgent.SEED_OFFSET = args.seed
    MyAgent.USE_OBSERVED_MODEL = not args.no_observed_model
    if args.generalization:
        MyAgent.OPENING_BOOK = {}
        MyAgent.USE_PERSISTENT_MEMORY = False
        MyAgent.USE_ENGINE_PLANNER = False
    if hasattr(MyAgent, "MAX_ACTIONS"):
        MyAgent.MAX_ACTIONS = min(MyAgent.MAX_ACTIONS, args.max_steps)

    runs = []
    for game_id in game_ids:
        with contextlib.redirect_stdout(sys.stderr):
            env = arcade.make(game_id)
            if env is None:
                runs.append({"gameId": game_id, "error": "Could not create environment."})
                continue

            agent = MyAgent(
                card_id="local-ui",
                game_id=game_id,
                agent_name=f"MyAgent.ui.{game_id}",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=env,
                tags=["local-ui"],
            )
            agent.main()

        source_frames = agent.frames[-1:] if args.scores_only else agent.frames
        frames = [serialize_frame(frame, index) for index, frame in enumerate(source_frames)]
        non_empty = [frame for frame in frames if frame["grids"]]
        exploration_log = [] if args.scores_only else json_safe(getattr(agent, "exploration_log", []))
        if isinstance(exploration_log, list):
            for index, entry in enumerate(exploration_log):
                frame_index = index + 1
                if frame_index >= len(non_empty):
                    break
                if not non_empty[frame_index].get("telemetry"):
                    non_empty[frame_index]["telemetry"] = entry
        final = non_empty[-1] if non_empty else frames[-1] if frames else None
        run_game = next((game for game in games if game["id"] == game_id), None)
        win_levels = final["winLevels"] if final else 0
        completed = final["levelsCompleted"] if final else 0
        level_completion_score = round((completed / win_levels) * 100, 2) if win_levels else 0
        runs.append(
            {
                "gameId": game_id,
                "title": (run_game or {}).get("title", game_id.upper()),
                "defaultFps": (run_game or {}).get("defaultFps", 8),
                "actions": int(getattr(agent, "action_counter", 0) or 0),
                "levelsCompleted": completed,
                "winLevels": win_levels,
                "score": level_completion_score,
                "levelCompletionScore": level_completion_score,
                "arcScore": None,
                "state": final["state"] if final else "UNKNOWN",
                "reward": round(float(getattr(agent, "cumulative_reward", 0.0)), 4),
                "knownStates": len(getattr(agent, "visited", {})),
                "archiveSize": len(getattr(agent, "state_paths", {})),
                "explorationLog": exploration_log,
                "frames": [] if args.scores_only else non_empty,
            }
        )

    total_levels = sum(run.get("winLevels", 0) for run in runs)
    completed_levels = sum(run.get("levelsCompleted", 0) for run in runs)
    level_completion_score = (
        round((completed_levels / total_levels) * 100, 2)
        if total_levels
        else 0
    )
    scorecard = get_scorecard(arcade)
    arc_score = scorecard_score(scorecard)
    environment_scores = scorecard_environment_scores(scorecard)
    for run in runs:
        run_arc_score = environment_scores.get(run.get("gameId"))
        if run_arc_score is not None:
            run["arcScore"] = run_arc_score
            run["score"] = run_arc_score
    print(
        json.dumps(
            {
                "runs": runs,
                "summary": {
                    "games": len(runs),
                    "actions": sum(run.get("actions", 0) for run in runs),
                    "reward": round(sum(run.get("reward", 0.0) for run in runs), 4),
                    "levelsCompleted": completed_levels,
                    "totalLevels": total_levels,
                    "score": arc_score if arc_score is not None else level_completion_score,
                    "arcScore": arc_score,
                    "levelCompletionScore": level_completion_score,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
