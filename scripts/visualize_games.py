"""Render each ARC-AGI-3 game's playthrough as an animated GIF.

Plays `agent/my_agent.py` locally against one or more games (same in-process
harness as `scripts/play_local.py`), then converts every recorded frame into
an RGB image (via `arc_agi.rendering`) and stitches them into a GIF per game.

Usage:
    .venv/bin/python scripts/visualize_games.py --game ls20 --max-steps 80
    .venv/bin/python scripts/visualize_games.py --max-steps 40  # all games
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import numpy as np
from arc_agi import OperationMode
from arc_agi.rendering import frame_to_rgb_array
from PIL import Image

import arc_agi


def load_my_agent_class():
    """Import MyAgent from agent/my_agent.py via importlib."""
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


def save_gif(grids: list[np.ndarray], out_path: Path, fps: int, scale: int) -> None:
    images = [
        Image.fromarray(frame_to_rgb_array(0, grid, scale=scale))
        for grid in grids
    ]
    duration_ms = max(1, round(1000 / fps))
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None,
                   help="Game id to visualize. If omitted, visualizes ALL games. "
                        "Comma-separated list also accepted, e.g. ls20,vc33.")
    p.add_argument("--max-steps", type=int, default=80,
                   help="Per-game cap on actions (overrides MyAgent.MAX_ACTIONS).")
    p.add_argument("--out-dir", default="game_gifs",
                   help="Directory to write <game_id>.gif files into.")
    p.add_argument("--scale", type=int, default=8,
                   help="Upscale factor per grid cell (64x64 -> 64*scale square).")
    p.add_argument("--fps", type=int, default=None,
                   help="Override playback FPS (default: each game's own default_fps).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()
    fps_by_id = {e.game_id.split("-")[0]: (e.default_fps or 8) for e in all_envs}

    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}. "
                              f"Run scripts/play_local.py --list.")
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]
        print(f"No --game specified; visualizing all {len(game_ids)} games.\n")

    MyAgentCls = load_my_agent_class()
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)

    for i, game_id in enumerate(game_ids, 1):
        print(f"=== [{i}/{len(game_ids)}] {game_id} ===")
        env = arc.make(game_id)
        if env is None:
            print(f"  could not create env for {game_id!r}, skipping")
            continue

        agent = MyAgentCls(
            card_id="local-dev",
            game_id=game_id,
            agent_name=f"MyAgent.viz.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["local-dev-viz"],
        )
        agent.main()

        grids = [g for frame in agent.frames for g in frame.frame]
        if not grids:
            print("  no frames recorded, skipping")
            continue

        fps = args.fps or fps_by_id.get(game_id, 8)
        out_path = out_dir / f"{game_id}.gif"
        save_gif([np.asarray(g) for g in grids], out_path, fps=fps, scale=args.scale)

        final = agent.frames[-1]
        print(f"  → {len(grids)} frames @ {fps} fps, state={final.state}, "
              f"levels_completed={final.levels_completed} → {out_path}")

    print(f"\nDone. GIFs written to {out_dir}/")


if __name__ == "__main__":
    main()
