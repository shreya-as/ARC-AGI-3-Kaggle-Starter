"""Run an agent the way the Kaggle rerun does (AGENT_PLAN.md Appendix A).

Starts a fresh competition-mode gateway on a free local port (one scorecard per
server, as on Kaggle), then runs the real framework Swarm over HTTP with one
thread per game, and prints levels, actions and the ARC score.

    .venv/bin/python scripts/kaggle_sim.py                                  # agent/my_agent.py, 18 dev games
    .venv/bin/python scripts/kaggle_sim.py --agent agent/my_agent_v1.py
    .venv/bin/python scripts/kaggle_sim.py --agent agent/my_agent_v1.py --games vc33,ft09 --max-actions 300
    .venv/bin/python scripts/kaggle_sim.py --agent agent/my_agent_v1.py --seed 1

The 7 sealed holdout games are skipped unless --holdout (dev + holdout) or
--holdout-only is given (§3.1, §6.2). Every run that plays a holdout game is
appended to HOLDOUT_RUNS.md, as §6.2 asks ("log every holdout run").
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = {"dc22", "g50t", "lp85", "s5i5", "sc25", "sk48", "tr87"}
API_KEY = "test-key-123"

SERVER_CODE = """
import logging, sys
import arc_agi
from arc_agi import OperationMode
logging.basicConfig(level=logging.WARNING)
arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
arc.listen_and_serve(host="127.0.0.1", port=int(sys.argv[1]), competition_mode=True)
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="agent/my_agent.py", help="agent file defining MyAgent")
    p.add_argument("--games", default="", help="comma-separated game id prefixes (default: all dev games)")
    p.add_argument("--max-actions", type=int, help="sets AGENT_MAX_ACTIONS (agent default otherwise)")
    p.add_argument("--seed", type=int, default=0, help="sets SEED_OFFSET")
    p.add_argument("--holdout", action="store_true", help="also run the sealed holdout games")
    p.add_argument("--holdout-only", action="store_true", help="run only the sealed holdout games (E12)")
    p.add_argument("--verbose", action="store_true", help="show INFO logs (level-ups, per-game summary)")
    args = p.parse_args()

    os.chdir(ROOT)
    port = free_port()
    rec_dir = tempfile.mkdtemp(prefix="arc_sim_rec_")
    os.environ.update({
        "OPERATION_MODE": "online", "ARC_BASE_URL": f"http://127.0.0.1:{port}/",
        "ARC_API_KEY": API_KEY, "ENVIRONMENTS_DIR": "",
        # Swarm records every frame; keep that out of the repo.
        "RECORDINGS_DIR": rec_dir,
        "SEED_OFFSET": str(args.seed),
    })
    if args.max_actions is not None:
        os.environ["AGENT_MAX_ACTIONS"] = str(args.max_actions)

    server = subprocess.Popen([sys.executable, "-c", SERVER_CODE, str(port)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
        logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)
        if not args.verbose:
            # arc_agi adds its own handlers and logs a harmless traceback per
            # non-game folder it scans; keep only real errors.
            logging.disable(logging.WARNING)

        import requests

        for _ in range(100):  # wait for the server
            try:
                games = requests.get(f"http://127.0.0.1:{port}/api/games",
                                     headers={"X-API-Key": API_KEY}, timeout=2).json()
                break
            except requests.RequestException:
                time.sleep(0.2)
        else:
            raise SystemExit("simulator server did not start")

        # The agent modules read AGENT_MAX_ACTIONS / SEED_OFFSET at import, so
        # import only after the environment is set.
        import agents
        from agents import Swarm

        spec = importlib.util.spec_from_file_location("user_agent", args.agent)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        agents.AVAILABLE_AGENTS["myagent"] = mod.MyAgent

        ids = [g["game_id"] for g in games]
        if args.games:
            ids = [g for g in ids if any(g.startswith(k) for k in args.games.split(","))]
        if args.holdout_only:
            ids = [g for g in ids if g[:4] in HOLDOUT]
        elif not args.holdout:
            ids = [g for g in ids if g[:4] not in HOLDOUT]
        print(f"agent {args.agent}  games {len(ids)}  seed {args.seed}  "
              f"max actions {mod.MyAgent.MAX_ACTIONS}")

        t0 = time.time()
        swarm = Swarm("myagent", f"http://127.0.0.1:{port}", ids)
        card = swarm.main()
        cleared = 0
        for a in sorted(swarm.agents, key=lambda a: a.game_id):
            f = a.frames[-1]
            cleared += f.levels_completed > 0
            ups = getattr(a, "level_ups", None)
            print(f"{a.game_id[:4]:6} levels {f.levels_completed}/{f.win_levels}  "
                  f"actions {a.action_counter:5}  final {f.state.name:12}"
                  + (f"  level-ups at {ups}" if ups else ""))
        print(f"level 1 cleared in {cleared}/{len(ids)} games")
        score = card.model_dump().get("score") if card is not None else None
        if score is not None:
            print("ARC SCORE", score)
        print(f"elapsed {time.time() - t0:.1f}s")
        played = sorted(g[:4] for g in ids if g[:4] in HOLDOUT)
        if played:
            with open(ROOT / "HOLDOUT_RUNS.md", "a") as log:
                log.write(f"- {time.strftime('%Y-%m-%d %H:%M')}  agent {args.agent}  seed {args.seed}  "
                          f"max actions {mod.MyAgent.MAX_ACTIONS}  games {len(ids)} "
                          f"(holdout {', '.join(played)})  L1 {cleared}/{len(ids)}  ARC {score}\n")
            print("logged to HOLDOUT_RUNS.md")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
