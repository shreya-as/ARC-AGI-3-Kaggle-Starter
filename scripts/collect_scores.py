"""Run local ARC-AGI-3 games and save a compact score report."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default=None, help="Optional game id or comma list.")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--trials", type=int, default=1, help="Run repeated trials and keep the best run per game.")
    parser.add_argument("--out", default="local_scores_latest.json")
    parser.add_argument("--generalization", action="store_true")
    parser.add_argument("--coarse-hash", action="store_true")
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--no-observed-model", action="store_true")
    args = parser.parse_args()

    trial_summaries = []
    best_by_game: dict[str, dict] = {}
    last_payload: dict | None = None

    for trial in range(max(1, args.trials)):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "replay_json.py"),
            "--max-steps",
            str(args.max_steps),
        ]
        if args.game:
            command.extend(["--game", args.game])

        command.extend(["--seed", str(trial)])
        for flag in ("generalization", "coarse_hash", "scores_only", "no_observed_model"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        output = subprocess.check_output(command, text=True)
        trial_payload = json.loads(
            [line for line in output.splitlines() if line.startswith("{")][-1]
        )
        last_payload = trial_payload
        trial_summaries.append(
            {
                "trial": trial + 1,
                **trial_payload.get("summary", {}),
            }
        )

        for run in trial_payload.get("runs", []):
            game_id = run.get("gameId")
            if not game_id:
                continue
            existing = best_by_game.get(game_id)
            if existing is None:
                best_by_game[game_id] = run
                continue
            current_rank = (
                float(run.get("arcScore") or 0.0),
                int(run.get("levelsCompleted") or 0),
                -int(run.get("actions") or 0),
            )
            existing_rank = (
                float(existing.get("arcScore") or 0.0),
                int(existing.get("levelsCompleted") or 0),
                -int(existing.get("actions") or 0),
            )
            if current_rank > existing_rank:
                best_by_game[game_id] = run

    if args.trials > 1:
        runs = list(best_by_game.values())
        total_levels = sum(run.get("winLevels", 0) for run in runs)
        completed_levels = sum(run.get("levelsCompleted", 0) for run in runs)
        scored_runs = [run for run in runs if isinstance(run.get("arcScore"), (int, float))]
        arc_score = (
            round(sum(float(run["arcScore"]) for run in scored_runs) / len(scored_runs), 6)
            if scored_runs
            else 0.0
        )
        payload = {
            "runs": runs,
            "summary": {
                "games": len(runs),
                "actions": sum(run.get("actions", 0) for run in runs),
                "reward": round(sum(run.get("reward", 0.0) for run in runs), 4),
                "levelsCompleted": completed_levels,
                "totalLevels": total_levels,
                "score": arc_score,
                "arcScore": arc_score,
                "levelCompletionScore": round((completed_levels / total_levels) * 100, 2)
                if total_levels
                else 0,
            },
            "trialSummaries": trial_summaries,
        }
    else:
        payload = last_payload or {"runs": [], "summary": {}}

    payload = {
        "generatedAt": dt.datetime.now(dt.UTC).isoformat(),
        "maxSteps": args.max_steps,
        "trials": max(1, args.trials),
        **payload,
    }

    trial_scores = [s["arcScore"] for s in trial_summaries
                    if isinstance(s.get("arcScore"), (int, float))]
    payload["evaluation"] = {
        "generalization": args.generalization,
        "observedModel": not args.no_observed_model,
        "stateHash": "coarse" if args.coarse_hash else "exact-change-coarse-exploration",
        "seeds": list(range(max(1, args.trials))),
        "meanArcScore": sum(trial_scores) / len(trial_scores) if trial_scores else None,
        "minArcScore": min(trial_scores) if trial_scores else None,
        "maxArcScore": max(trial_scores) if trial_scores else None,
        "summaryAggregation": "best-per-game" if args.trials > 1 else "single-trial",
    }
    print("Mean trial ARC:", payload["evaluation"]["meanArcScore"])
    print("Table aggregation:", payload["evaluation"]["summaryAggregation"])
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("GAME     ARC       LVL%    LEVELS       ACTIONS   REWARD    STATES   ARCHIVE   STATE")
    print("-" * 90)
    for run in payload["runs"]:
        arc_score = run.get("arcScore")
        arc_text = f"{arc_score:>7.3f}" if isinstance(arc_score, (int, float)) else "      -"
        print(
            f"{run['gameId']:<8} "
            f"{arc_text}  "
            f"{run.get('levelCompletionScore', run['score']):>6.2f}%  "
            f"{run['levelsCompleted']:>2}/{run['winLevels']:<8} "
            f"{run['actions']:>6}   "
            f"{run.get('reward', 0):>7.2f}   "
            f"{run.get('knownStates', 0):>6}   "
            f"{run.get('archiveSize', 0):>7}   "
            f"{run['state']}"
        )

    summary = payload["summary"]
    arc_score = summary.get("arcScore")
    arc_text = f"{arc_score:>7.3f}" if isinstance(arc_score, (int, float)) else "      -"
    print("-" * 90)
    print(
        f"TOTAL    "
        f"{arc_text}  "
        f"{summary.get('levelCompletionScore', summary['score']):>6.2f}%  "
        f"{summary['levelsCompleted']:>2}/{summary['totalLevels']:<8} "
        f"{summary['actions']:>6}   "
        f"{summary.get('reward', 0):>7.2f}"
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
