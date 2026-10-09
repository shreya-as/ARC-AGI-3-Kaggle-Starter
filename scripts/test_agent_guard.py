"""Static guard for an agent file (AGENT_PLAN.md §2.2, §5.1 P1/P4, §5.7).

Fails if the agent could only score locally or could break on Kaggle:
  - mentions a public game id (per-game branches, opening books, comments
    that leak tuning to specific games);
  - touches engine internals (`_game`, deepcopy of the env): there is no
    engine behind the Kaggle gateway;
  - imports anything beyond the standard library, numpy, arcengine and the
    framework's `agents` package (the rerun has no internet and no pip);
  - does not define `MyAgent`, or does not compile.

    .venv/bin/python scripts/test_agent_guard.py agent/my_agent_v4.py
    .venv/bin/python scripts/test_agent_guard.py            # checks agent/my_agent.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_IMPORTS = {
    "__future__", "collections", "logging", "os", "random", "time", "typing", "zlib",
    "math", "itertools", "heapq", "functools", "dataclasses",
    "numpy", "arcengine", "agents",
}
ENGINE_PATTERNS = [r"\._game\b", r"deepcopy\s*\(", r"\bLocalEnvironmentWrapper\b", r"\bARCBaseGame\b"]


def public_game_ids() -> list[str]:
    """4-character prefixes of every game shipped in environment_files/."""
    env = ROOT / "environment_files"
    ids = {p.name[:4] for p in env.iterdir() if p.is_dir() and len(p.name) >= 4} if env.exists() else set()
    return sorted(ids)


def check(path: Path) -> list[str]:
    src = path.read_text()
    problems = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"does not compile: {e}"]

    for gid in public_game_ids():
        for n, line in enumerate(src.splitlines(), 1):
            if re.search(rf"\b{re.escape(gid)}\b", line):
                problems.append(f"line {n}: mentions public game id '{gid}'")

    for pat in ENGINE_PATTERNS:
        for n, line in enumerate(src.splitlines(), 1):
            if re.search(pat, line):
                problems.append(f"line {n}: touches engine internals ({pat})")

    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if name not in ALLOWED_IMPORTS:
                problems.append(f"line {node.lineno}: imports '{name}' (not available on the Kaggle rerun)")

    if not any(isinstance(n, ast.ClassDef) and n.name == "MyAgent" for n in tree.body):
        problems.append("no top-level class named MyAgent")
    return problems


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "agent" / "my_agent.py"
    problems = check(path)
    if problems:
        print(f"FAIL {path}")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(f"OK   {path}  ({len(public_game_ids())} public game ids checked)")


if __name__ == "__main__":
    main()
