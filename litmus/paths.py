"""
paths.py

Where the harness reads and writes, relative to the repository root. Every
module takes its locations from here, so moving a folder is a one-line change.

    tasks/            benchmark task files (checked in)
    tasks/snapshots/      recorded REST market scenarios, replayed offline
    tasks/mcp_snapshots/  recorded CMC MCP scenarios, replayed offline
    runs/             agent trajectories and benchmark runs (mostly gitignored)
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
TASKS_DIR = ROOT / "tasks"
SNAPSHOT_ROOT = TASKS_DIR / "snapshots"
MCP_SNAPSHOT_ROOT = TASKS_DIR / "mcp_snapshots"
RUNS_DIR = ROOT / "runs"
BENCHMARK_RUNS_DIR = RUNS_DIR / "benchmark"
