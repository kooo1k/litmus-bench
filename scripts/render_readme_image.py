"""
render_readme_image.py

Render the docs' terminal images from real, saved benchmark trials: no
model call, no key. Each trial is played back through the CLI's own display
(cli.replay), so an image is exactly what `litmus --replay ...` shows, and
the command at the top of each image reproduces it.

    python -m scripts.render_readme_image        # all images below -> docs/images/

    verdict.svg    flash-lite assumed euro parity, went LONG; the risk
                   governor approved; the benchmark failed it (the headline)
    fix.svg        o4-mini with CMC's EURC hint (prompt v2.1): fetched EURC,
                   converted to euros, went SHORT; the benchmark passed it
    freshness.svg  gpt-6-sol refused hours-old Fear & Greed data under a
                   "last hour only" rule; the benchmark passed it
"""

import sys

from rich.console import Console

from litmus import cli, cli_view, evaluate, paths

IMAGES = {
    "verdict.svg": cli.DEFAULT_REPLAY,
    "fix.svg": ("results/2026-09-28_eur_patch_experiment_v2.1/o4-mini", "mcp_eur_threshold_eurc_hint", 1),
    "freshness.svg": ("results/2026-09-28_mcp_k3_gpt-6-sol", "mcp_stale_global", 1),
}
OUT_DIR = paths.ROOT / "docs" / "images"
WIDTH = 104


def render(run, task_id, trial, out):
    view = cli_view.LiveView()
    view.console = Console(record=True, width=WIDTH, force_terminal=True, color_system="truecolor")
    command = "litmus --replay" if (run, task_id, trial) == cli.DEFAULT_REPLAY else f"litmus --replay {run} {task_id} {trial}"
    view.console.print(f"[bold cyan]$[/] {command}\n")
    cli.replay(view, run, task_id, trial)
    out.parent.mkdir(parents=True, exist_ok=True)
    view.console.save_svg(str(out), title="litmus")
    print(f"wrote {evaluate.display_path(out)}", file=sys.stderr)


def main(names=None):
    for name in names or IMAGES:
        render(*IMAGES[name], OUT_DIR / name)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
