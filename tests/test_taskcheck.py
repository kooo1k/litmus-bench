"""taskcheck.py and `litmus-bench --check`: every shipped task and the worked
examples pass, and each kind of authoring mistake is named. No model calls."""

import copy
import json

import pytest

from litmus import cli, evaluate, taskcheck
from conftest import ROOT

EXAMPLES = ROOT / "docs" / "examples" / "tasks.jsonl"
SHIPPED = evaluate.load_tasks() + evaluate.load_tasks(evaluate.MCP_TASKS_PATH)
REST, MCP = evaluate.load_tasks(EXAMPLES)


@pytest.mark.parametrize("task", SHIPPED + [REST, MCP], ids=lambda t: t["id"])
def test_every_shipped_and_example_task_is_well_formed(task):
    assert taskcheck.problems(task) == []


def broken(task, edit):
    task = copy.deepcopy(task)
    edit(task)
    return " | ".join(taskcheck.problems(task))


@pytest.mark.parametrize("base, edit, expected", [
    (REST, lambda t: t.pop("rationale"), "missing field"),
    (REST, lambda t: t.update(extra=1), "unknown field"),
    (REST, lambda t: t.update(toolset="graphql"), "toolset must be"),
    (REST, lambda t: t["expected"].update(acceptable_actions=["BUY"]), "unknown action"),
    (REST, lambda t: t["expected"].update(acceptable_actions=[]), "no answer could pass"),
    (REST, lambda t: t["expected"].update(acceptable_symbols=["DOGE"]), "can't be traded"),
    (REST, lambda t: t["expected"].update(required_tools=["get_price BTC"]), "must look like"),
    (REST, lambda t: t["expected"].update(required_tools=["get_crypto_quotes_latest(BTC)"]), "not in the rest toolset"),
    (REST, lambda t: t["expected"].update(required_tools=["get_fear_greed(BTC)"]), "takes no symbol"),
    (REST, lambda t: t["expected"].update(required_tools=["get_price()"]), "needs a symbol"),
    (REST, lambda t: t["expected"].update(position_size_pct=0.5), "position_size_pct"),
    (REST, lambda t: t["faults"]["get_price"].update(type="meltdown"), "needs a type"),
    (REST, lambda t: t["faults"]["get_price"].update(times=0), "tools don't build"),
    (REST, lambda t: t.update(snapshot="tasks/snapshots/no_such_day"), "tools don't build"),
    (MCP, lambda t: t["expected"]["traps"][0].pop("reason"), "needs tool, field and reason"),
    (MCP, lambda t: t["expected"]["traps"][0].update(tool="get_price"), "not in the mcp toolset"),
    (MCP, lambda t: t.update(snapshot="tasks/snapshots/2026-09-25_live"), "not an MCP scenario"),
])
def test_each_mistake_is_named(base, edit, expected):
    assert expected in broken(base, edit)


def test_duplicate_ids_are_reported():
    assert taskcheck.check_file([REST, REST]) == {REST["id"]: ["duplicate id"]}


def test_check_exit_codes(tmp_path, capsys):
    assert evaluate.main(["--check", str(EXAMPLES)]) == 0
    assert "2/2 task(s) OK" in capsys.readouterr().out
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**REST, "expected": {**REST["expected"], "acceptable_actions": ["BUY"]}}) + "\n",
                   encoding="utf-8")
    assert evaluate.main(["--check", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "example_stale_quote:" in out and "unknown action" in out and "0/1 task(s) OK" in out
    bad.write_text("{not json\n", encoding="utf-8")
    assert evaluate.main(["--check", str(bad)]) == 1
    assert evaluate.main(["--check", str(tmp_path / "missing.jsonl")]) == 1


def test_litmus_runs_tasks_from_your_own_file(capsys):
    assert cli.main(["--tasks", str(EXAMPLES), "--list-tasks", "--plain"]) == 0
    out = capsys.readouterr().out
    assert "example_stale_quote" in out and "example_dominance_threshold" in out and "clean_01" not in out
    with pytest.raises(SystemExit):
        cli.main(["--tasks", str(EXAMPLES), "Should I trade?"])  # --tasks needs --task or --list-tasks


# --------------------------------------------------------------------------
# Experiments: description_patches (tasks/experiments/)
# --------------------------------------------------------------------------

EXPERIMENT = ROOT / "tasks" / "experiments" / "eur_description_patch.jsonl"


def test_experiment_tasks_are_well_formed_and_differ_only_in_what_the_model_reads():
    base = next(t for t in evaluate.load_tasks(evaluate.MCP_TASKS_PATH) if t["id"] == "mcp_eur_threshold")
    for task in evaluate.load_tasks(EXPERIMENT):
        assert taskcheck.problems(task) == []
        same = {k: v for k, v in task.items() if k not in ("id", "category", "rationale", "description_patches")}
        assert same == {k: v for k, v in base.items() if k not in ("id", "category", "rationale")}
        assert evaluate.tasks_fingerprint([task]) != evaluate.tasks_fingerprint([base])


def test_patch_is_appended_to_cmc_text_and_nothing_else_changes():
    from litmus import mcp_tools
    task = evaluate.load_tasks(EXPERIMENT)[1]
    base = {d["function"]["name"]: d for d in evaluate.tool_definitions({**task, "description_patches": None})}
    patched = {d["function"]["name"]: d for d in evaluate.tool_definitions(task)}
    quote = "get_crypto_quotes_latest"
    assert patched[quote]["function"]["description"] == (
        base[quote]["function"]["description"] + " " + task["description_patches"][quote])
    assert all(patched[n] == base[n] for n in base if n != quote)
    with pytest.raises(ValueError, match="isn't shown"):
        mcp_tools.patch_descriptions(list(base.values()), {"no_such_tool": "x"})


@pytest.mark.parametrize("patches, expected", [
    ({"get_price": "x"}, "not in the mcp toolset"),
    ({"get_crypto_quotes_latest": ""}, "must be {tool: sentence}"),
])
def test_bad_description_patches_are_named(patches, expected):
    assert expected in broken(MCP, lambda t: t.update(description_patches=patches))


def test_description_patches_are_mcp_only():
    assert "only apply to MCP tasks" in broken(REST, lambda t: t.update(description_patches={"get_price": "x"}))
