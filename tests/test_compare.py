"""compare.py and --fail-under: diffing two saved runs, and the CI exit codes.
Runs on the published results in results/, so the numbers checked here are
the ones the docs quote. No model calls, no keys."""

import pytest

from litmus import compare, evaluate
from litmus.score import fisher_exact_p, wilson_interval

RESULTS = evaluate.ROOT / "results"
O4_V1 = str(RESULTS / "2026-09-28_mcp_k3_o4-mini")
O4_V21 = str(RESULTS / "2026-09-28_prompt_v2.1" / "o4-mini")
O4_PATCH = str(RESULTS / "2026-09-28_eur_patch_experiment_v1" / "o4-mini")
FLASH = str(RESULTS / "2026-09-27_mcp_k3_flash-lite")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def test_wilson_interval_matches_known_values():
    low, high = wilson_interval(16, 24)
    assert (round(low, 3), round(high, 3)) == (0.467, 0.820)
    assert wilson_interval(0, 3)[0] == 0.0 and wilson_interval(3, 3)[1] == 1.0  # stays inside [0, 1]
    assert wilson_interval(0, 0) is None


def test_fisher_exact_p_matches_known_values():
    assert fisher_exact_p(0, 3, 3, 3) == pytest.approx(0.1)  # 1/20 each tail
    assert fisher_exact_p(17, 24, 17, 24) == pytest.approx(1.0)
    assert fisher_exact_p(17, 24, 15, 24) == pytest.approx(fisher_exact_p(15, 24, 17, 24))  # symmetric


# --------------------------------------------------------------------------
# Pairing and diffing
# --------------------------------------------------------------------------

def test_pair_tasks_explicit_pairs_first_then_shared_ids_in_a_order():
    pairs, only_a, only_b = compare.pair_tasks(["x", "y", "z"], ["z", "y2", "x"], [("y", "y2")])
    assert pairs == [("y", "y2"), ("x", "x"), ("z", "z")]
    assert (only_a, only_b) == ([], [])
    pairs, only_a, only_b = compare.pair_tasks(["x", "y"], ["y", "w"])
    assert (pairs, only_a, only_b) == ([("y", "y")], ["x"], ["w"])


def test_prompt_v1_vs_v21_on_o4_mini():
    (a_config, a), (b_config, b) = compare.load_run(O4_V1), compare.load_run(O4_V21)
    pairs, _, _ = compare.pair_tasks(a_config["task_ids"], b_config["task_ids"])
    result = compare.compare(a, b, pairs)
    changes = {r["a_id"]: r["change"] for r in result["rows"]}
    assert changes["mcp_stale_global"] == "worse" and changes["mcp_dominance_history"] == "better"
    assert list(changes.values()).count("better") == 2 and list(changes.values()).count("worse") == 3
    stale = next(r for r in result["rows"] if r["a_id"] == "mcp_stale_global")
    assert stale["checks"] == ["required_tools 0->2"]
    o = result["overall"]
    assert (o["a_passed"], o["a_n"], o["b_passed"], o["b_n"]) == (17, 24, 15, 24)
    assert o["regressed"] and o["p_value"] > 0.05  # lower, but not distinguishable from noise


def test_prints_the_table_the_intervals_and_the_noise_verdict(capsys):
    assert evaluate.main(["--compare", O4_V1, O4_V21]) == 0
    out = capsys.readouterr().out
    assert "mcp_stale_global         3/3    1/3  - worse" in out
    assert "2 better, 3 worse, 3 same" in out
    assert "A 17/24 = 71% (95% CI 51%-85%)" in out
    assert "not distinguishable from noise" in out
    assert "the agent prompt changed (v1 -> v2.1)" in out


def test_fail_on_regression_exits_1_only_when_b_is_lower(capsys):
    assert evaluate.main(["--compare", O4_V1, O4_V21, "--fail-on-regression"]) == 1
    assert "REGRESSION" in capsys.readouterr().out
    assert evaluate.main(["--compare", O4_V21, O4_V1, "--fail-on-regression"]) == 0
    assert evaluate.main(["--compare", O4_V1, O4_V1, "--fail-on-regression"]) == 0  # equal is not a regression


def test_tolerance_lets_ci_ignore_a_drop_the_size_of_noise(capsys):
    """o4-mini v1 -> v2.1 is 17 -> 15 of 24 (p = 0.76): a 2-trial drop."""
    assert evaluate.main(["--compare", O4_V1, O4_V21, "--fail-on-regression", "--tolerance", "1"]) == 1
    assert "REGRESSION: B passed 2.0 trial(s) fewer than A's rate predicts (--tolerance 1)" in capsys.readouterr().out
    assert evaluate.main(["--compare", O4_V1, O4_V21, "--fail-on-regression", "--tolerance", "2"]) == 0
    assert "within --tolerance 2, not a regression" in capsys.readouterr().out


def test_shortfall_is_scaled_to_b_s_trial_count():
    a = {"by_task": {"t": {"passed": 2, "scored": 3, "failures": {}}}}
    b = {"by_task": {"t": {"passed": 4, "scored": 6, "failures": {}}}}  # same rate, twice the trials
    o = compare.compare(a, b, [("t", "t")])["overall"]
    assert o["shortfall"] == pytest.approx(0) and not o["regressed"]


def test_pair_matches_a_description_patch_variant(capsys):
    """The CMC workflow: the same question with a patched tool description,
    under another task id."""
    assert evaluate.main(["--compare", O4_V1, O4_PATCH,
                          "--pair", "mcp_eur_threshold=mcp_eur_threshold_eurc_hint"]) == 0
    out = capsys.readouterr().out
    assert "mcp_eur_threshold -> mcp_eur_threshold_eurc_hint    0/3    2/3  + better" in out
    assert "required_tools 3->0" in out  # it now fetched the euro rate
    assert "only in B (not compared): mcp_eur_threshold_usd_note" in out


@pytest.mark.parametrize("argv, message", [
    (["--compare", O4_V1, O4_V21, "--pair", "no-equals-sign"], "A_TASK=B_TASK"),
    (["--compare", O4_V1, O4_PATCH, "--pair", "nope=mcp_eur_threshold_eurc_hint"], "not a task in run A"),
    (["--compare", O4_V1, O4_PATCH, "--pair", "mcp_eur_threshold=nope"], "not a task in run B"),
    (["--compare", O4_V1, "no_such_run"], "no run 'no_such_run'"),
    (["--compare", FLASH, str(RESULTS / "2026-09-27_rest_k3_flash-lite")], "share no task ids"),
    (["--compare", O4_V1, O4_V21, "--fail-under", "0.5"], "can't be combined"),
    (["--pair", "a=b"], "only work with --compare"),
    (["--report", FLASH, "--fail-on-regression"], "only work with --compare"),
    (["--report", FLASH, "--fail-under", "70"], "between 0 and 1"),
    (["--compare", O4_V1, O4_V21, "--tolerance", "2"], "goes with --fail-on-regression"),
    (["--compare", O4_V1, O4_V21, "--fail-on-regression", "--tolerance", "-1"], "goes with --fail-on-regression"),
    (["--report", FLASH, "--tolerance", "2"], "only work with --compare"),
])
def test_bad_compare_arguments_are_usage_errors(argv, message, capsys):
    with pytest.raises(SystemExit) as e:
        evaluate.main(argv)
    assert e.value.code == 2
    assert message in capsys.readouterr().err


# --------------------------------------------------------------------------
# --fail-under and the interval in the report
# --------------------------------------------------------------------------

def test_report_prints_the_overall_confidence_interval(capsys):
    assert evaluate.main(["--report", FLASH]) == 0
    assert "overall pass rate 16/24: 95% CI 47%-82% (Wilson)" in capsys.readouterr().out


def test_fail_under_on_a_saved_run(capsys):
    assert evaluate.main(["--report", FLASH, "--fail-under", "0.6"]) == 0
    assert evaluate.main(["--report", FLASH, "--fail-under", "0.7"]) == 1
    assert "BELOW FLOOR: overall pass rate 0.67 < --fail-under 0.70" in capsys.readouterr().out


def test_a_run_with_nothing_scored_is_below_any_floor(capsys):
    assert evaluate.below_floor({"overall": None, "scored_trials": 0}, 0.0)
    assert not evaluate.below_floor({"overall": None}, None)


def test_report_counts_what_the_strict_trap_rule_costs(capsys):
    """gpt-6-sol's correct LONG that cited the whole-market RSI only to
    dismiss it: failed, and the report says so with the lenient count."""
    assert evaluate.main(["--report", str(RESULTS / "2026-09-28_mcp_k3_gpt-6-sol")]) == 0
    out = capsys.readouterr().out
    assert "strict trap rule: 1 trial(s) failed only the trap check; without it they would pass (19/24)" in out
    assert evaluate.main(["--report", FLASH]) == 0
    assert "strict trap rule" not in capsys.readouterr().out  # printed only when it cost something
