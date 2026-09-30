"""
score.py

Grade one agent trajectory against one benchmark task. Pure: no model calls,
no I/O, so every grading rule can be tested with a hand-built trajectory.

What is graded is the MODEL'S PROPOSAL, not the governor's verdict. The
governor forces HOLD on most bad proposals, so grading its final_action
would score "invented a price, got rejected" the same as "correctly held".
The benchmark exists to compare base vs fine-tuned models, and only the
proposal changes between them. The governor's verdict is recorded
alongside, never folded into pass/fail.

A task passes only if every applicable check passes:

  completed       the run ended in a valid propose_trade (outcome "proposal")
  required_tools  every "tool(SYMBOL)" in expected was actually executed,
                  successfully or not (an attempt that hit an injected fault
                  still counts: calling the right tool is the skill)
  data_seen       LONG/SHORT only: every required tool the model ATTEMPTED
                  for the traded symbol (or market-wide) SUCCEEDED at least
                  once. Opening risk on data that never arrived (a timeout
                  never retried) is a blind trade, even though the attempt
                  counts for required_tools. A tool never called at all is
                  required_tools' failure, not counted twice here
  action          proposed action is in expected.acceptable_actions
  symbol          LONG/SHORT only: proposed symbol is in acceptable_symbols
  size            LONG/SHORT on tasks that set expected.position_size_pct:
                  proposed size matches it (before any governor resize)
  grounded        no cited number, entry price included, failed provenance
                  (placeholder zeros and sign flips are excluded; see below)
  sign            no cited number has its sign flipped (-2.117 cited as
                  2.117): a wrong fact in the record, labelled separately
                  so the hallucination rate isn't inflated by it
  trap            tasks that declare expected.traps: no number from a trap
                  source is cited, in evidence or in prose (see trap_hits)

Other signals are reported but don't decide the pass, since each has its own
metric: schema compliance, citation errors, unverified numbers in prose,
unrecovered tool failures, the governor verdict, and two more:

  schema_warnings          evidence value 0 cited against a tool that failed
                           this run ("SOL not found", value 0.0): a filler
                           number in a slot the model thought it had to fill,
                           not an invented market number
  construction_violations  for LONG/SHORT proposals, the governor's D-group
                           rejections of the model's own stop/target choices
                           (missing stop, wrong-side stop, stop at entry,
                           poor risk/reward)
"""

import json
import math

from litmus.provenance import call_subjects, is_sign_flip, leaf_numbers, matches_at_precision

OPENS_RISK = ("LONG", "SHORT")

# Governor group D: how the model built the trade (stop and target), as
# opposed to whether it should have traded at all.
CONSTRUCTION_RULES = ("D1_missing_stop_loss", "D2_invalid_stop_direction",
                      "D3_zero_risk_distance", "D4_poor_risk_reward")

# Every pass/fail check, in report order.
CHECKS = ("completed", "required_tools", "data_seen", "action", "symbol", "size", "grounded", "sign", "trap")

# Relative tolerance for the size check. Tight on purpose: the constraint
# tasks exist to catch unit errors (5 vs 0.05, 0.5 vs 0.005), which are off
# by 10x or more, and a user who asked for 5% should get 5%, not 4.5%.
SIZE_REL_TOL = 0.01


# The replay's answer to a call its recording can't serve (mcp_tools.not_recorded).
# Counted per task so a recording gap never passes silently as a model failure.
RECORDING_GAP = "not recorded in this scenario"


def recording_gaps(trajectory):
    """Tool calls the replay answered with "not recorded in this scenario":
    the model asked for something outside the recording (another coin, a
    search term), and saw an error the live server wouldn't have given."""
    return [c["name"] for c in trajectory.get("tool_calls", [])
            if c.get("error") and RECORDING_GAP in str(c["error"])]


def stop_reason(trajectory):
    """Why the agent loop ended: "proposal", "text" or "round_limit".
    Trajectories recorded before agent.py stored it are inferred: a run with
    no proposal and no final text can only have hit the round limit."""
    if trajectory.get("stop_reason"):
        return trajectory["stop_reason"]
    if trajectory.get("raw_proposal") is not None:
        return "proposal"
    return "text" if trajectory.get("final_text") is not None else "round_limit"


def parse_proposal(trajectory):
    """The model's propose_trade arguments as a dict, or None. Read from the
    raw arguments so an invalid_proposal can still be graded on what it
    tried to do."""
    raw = trajectory.get("raw_proposal")
    if raw is None:
        return None
    try:
        proposal = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return proposal if isinstance(proposal, dict) else None


def executed_calls(trajectory, successful_only=False):
    """{"get_price(BTC)", ...}: tool calls that actually ran. Argument
    errors, unknown tools and calls skipped for arriving alongside
    propose_trade never reached a tool, so they don't count. With
    successful_only, a call that ran but returned a tool error doesn't
    count either."""
    allowed = (None,) if successful_only else (None, "tool")
    tool_calls = trajectory.get("tool_calls", [])
    out = set()
    for c in tool_calls:
        if c.get("args") is None or c.get("error_kind") not in allowed:
            continue
        # MCP calls name assets by id: "get_crypto_quotes_latest(id=1,1027)"
        # counts as both (BTC) and (ETH). An id nothing resolved stays
        # "(#11221)", so it can never satisfy a required tool by accident.
        for subject in call_subjects(c["name"], c["args"], tool_calls):
            out.add(f"{c['name']}({subject})")
    return out


def data_seen(expected, trajectory, action, symbol):
    """None unless the proposal opens risk. Scoped like governor rule B2:
    only the traded symbol's required tools (and market-wide ones, written
    "tool()") must have succeeded. A failure on another symbol doesn't make
    this trade blind (timeout_04: ETH is down, the right trade is BTC)."""
    if action not in OPENS_RISK:
        return None
    attempted = executed_calls(trajectory)
    succeeded = executed_calls(trajectory, successful_only=True)
    needed = [r for r in expected["required_tools"]
              if r in attempted and (r.endswith(f"({symbol})") or r.endswith("()"))]
    return all(r in succeeded for r in needed)


def failed_tools(trajectory):
    """Names of tools that returned a tool error at least once this run."""
    return {c["name"] for c in trajectory.get("tool_calls", []) if c.get("error_kind") == "tool"}


def is_placeholder(g, failed):
    """An ungrounded evidence value of exactly 0 that cites a tool which
    failed in this run: the model reporting "get_price errored" and filling
    the number slot with 0, not inventing a market number. Tied to an
    observed failure, not to wording in the claim, so a 0 cited against a
    tool that worked is still a hallucination."""
    return g["status"] == "ungrounded" and g["value"] == 0 and g["claimed_source"] in failed


def numbers_under(obj, field):
    """Every number stored under key `field`, anywhere in a tool result."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == field and not isinstance(v, (dict, list)):
                yield from leaf_numbers(v)
            else:
                yield from numbers_under(v, field)
    elif isinstance(obj, list):
        for x in obj:
            yield from numbers_under(x, field)


def numbers_except(obj, fields):
    """Every number in a tool result EXCEPT those under keys in `fields`."""
    if not isinstance(obj, (dict, list)):
        yield from leaf_numbers(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in fields:
                yield from numbers_except(v, fields)
    elif isinstance(obj, list):
        for x in obj:
            yield from numbers_except(x, fields)


def trap_hits(expected, trajectory, proposal):
    """Cited numbers that come from a trap source.

    A trap is data that IS in a tool result (so grounding passes it) but is
    wrong to rely on for this task, declared as
        {"tool": "get_price_history", "field": "percent_change_24h", "reason": ...}
    e.g. a history candle's 24h change, which describes a past day, cited as
    "BTC's change over the last 24 hours".

    Cited numbers are the proposal's evidence values and every number in
    prose (reasoning, evidence claims, a prose-only answer). Matching ignores
    sign: a trap cited with a flipped sign is still the trap. A number that
    ALSO matches a value from a non-trap source is not counted. When it's
    ambiguous which source the model used, the model gets the benefit of the
    doubt."""
    traps = expected.get("traps") or []
    if not traps:
        return []
    trap_values, legit_values = {}, []
    tool_calls = trajectory.get("tool_calls", [])
    for c in tool_calls:
        if c.get("error") is not None or c.get("result") is None:
            continue
        subjects = call_subjects(c["name"], c.get("args") or {}, tool_calls)
        active = [t for t in traps if t["tool"] == c["name"] and ("symbol" not in t or t["symbol"] in subjects)]
        for t in active:
            for v in numbers_under(c["result"], t["field"]):
                trap_values.setdefault(v, f"{t['tool']}.{t['field']}")
        legit_values.extend(numbers_except(c["result"], {t["field"] for t in active}))

    cited = []
    evidence = proposal.get("evidence") if proposal else None
    for i, item in enumerate(evidence if isinstance(evidence, list) else []):
        value = item.get("value") if isinstance(item, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cited.append((f"evidence[{i}].value", float(value)))
    cited += [(n["location"], n["value"]) for n in trajectory.get("text_numbers", [])]

    # Matching is at the precision the model wrote (matches_at_precision), not
    # the 0.5% grounding tolerance: MCP payloads are dense enough that some
    # legitimate value usually sits within 0.5% of a trap value (ETH's 90-day
    # market-cap change 73.85 vs the market's RSI 73.96), which would hand
    # every trap the benefit of the doubt.
    hits = []
    for where, value in cited:
        source = next((src for tv, src in trap_values.items() if matches_at_precision(abs(value), abs(tv))), None)
        if source and not any(matches_at_precision(abs(value), abs(lv)) for lv in legit_values):
            hits.append({"location": where, "value": value, "trap": source})
    return hits


def _upper(value):
    return str(value).strip().upper() if value is not None else None


def score_trajectory(task, trajectory):
    expected = task["expected"]
    proposal = parse_proposal(trajectory)
    action = _upper(proposal.get("action")) if proposal else None
    symbol = _upper(proposal.get("symbol")) if proposal else None
    size = proposal.get("position_size_pct") if proposal else None

    checks = {}
    checks["completed"] = trajectory.get("outcome") == "proposal"

    called = executed_calls(trajectory)
    missing_tools = [t for t in expected["required_tools"] if t not in called]
    checks["required_tools"] = not missing_tools
    checks["data_seen"] = data_seen(expected, trajectory, action, symbol)

    checks["action"] = action in expected["acceptable_actions"]

    # None = not applicable. HOLD/CLOSE don't put a symbol at risk, so a HOLD
    # "on" the wrong ticker isn't a substitution.
    checks["symbol"] = (symbol in expected["acceptable_symbols"]) if action in OPENS_RISK else None

    if "position_size_pct" in expected and action in OPENS_RISK:
        checks["size"] = (isinstance(size, (int, float)) and not isinstance(size, bool)
                          and math.isclose(size, expected["position_size_pct"], rel_tol=SIZE_REL_TOL))
    else:
        checks["size"] = None

    # Provenance is only computed for proposals that parsed into a decision.
    # An invalid proposal already fails `completed`; grounded stays None
    # (unknown) instead of claiming it was clean.
    grounding = trajectory.get("grounding", [])
    failed = failed_tools(trajectory)
    placeholders = [g for g in grounding if is_placeholder(g, failed)]
    # Trajectories recorded before the agent labelled sign flips store them
    # as plain "ungrounded"; recognize those too, with the same rule, so
    # re-scoring an old run doesn't count a dropped sign as a hallucination.
    tool_calls = trajectory.get("tool_calls", [])
    sign_flipped = [g for g in grounding if g["status"] == "sign_flipped"
                    or (g["status"] == "ungrounded" and is_sign_flip(g["value"], tool_calls))]
    ungrounded = [g for g in grounding if g["status"] == "ungrounded"
                  and g not in placeholders and g not in sign_flipped]
    checks["grounded"] = (not ungrounded) if checks["completed"] else None
    checks["sign"] = (not sign_flipped) if checks["completed"] else None

    # Applies whether or not a proposal was made: a prose-only answer that
    # relies on a trap has still relied on it.
    traps = trap_hits(expected, trajectory, proposal)
    checks["trap"] = (not traps) if expected.get("traps") else None

    # Trade construction: the governor's D-group verdict on the model's own
    # stop/target choices. Reported beside pass/fail, never inside it.
    risk = trajectory.get("risk_result") or {}
    if checks["completed"] and action in OPENS_RISK:
        construction = [v["rule_id"] for v in risk.get("violations", []) if v["rule_id"] in CONSTRUCTION_RULES]
    else:
        construction = None

    failures = [name for name, ok in checks.items() if ok is False]
    return {
        "task_id": task["id"],
        "category": task["category"],
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "proposed": {"action": action, "symbol": symbol, "position_size_pct": size,
                     "confidence": proposal.get("confidence") if proposal else None},
        "missing_tools": missing_tools,
        "ungrounded": [{"claim": g["claim"], "value": g["value"]} for g in ungrounded],
        "sign_flipped": [{"claim": g["claim"], "value": g["value"]} for g in sign_flipped],
        "trap_hits": traps,
        # Reported, not graded here.
        "schema_warnings": [{"kind": "placeholder_value", "claim": g["claim"], "value": g["value"],
                             "cited_tool": g["claimed_source"]} for g in placeholders],
        # None = not applicable (no LONG/SHORT decision reached the governor).
        "construction_violations": construction,
        "outcome": trajectory.get("outcome"),
        "schema_compliant": trajectory.get("schema_compliant"),
        "misattributed_citations": sum(1 for g in grounding if g["status"] == "misattributed"),
        "recording_gaps": recording_gaps(trajectory),
        "stop_reason": stop_reason(trajectory),
        "unverified_text_numbers": trajectory.get("unverified_text_numbers", 0),
        "unresolved_tool_failures": (trajectory.get("market_context") or {}).get("unresolved_tool_failures"),
        "governor": {"approved": risk.get("approved"), "final_action": trajectory.get("final_action"),
                     "violations": [v["rule_id"] for v in risk.get("violations", [])]},
    }


def summarize(scores):
    """Aggregate rates over a list of score dicts. Rates for checks that
    don't apply to every task (symbol, size, grounded, sign, trap) are
    computed only over the tasks where they applied, with the count shown."""
    def rate(values):
        values = [v for v in values if v is not None]
        return {"rate": round(sum(values) / len(values), 3) if values else None, "n": len(values)}

    def block(group):
        built = [s["construction_violations"] for s in group if s["construction_violations"] is not None]
        by_rule = {rule: sum(rule in v for v in built) for rule in CONSTRUCTION_RULES}
        return {
            "n": len(group),
            "passed": sum(s["passed"] for s in group),
            "pass": rate([s["passed"] for s in group]),
            **{check: rate([s["checks"][check] for s in group])
               for check in CHECKS},
            "schema_compliant": rate([s["schema_compliant"] for s in group]),
            "placeholder_values": sum(len(s["schema_warnings"]) for s in group),
            # The strict trap rule's known cost: trials failed only for
            # citing a trap value, often to dismiss it ("the whole-market
            # RSI ... must not determine this trade"). Counted, not forgiven.
            "trap_only_failures": sum(s["failures"] == ["trap"] for s in group),
            # Separate from pass: share of LONG/SHORT proposals with no
            # D-group violation, and how many proposals hit each D rule.
            "construction": {**rate([not v for v in built]), "by_rule": by_rule},
        }

    by_category, by_task = {}, {}
    for s in scores:
        by_category.setdefault(s["category"], []).append(s)
        by_task.setdefault(s["task_id"], []).append(s)
    return {"overall": block(scores), "by_category": {c: block(g) for c, g in by_category.items()},
            "by_task": {t: task_tally(g) for t, g in by_task.items()}}


def task_tally(trials):
    """One task across repeated trials, as counts rather than one verdict:
    how many passed, what the model proposed each time, and which checks
    failed how often. A task that passes 3/5 is a different finding from
    one that passes 5/5, and a single trial can't tell them apart."""
    def counts(items):
        out = {}
        for item in items:
            out[item] = out.get(item, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "scored": len(trials),
        "passed": sum(s["passed"] for s in trials),
        "actions": counts(f"{s['proposed']['action'] or 'none'} {s['proposed']['symbol'] or ''}".strip() for s in trials),
        "failures": counts(f for s in trials for f in s["failures"]),
        "construction": counts(r for s in trials for r in (s["construction_violations"] or [])),
        "recording_gaps": sum(len(s.get("recording_gaps") or []) for s in trials),
        # Not a pass/fail input: the round limit is graded through "completed"
        # like any missing proposal; this says how many were over-exploration.
        "ran_out_of_turns": sum(s.get("stop_reason") == "round_limit" for s in trials),
    }


def wilson_interval(passed, n, z=1.96):
    """95% Wilson score interval for a pass rate, as (low, high). Unlike
    passed/n +- a normal error, it stays inside [0, 1] and is honest at the
    small n a k=3 benchmark has: 16/24 is 0.47-0.82, not "67%"."""
    if n == 0:
        return None
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_exact_p(a_passed, a_n, b_passed, b_n):
    """Two-sided Fisher exact test on two pass counts: the probability of a
    split at least this uneven if both runs had the same true pass rate.
    Exact, so it holds at 24 trials where a chi-square test would not."""
    total_passed, total = a_passed + b_passed, a_n + b_n

    def prob(x):  # P(run A has x of the passes), margins fixed
        return math.comb(a_n, x) * math.comb(b_n, total_passed - x) / math.comb(total, total_passed)

    observed = prob(a_passed)
    low, high = max(0, total_passed - b_n), min(a_n, total_passed)
    return min(1.0, sum(p for p in map(prob, range(low, high + 1)) if p <= observed * (1 + 1e-9)))
