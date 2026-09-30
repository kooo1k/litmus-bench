"""
decision.py

From an untrusted proposal to a checked decision. The model's propose_trade
arguments are validated against the schema, every cited number is traced to
the tool result it came from (provenance), numbers in the free text are
checked the same way, and data age and volatility are computed from the
actual tool results. The result is the TradeDecision the risk governor
judges, plus the grounding report. Nothing here trusts the model's claims.
"""

import re
from datetime import datetime

from litmus.provenance import is_sign_flip, locate, rows_as_dicts, successful_results, values_match
from litmus.risk_governor import Action, EvidenceItem, TradeDecision


def quote_rows(call):
    """The quotes in one successful call, normalized across the REST tool
    (get_price: one dict) and CMC's MCP tool (get_crypto_quotes_latest: one
    row, or a {headers, rows} table), as dicts with symbol, price_usd,
    last_updated and percent_change_1h. [] for any other tool."""
    r = call["result"]
    if call["name"] == "get_price" and isinstance(r, dict):
        return [r]
    if call["name"] == "get_crypto_quotes_latest":
        return [{"symbol": row.get("symbol"), "price_usd": row.get("price"),
                 "last_updated": row.get("last_updated_time"),
                 "percent_change_1h": row.get("percent_change_1h")} for row in rows_as_dicts(r)]
    return []


def observed_volatility(tool_calls):
    """symbol -> |1h move| as a fraction, from the latest successful quote.
    CMC reports percent (6.0 = 6%); the governor uses fractions (0.06)."""
    vol = {}
    for c in successful_results(tool_calls):
        for q in quote_rows(c):
            if isinstance(q.get("percent_change_1h"), (int, float)) and not isinstance(q["percent_change_1h"], bool):
                vol[str(q.get("symbol", "")).upper()] = abs(q["percent_change_1h"]) / 100
    return vol


# --------------------------------------------------------------------------
# Provenance verification. Governor rule B4 only fires when source_tool is
# empty, so a model that writes source_tool="get_price" next to an invented
# number would pass. We don't trust the attribution: we look for the number
# in the actual tool results, and blank source_tool when it isn't there, so
# B4 fires as intended.
#
# Limitation: only the structured `value` is checked, not numbers written in
# the free-text `claim`.
# --------------------------------------------------------------------------

def find_source(value, claimed_tool, tool_calls):
    """Returns (verified_tool, status). The claimed tool is tried first; a match
    in a different tool is a citation error, not a hallucination.

    "sign_flipped": the number isn't in any tool result but its negation is
    (-2.117 cited as 2.117). Models drop signs when the claim already says
    "down"; that's a wrong fact in the record, but not an invented number,
    and counting it as one would inflate the hallucination rate. The
    verified tool stays empty, so governor rule B4 still rejects a trade
    that cites it: the number as written doesn't match the data."""
    if locate(value, tool_calls, only_tool=claimed_tool):
        return claimed_tool, "grounded"
    other = locate(value, tool_calls)
    if other:
        return other, "misattributed"
    if is_sign_flip(value, tool_calls):
        return "", "sign_flipped"
    return "", "ungrounded"


# --------------------------------------------------------------------------
# Numbers in free text. The structured check above only sees evidence
# `value` fields, but live runs quote prices in `reasoning` too ("pulled back
# from 86,597.82"). This pass extracts numbers from prose and looks them up.
#
# It REPORTS and never blocks. Pulling numbers out of prose with a regex is
# too noisy to veto a trade on, and some numbers are legitimately derived
# (e.g. "up ~10% this week" computed from history). So the status is
# "unverified", not "ungrounded", and the count is an upper bound on
# hallucination. Every unverified number links to its exact text, so the
# benchmark report can be audited by hand.
# --------------------------------------------------------------------------

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+Z?)?")
_NUMBER = re.compile(r"(?<![\w.])(\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|[kKMBb](?![a-zA-Z]))?")
# A number followed by one of these is a duration or leverage ("24h",
# "7-day", "3x"), not market data. Lowercase m is minutes, never millions.
_UNIT_AFTER = re.compile(r"\s*-?\s*(?:h|hrs?|hours?|d|days?|w|weeks?|m|mins?|minutes?|months?|y|yrs?|years?|x)\b", re.I)
_MULTIPLIER = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "b": 1e9}


def extract_numbers(text):
    """(token, value) pairs for the market-data-like numbers in `text`.

    Skipped: timestamps; durations/leverage; bare integers under 100 with no
    $, %, decimals or suffix, which are almost always counts ("3 days",
    "top 5") rather than market data. Signs are dropped, because prose puts
    direction in words ("down 2.1%"); matching is done on magnitudes.
    """
    text = _TIMESTAMP.sub(" ", text or "")
    found = []
    for m in _NUMBER.finditer(text):
        dollar, whole, decimals, suffix = m.groups()
        if not suffix and _UNIT_AFTER.match(text, m.end()):
            continue
        value = float(whole.replace(",", "") + (decimals or ""))
        if suffix in _MULTIPLIER:
            value *= _MULTIPLIER[suffix]
        if not (dollar or suffix or decimals) and value < 100:
            continue
        found.append((m.group(0), value))
    return found


def check_text_numbers(texts, tool_calls, own_numbers=()):
    """texts: (location, text) pairs. own_numbers: the proposal's own chosen
    values (entry, stop, target, size, leverage, confidence). Restating your
    own stop-loss is a decision, not a claim about the market, so those are
    excluded. Size is also matched as a percent ("3% of equity" = 0.03)."""
    own = [v for v in own_numbers if v is not None]
    own += [v * 100 for v in own]
    report = []
    for where, text in texts:
        for token, value in extract_numbers(text):
            if any(values_match(value, o) for o in own):
                continue
            source = locate(value, tool_calls, ignore_sign=True)
            report.append({"location": where, "token": token, "value": value,
                           "found_in": source, "status": "grounded" if source else "unverified"})
    return report


def text_numbers(final_text, proposal, tool_calls):
    """The number report for everything the model wrote in prose: a
    prose-only answer, the proposal's reasoning, and each evidence claim.
    One function so every producer of a trajectory (the agent loop, and
    build_dataset when it swaps in a corrected decision) reports the same way."""
    texts = [("final_text", final_text)]
    own_numbers = []
    if proposal is not None:
        texts.append(("reasoning", proposal.get("reasoning")))
        evidence = proposal.get("evidence")
        for i, item in enumerate(evidence if isinstance(evidence, list) else []):
            if isinstance(item, dict):
                texts.append((f"evidence[{i}].claim", item.get("claim")))
        for key in ("entry_price", "stop_loss_price", "take_profit_price", "position_size_pct", "leverage", "confidence"):
            if isinstance(proposal.get(key), (int, float)) and not isinstance(proposal.get(key), bool):
                own_numbers.append(float(proposal[key]))
    return check_text_numbers([(where, str(text)) for where, text in texts if text], tool_calls, own_numbers)


# --------------------------------------------------------------------------
# Schema compliance. Gemini's OpenAI-compatible endpoint does not enforce the
# function schema (live runs omitted required fields), so compliance is
# measured rather than assumed. It is validated against propose_trade's own
# schema: one source of truth, so the check can't drift from what the model
# was actually shown.
# --------------------------------------------------------------------------

_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "array": list, "object": dict, "boolean": bool}


def schema_violations(value, schema, path="propose_trade"):
    """Violations of the JSON-schema subset used here: type, enum, required,
    unexpected properties, array items. An optional property sent as null is
    treated as absent, since it means the same thing as leaving it out."""
    expected = schema.get("type")
    if expected:
        is_bool = isinstance(value, bool)
        if not isinstance(value, _JSON_TYPES[expected]) or (is_bool and expected != "boolean"):
            return [f"{path}: expected {expected}, got {type(value).__name__}"]
    out = []
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: {value!r} not one of {schema['enum']}")
    if expected == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        out += [f"{path}: missing required field '{k}'" for k in required if k not in value]
        for key, item in value.items():
            if key not in props:
                out.append(f"{path}: unexpected field '{key}'")
            elif not (item is None and key not in required):
                out += schema_violations(item, props[key], f"{path}.{key}")
    if expected == "array" and "items" in schema:
        for i, item in enumerate(value):
            out += schema_violations(item, schema["items"], f"{path}[{i}]")
    return out


def latest_quote(symbol, tool_calls):
    """Most recent successful quote for this symbol from either quote tool,
    normalized by quote_rows plus "tool" (which tool served it), or None."""
    for c in reversed(successful_results(tool_calls)):
        for q in quote_rows(c):
            if str(q.get("symbol", "")).upper() == symbol.upper():
                return {**q, "tool": c["name"]}
    return None


def data_age_seconds(quote, now):
    """Trusted freshness: computed from the tool's own timestamp."""
    try:
        ts = datetime.fromisoformat(str(quote["last_updated"]).replace("Z", "+00:00"))
        return (now - ts).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


class InvalidProposal(ValueError):
    pass


def build_decision(proposal, tool_calls, now):
    """Turn the model's raw propose_trade arguments into a TradeDecision whose
    trust-critical fields are set by the harness. Returns (decision, grounding)."""
    if "evidence" not in proposal:
        # Live runs omit evidence on HOLD. Rejecting a safe HOLD as invalid
        # would count caution as failure, so default to [] and let governor
        # rule B3 decide (it demands evidence only for LONG/SHORT). The
        # omission is still counted by schema_violations().
        proposal = {**proposal, "evidence": []}
    try:
        action = str(proposal["action"])
        symbol = str(proposal["symbol"])  # not normalized: a bad ticker is an argument-validity failure, rule A2
        confidence = float(proposal["confidence"])
        size = float(proposal["position_size_pct"])
        leverage = float(proposal["leverage"])
        horizon = str(proposal["time_horizon"])
        raw_evidence = proposal["evidence"]
        reasoning = str(proposal.get("reasoning", ""))
        entry = proposal.get("entry_price")
        stop = proposal.get("stop_loss_price")
        target = proposal.get("take_profit_price")
        entry = None if entry is None else float(entry)
        stop = None if stop is None else float(stop)
        target = None if target is None else float(target)
        if not isinstance(raw_evidence, list):
            raise TypeError("evidence must be a list")
    except (KeyError, TypeError, ValueError) as e:
        raise InvalidProposal(f"{type(e).__name__}: {e}") from e

    opens_risk = action in (Action.LONG.value, Action.SHORT.value)
    if opens_risk and entry is None:
        raise InvalidProposal(f"{action} proposed without an entry_price")

    grounding = []
    evidence = []
    for item in raw_evidence:
        if not isinstance(item, dict) or "claim" not in item:
            raise InvalidProposal(f"malformed evidence item: {item!r}")
        claimed = str(item.get("source_tool", ""))
        value = item.get("value")
        if value is None:
            evidence.append(EvidenceItem(str(item["claim"]), claimed, None))
            continue
        try:
            value = float(value)
        except (TypeError, ValueError) as e:
            raise InvalidProposal(f"evidence value not a number: {value!r}") from e
        verified, status = find_source(value, claimed, tool_calls)
        grounding.append({"claim": item["claim"], "value": value, "claimed_source": claimed,
                          "verified_source": verified, "status": status})
        evidence.append(EvidenceItem(str(item["claim"]), verified, value))

    quote = latest_quote(symbol, tool_calls)
    if opens_risk:
        # The entry price is a cited number like any other, but it must come
        # from a quote for *this* symbol — a BTC price can't justify an ETH entry.
        quote_tool = quote["tool"] if quote is not None else "get_price"
        if (quote is not None and isinstance(quote.get("price_usd"), (int, float))
                and values_match(entry, float(quote["price_usd"]))):
            verified, status = quote_tool, "grounded"
        else:
            verified, status = "", "ungrounded"
        grounding.append({"claim": "[harness] entry_price", "value": entry, "claimed_source": quote_tool,
                          "verified_source": verified, "status": status})
        evidence.append(EvidenceItem("[harness] entry_price", verified, entry))

    decision = TradeDecision(
        action=action,
        symbol=symbol,
        confidence=confidence,
        position_size_pct=size,
        leverage=leverage,
        # HOLD/CLOSE carry no entry; 0.0 is a placeholder no governor rule reads for those actions.
        entry_price=entry if entry is not None else 0.0,
        stop_loss_price=stop,
        take_profit_price=target,
        time_horizon=horizon,
        evidence=evidence,
        data_last_updated_seconds_ago=data_age_seconds(quote, now) if quote else None,
        reasoning=reasoning,
    )
    return decision, grounding
