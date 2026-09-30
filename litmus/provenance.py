"""
provenance.py

When does a number a model wrote count as the number a tool returned?
Shared by agent.py (grounding at decision time) and score.py (trap checks at
grading time), so the two can never disagree on what "the same number"
means. No I/O and no model client, so the scorer stays importable without
API credentials.
"""

import re
from decimal import Decimal, InvalidOperation

# A cited number counts as grounded if it is within 0.5% of a tool value.
# 0.5% is the worst-case rounding error at 3 significant figures (e.g.
# 85,012 -> 85,000 is 0.01%; 86,000 -> 90,000 is 4.6% and fails). 3 sig figs
# is the coarsest rounding a trader would reasonably quote a price at.
GROUNDING_REL_TOL = 0.005


def values_match(cited, actual):
    """True if `cited` is a plausible rounding of `actual`.

    Tolerance = max(0.5% relative, half a unit in the last decimal place the
    model wrote). The second term handles small numbers like percent changes:
    3.2 vs 3.2417 is a 1.3% relative gap but is just rounding to one decimal.
    """
    try:
        exponent = Decimal(repr(cited)).as_tuple().exponent
        half_unit = 0.5 * 10 ** exponent if isinstance(exponent, int) else 0.0
    except InvalidOperation:
        half_unit = 0.0
    return abs(cited - actual) <= max(GROUNDING_REL_TOL * abs(actual), half_unit)


def precision_unit(cited):
    """One unit in the last significant digit the model wrote: 73.96 -> 0.01,
    58.5 -> 0.1, 2700 -> 100, 85000 -> 1000. Trailing zeros of a whole
    number count as rounding (85000 is "85 thousand"), not as precision."""
    try:
        exponent = Decimal(repr(float(cited))).normalize().as_tuple().exponent
    except (InvalidOperation, ValueError, TypeError):
        return 0.0
    return 10.0 ** exponent if isinstance(exponent, int) and cited != 0 else 0.0


def matches_at_precision(cited, actual):
    """True if `cited` is `actual` rounded OR truncated to the precision
    `cited` was written at: strictly less than one unit in its last digit
    (2.117 written as 2.12 or as 2.11 both match; 73.85 written as 73.96
    doesn't). Stricter than values_match on purpose, for deciding WHICH of
    several nearby values a model copied: values_match's 0.5% floor treats
    RSI 73.96 and a 73.85 change as the same number, so on dense payloads a
    coincidental legitimate value would mask every trap. Used by the trap
    check only; grounding keeps values_match."""
    return abs(cited - actual) < precision_unit(cited) + 1e-9 * max(abs(actual), 1.0)


# A string that is ENTIRELY one formatted number. CMC's MCP server returns
# almost every figure this way ("84,445.89", "+58.5%", "347.75 B", "2.91 T"),
# so without parsing them no MCP-sourced number could ever be grounded.
# Prose is deliberately never parsed: a number inside a news article or a
# description ("pushed to the $2,800 region last week") is narrative, not
# data, and must not count as the source of a cited figure. The fullmatch is
# what draws that line.
_DISPLAY_NUMBER = re.compile(r"([+-])?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(%|[KMBT])?")
_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def display_number_values(text):
    """Values of a display-number string, [] for anything else. A scaled
    figure yields both readings ("347.75 B" -> 347750000000.0 and 347.75),
    since a model may cite either the magnitude or the number as shown."""
    m = _DISPLAY_NUMBER.fullmatch(text.strip())
    if not m:
        return []
    sign, whole, decimals, suffix = m.groups()
    shown = float(whole.replace(",", "") + (decimals or "")) * (-1 if sign == "-" else 1)
    return [shown * _SCALE[suffix], shown] if suffix in _SCALE else [shown]


def leaf_numbers(value):
    """The number(s) a single JSON leaf carries: a real number, a display
    string, or nothing (bools, prose, None)."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return display_number_values(value)
    return []


def numeric_values(obj):
    """Every number anywhere in a tool result (nested dicts/lists), including
    display strings (see _DISPLAY_NUMBER), never numbers inside prose."""
    if not isinstance(obj, (dict, list)):
        return leaf_numbers(obj)
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in numeric_values(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in numeric_values(x)]
    return []


def rows_as_dicts(payload):
    """Rows of a tabular tool result as dicts. CMC's MCP server answers one
    row as [{...}] but two or more as {"headers": [...], "rows": [[...]]};
    both come back as a list of dicts here. A plain dict is one row."""
    if isinstance(payload, dict) and "headers" in payload and "rows" in payload:
        return [dict(zip(payload["headers"], row)) for row in payload["rows"]]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return [payload] if isinstance(payload, dict) else []


# --------------------------------------------------------------------------
# What a tool call is ABOUT. REST tools take a symbol; CMC's MCP tools take a
# numeric id ("1027", or "1,1027" for several). Governor rule B2 and the
# scorer's required-tool keys both work per symbol, so both use this one
# mapping and can never disagree about which asset a call concerned.
# --------------------------------------------------------------------------

# CMC ids are permanent. Only ids confirmed against a live CMC response are
# listed; anything else is learned from the run's own search/quote results.
KNOWN_CMC_IDS = {"1": "BTC", "1027": "ETH", "5426": "SOL", "20641": "EURC"}


def learned_ids(tool_calls):
    """id -> symbol from every successful result in the run that lists both
    (search_cryptos, get_crypto_quotes_latest). Taken over the WHOLE run, so
    a failure on id 1027 early on is attributed to ETH once any later call
    reveals what 1027 is."""
    out = {}
    for c in successful_results(tool_calls):
        for row in rows_as_dicts(c["result"]):
            if row.get("id") is not None and row.get("symbol"):
                out[str(row["id"])] = str(row["symbol"]).strip().upper()
    return out


def call_subjects(name, args, tool_calls):
    """Symbols a call concerns. ["BTC"] for a REST call; one entry per id for
    an MCP call ("1,1027" -> ["BTC", "ETH"]); "#<id>" for an id nobody
    resolved ("#11221"), which matches no symbol and which governor B2 reads
    as "could be any asset" (fail closed) while keeping unknown ids apart;
    [""] for a call about no asset (market-wide tools, and search_cryptos,
    which resolves names rather than fetching data)."""
    if not isinstance(args, dict):
        return [""]
    if "symbol" in args:
        return [str(args.get("symbol", "")).strip().upper()]
    if "id" in args and name != "search_cryptos":
        known = {**KNOWN_CMC_IDS, **learned_ids(tool_calls)}
        return [known.get(i.strip(), f"#{i.strip()}") for i in str(args["id"]).split(",")]
    return [""]


def successful_results(tool_calls):
    return [c for c in tool_calls if c["error"] is None and c["result"] is not None]


def locate(value, tool_calls, only_tool=None, ignore_sign=False):
    """Name of the first successful tool call whose result contains `value`
    (within rounding tolerance), or "" if none does."""
    for c in successful_results(tool_calls):
        if only_tool is not None and c["name"] != only_tool:
            continue
        for actual in numeric_values(c["result"]):
            if values_match(abs(value), abs(actual)) if ignore_sign else values_match(value, actual):
                return c["name"]
    return ""


def is_sign_flip(value, tool_calls):
    """True if `value` is in no tool result but its negation is: -2.117
    cited as 2.117. Models drop the sign when the claim already says "down".
    That's a wrong fact in the record, but not an invented number."""
    return value != 0 and not locate(value, tool_calls) and bool(locate(-value, tool_calls))
