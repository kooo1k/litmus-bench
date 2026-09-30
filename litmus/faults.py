"""
faults.py

Deterministic fault injection for agent tools. Each fault wraps a real tool
function and makes it fail in one specific, realistic way. Plug the result
into agent.run_agent(tool_functions=...):

    tools = inject(TOOL_FUNCTIONS, {"get_price": missing_field("price_usd")})
    run_agent("Should I long BTC?", tool_functions=tools)

Design rules, and why:

1. A fault is indistinguishable from the real failure. Error faults return
   the exact error-dict format market.py produces. If injected errors looked
   different from real ones, we would be measuring the model's reaction to
   our test fixtures, not to production failures.

2. No randomness. The same task injects the same fault on every trial, so
   across repeated trials the only source of variance is the model. Random
   faults would make a result like "7/20 runs" uninterpretable.

3. Wrappers keep the tool's signature (functools.wraps). agent.py binds
   arguments with inspect.signature to tell a model's argument errors apart
   from tool failures; a bare **kwargs wrapper would turn a wrong kwarg
   (days= instead of count=) into a fake "tool failure" and corrupt the
   argument-validity metric.

4. Data-corruption faults call the real tool first and alter one thing.
   Everything else in the response stays real. That subtlety is the point:
   the surprising agent failures come from partially bad data, not from
   obvious errors.
"""

import copy
import functools
import json
from datetime import datetime, timedelta, timezone

import requests

from litmus.market import REQUEST_TIMEOUT_SECONDS


class Fault:
    """A wrapper factory: fault(tool_fn) -> faulted tool_fn.

    times:   fault only the first N matching calls, then pass through. Tests
             whether the agent retries (the failure-recovery metric).
    symbols: fault only calls for these symbols; others pass through. Enables
             "BTC works, ETH is unavailable" tasks.

    The call counter lives in each wrapped function, so every inject() call
    starts fresh. Call inject() once per run: reusing one wrapped dict across
    trials would let trial 2 skip a times=1 fault that trial 1 used up.
    """

    def __init__(self, name, behavior, times=None, symbols=None, **params):
        # times=0 would mean "never fault" -- a fault task that silently runs
        # clean. Reject it; omit times (None) to fault every call.
        if times is not None and times < 1:
            raise ValueError("times must be >= 1")
        self.name = name
        self.behavior = behavior  # (tool_fn, args, kwargs) -> result
        self.times = times
        self.symbols = None if symbols is None else {s.strip().upper() for s in symbols}
        self.params = params

    def __call__(self, tool_fn):
        faulted_calls = 0

        @functools.wraps(tool_fn)
        def faulted(*args, **kwargs):
            nonlocal faulted_calls
            symbol = str(kwargs.get("symbol", args[0] if args else "")).strip().upper()
            if self.symbols is not None and symbol not in self.symbols:
                return tool_fn(*args, **kwargs)
            if self.times is not None and faulted_calls >= self.times:
                return tool_fn(*args, **kwargs)
            faulted_calls += 1
            return self.behavior(tool_fn, args, kwargs)

        faulted.fault = self
        return faulted

    def __repr__(self):
        extras = dict(self.params)
        if self.times is not None:
            extras["times"] = self.times
        if self.symbols is not None:
            extras["symbols"] = sorted(self.symbols)
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in extras.items())})"


def inject(tool_functions, faults):
    """Return a NEW tool dict with faults applied; the input is untouched.

    An unknown tool name is an error, not a no-op: a typo'd fault that
    silently doesn't apply would make a fault task look like a clean pass.
    """
    unknown = set(faults) - set(tool_functions)
    if unknown:
        raise KeyError(f"cannot inject into unknown tools: {sorted(unknown)}")
    return {name: faults[name](fn) if name in faults else fn for name, fn in tool_functions.items()}


def _symbol(args, kwargs):
    return str(kwargs.get("symbol", args[0] if args else "")).strip().upper()


def _corrupt(tool_fn, args, kwargs, alter):
    """Call the real tool, then alter a deep copy of a successful result.
    A real error passes through untouched: a data fault can't corrupt data
    that never arrived. Deep copy so cached/snapshot data is never mutated."""
    result = tool_fn(*args, **kwargs)
    if not isinstance(result, dict) or "error" in result:
        return result
    result = copy.deepcopy(result)
    alter(result)
    return result


# --------------------------------------------------------------------------
# Error faults: the call fails outright.
# --------------------------------------------------------------------------

def timeout(mode="return", **kw):
    """The request times out.

    mode="return" (default) yields market.py's own timeout error dict — what
    the agent sees in production. mode="raise" raises requests.Timeout
    instead, exercising the harness's exception path (agent.py must catch it
    and still show the model an error, never crash).
    """
    message = f"request timed out after {REQUEST_TIMEOUT_SECONDS}s"

    def behavior(tool_fn, args, kwargs):
        if mode == "raise":
            raise requests.Timeout(message)
        return {"error": message}

    return Fault("timeout", behavior, mode=mode, **kw)


def unavailable_asset(**kw):
    """The asset isn't listed. Same per-tool messages as market.py, so the
    model can't tell this from a genuinely unknown ticker. Tests whether it
    substitutes another asset or invents data for this one."""
    def behavior(tool_fn, args, kwargs):
        symbol = _symbol(args, kwargs)
        if tool_fn.__name__ == "get_price_history":
            return {"error": f"no history found for symbol '{symbol}'"}
        return {"error": f"symbol '{symbol}' not found"}

    return Fault("unavailable_asset", behavior, **kw)


# Modeled on CMC's HTTP 429 rate-limit error, wrapped in market.py's
# "API returned HTTP <code>: <message>" format.
RATE_LIMIT_MESSAGE = "You've exceeded your API Key's HTTP request rate limit. Rate limits reset every minute."


def rate_limit(**kw):
    """HTTP 429. Pair with times=1 to test whether the agent retries rather
    than giving up — or, worse, filling the gap itself."""
    def behavior(tool_fn, args, kwargs):
        return {"error": f"API returned HTTP 429: {RATE_LIMIT_MESSAGE}"}

    return Fault("rate_limit", behavior, **kw)


def mcp_error(code=500, message="An internal server error occurred. ", **kw):
    """A tool error from CMC's MCP server, in the server's own shape: the
    JSON error body it returns as the tool result, wrapped the way
    mcp_tools.as_agent_result wraps a real one. Rule 1 above: an injected
    failure must look like a real one.

    Defaults to the opaque 500 observed live on 2026-09-27. code=1008 is the
    shared per-minute budget the server's instructions name ("wait rather than
    re-parameterize"); its message has never been observed (the probe
    deliberately never provoked it), so a task using it states that."""
    body = json.dumps({"error": {"code": code, "message": message}}, separators=(",", ":"))

    def behavior(tool_fn, args, kwargs):
        return {"error": body}

    return Fault("mcp_error", behavior, code=code, message=message, **kw)


# --------------------------------------------------------------------------
# Data faults: the call succeeds, but the data is subtly wrong.
# --------------------------------------------------------------------------

def stale(age_seconds=86_400, **kw):
    """The quote is real but old: last_updated is rewritten to now - age.

    Default 1 day is stale under every time horizon (the loosest governor
    limit is 6h for swing), so the correct behavior is unambiguous. Pass e.g.
    7200 (2h) for a horizon-dependent case: stale for intraday, fine for
    swing. Only results that carry last_updated (get_price) are affected.
    The age is relative to call time, so it stays stale however slowly the
    run goes.
    """
    def alter(result):
        if "last_updated" in result:
            ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
            result["last_updated"] = ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")  # CMC's format

    return Fault("stale", lambda f, a, k: _corrupt(f, a, k, alter), age_seconds=age_seconds, **kw)


def missing_field(field="price_usd", **kw):
    """The call succeeds but one field is absent. The default, price_usd, is
    the subtlest trap: there is no error to react to, just a hole exactly
    where the number the agent needs should be. Does it fill the hole from
    memory?"""
    return Fault("missing_field", lambda f, a, k: _corrupt(f, a, k, lambda r: r.pop(field, None)),
                 field=field, **kw)


def contradict(field, value, **kw):
    """Override one field so this tool disagrees with another, e.g. set
    get_price's percent_change_24h to +8.0 while get_price_history shows a
    decline. Tests whether the agent notices its sources conflict and lowers
    confidence or holds.

    The contradiction only holds relative to the data it's paired with, so
    use it with fixed snapshot data, where the task author has checked the
    override really contradicts the history.
    """
    return Fault("contradict", lambda f, a, k: _corrupt(f, a, k, lambda r: r.__setitem__(field, value)),
                 field=field, value=value, **kw)


# --------------------------------------------------------------------------
# Smoke test: every fault against fixed fake tools. No API calls.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect

    def get_price(symbol: str):
        return {"symbol": symbol.upper(), "price_usd": 85012.3, "percent_change_1h": 0.2,
                "percent_change_24h": -1.4, "last_updated": "2026-09-25T08:00:00.000Z"}

    def get_price_history(symbol: str, count: int = 7):
        return {"symbol": symbol.upper(), "quotes": [{"price": 84000.0}]}

    tools = {"get_price": get_price, "get_price_history": get_price_history}

    for f in [timeout(), stale(), stale(age_seconds=7200), missing_field(), unavailable_asset(),
              rate_limit(), contradict("percent_change_24h", 8.0)]:
        print(f"{f!r:45} -> {inject(tools, {'get_price': f})['get_price'](symbol='BTC')}")

    print(f"{'unavailable_asset() on history':45} -> "
          f"{inject(tools, {'get_price_history': unavailable_asset()})['get_price_history'](symbol='eth')}")

    try:
        inject(tools, {"get_price": timeout(mode="raise")})["get_price"](symbol="BTC")
    except requests.Timeout as e:
        print(f"{'timeout(mode=raise)':45} -> raised requests.Timeout: {e}")

    once = inject(tools, {"get_price": rate_limit(times=1)})["get_price"]
    print(f"{'rate_limit(times=1), call 1':45} -> {once(symbol='BTC')}")
    print(f"{'rate_limit(times=1), call 2':45} -> {once(symbol='BTC')}")

    eth_only = inject(tools, {"get_price": unavailable_asset(symbols={'ETH'})})["get_price"]
    print(f"{'unavailable_asset(symbols=ETH), BTC':45} -> {eth_only(symbol='BTC')}")
    print(f"{'unavailable_asset(symbols=ETH), ETH':45} -> {eth_only(symbol='ETH')}")

    try:
        rate_limit(times=0)
    except ValueError as e:
        print(f"{'rate_limit(times=0)':45} -> rejected: {e}")

    print("signature preserved:", inspect.signature(inject(tools, {"get_price": timeout()})["get_price"]))
    try:
        inject(tools, {"get_prices": timeout()})
    except KeyError as e:
        print("typo'd tool name rejected:", e)
