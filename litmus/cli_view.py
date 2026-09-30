"""
cli_view.py

Terminal display for the `litmus` command (cli.py): a live progress view
while the agent works (spinner plus one line per tool call) and a final
screen with the decision, the evidence and where each number came from, the risk governor's
verdict, and, for a benchmark task, the grade.

Display only: nothing here changes what the agent does or how it is scored.
Uses `rich` when installed and falls back to plain text otherwise, so the
benchmark and the tests never depend on it.
"""

import json
import sys

from litmus.provenance import rows_as_dicts

try:
    from rich.console import Console, Group
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAVE_RICH = True
except ImportError:  # pragma: no cover - exercised only without rich installed
    HAVE_RICH = False

    def escape(text):
        return text

# What each failed check means, in the words a reader of the verdict needs.
CHECK_MEANING = {
    "completed": "no valid trade proposal",
    "required_tools": "never fetched data the question needs",
    "data_seen": "traded on data that never arrived",
    "action": "wrong decision",
    "symbol": "wrong asset",
    "size": "wrong position size",
    "grounded": "cited a number no tool returned",
    "sign": "flipped the sign of a number",
    "trap": "relied on a tempting but wrong field",
}
STOP_MEANING = {"round_limit": "ran out of turns while still calling tools",
                "text": "answered in prose instead of proposing"}


# --------------------------------------------------------------------------
# One-line tool summaries
# --------------------------------------------------------------------------

def _money(v):
    if v is None:
        return "–"  # not set, e.g. a proposal without a take-profit
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"${v:,.2f}"


def _num(v):
    """A cited value for display: 2714.9470109474396 -> 2,714.95; 0.03 -> 0.03."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4g}"


def _pct(v):
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _quotes(rows):
    return " · ".join(f"{r.get('symbol')} {_money(r.get('price'))} (24h {_pct(r.get('percent_change_24h'))})" for r in rows)


def _summaries():
    def search(r, args):
        rows = rows_as_dicts(r)
        return ", ".join(f"{x.get('symbol')} ({x.get('name')}) → id {x.get('id')}" for x in rows) or "no match"

    def ta(r, args):
        if r == {}:
            return "{} — empty, no error (the filter matched no block)"
        parts = []
        if isinstance(r.get("rsi"), dict):
            parts.append(f"RSI14 {r['rsi'].get('rsi14')}")
        ma = r.get("moving_averages") or {}
        if ma.get("simple_moving_average_7_day"):
            parts.append(f"SMA7 {ma['simple_moving_average_7_day']}")
        if isinstance(r.get("macd"), dict):
            parts.append(f"MACD {r['macd'].get('histogram', '')}".strip())
        return ", ".join(parts) or f"blocks: {', '.join(r)}"

    def mcap_ta(r, args):
        return (f"TOTAL market: RSI14 {(r.get('rsi') or {}).get('rsi14')}, cap {r.get('currentMarketCap')}"
                + ("  (id ignored: this tool is market-wide)" if args.get("id") else ""))

    def global_metrics(r, args):
        dom = ((r.get("dominance") or {}).get("btc") or {}).get("current")
        fg = (((r.get("sentiment") or {}).get("fear_greed") or {}).get("current") or {})
        return f"BTC dominance {dom} · Fear & Greed {fg.get('index')} ({fg.get('value')}) · updated {r.get('last_updated')}"

    def derivatives(r, args):
        day = (r.get("btc_liquidations") or {}).get("total_usd_24h") or {}
        oi = (r.get("totalOpenInterest") or {}).get("current")
        return f"BTC liquidations 24h: longs {day.get('long')} / shorts {day.get('short')} · open interest {oi} · no timestamp"

    def news(r, args):
        rows = rows_as_dicts(r)
        if not rows:
            return "no articles"
        return f"{len(rows)} article(s); latest: {str(rows[0].get('title', ''))[:70]}"

    def count(r, args):
        rows = rows_as_dicts(r)
        n = len(rows) if rows else (len(r) if isinstance(r, (list, dict)) else 0)
        return f"{n} row(s)"

    def skills(r, args):
        names = [c.get("uniqueName") for c in (r or {}).get("candidates", [])]
        return f"{len(names)} skill(s): " + ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")

    return {
        "search_cryptos": search,
        "get_crypto_quotes_latest": lambda r, a: _quotes(rows_as_dicts(r)) or "no rows",
        "get_crypto_technical_analysis": ta,
        "get_crypto_marketcap_technical_analysis": mcap_ta,
        "get_global_metrics_latest": global_metrics,
        "get_global_crypto_derivatives_metrics": derivatives,
        "get_crypto_latest_news": news,
        "trending_crypto_narratives": count,
        "get_upcoming_macro_events": count,
        "find_skill": skills,
        "get_price": lambda r, a: f"{r.get('symbol')} {_money(r.get('price_usd'))} (24h {_pct(r.get('percent_change_24h'))})",
        "get_price_history": lambda r, a: f"{r.get('symbol')}: {len(r.get('quotes', []))} daily closes",
        "get_ohlcv": lambda r, a: f"{r.get('symbol')}: {len(r.get('candles', []))} daily candles",
        "get_market_overview": lambda r, a: f"total cap ${r.get('total_market_cap_usd', 0) / 1e12:.2f}T · BTC dominance {r.get('btc_dominance_pct')}%",
        "get_fear_greed": lambda r, a: f"Fear & Greed {r.get('value')} ({r.get('classification')})",
    }


SUMMARIES = _summaries()


def safe_console():
    """Never crash on a character the console can't show. A Windows console
    or a redirected stream is often cp1252, which has no "→" or "✗": print a
    placeholder instead of raising UnicodeEncodeError mid-run."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def summarize_tool(record):
    """One line for one tool call: what came back, in human terms."""
    if record.get("error_kind") == "skipped":
        return "skipped (sent in the same turn as the proposal)"
    if record.get("error"):
        return f"ERROR {str(record['error'])[:140]}"
    fn = SUMMARIES.get(record.get("name"))
    try:
        if fn is not None:
            return fn(record.get("result"), record.get("args") or {})
    except Exception:  # a display summary must never break a run
        pass
    text = json.dumps(record.get("result"), default=str)
    return text[:120] + ("..." if len(text) > 120 else "")


def call_label(record):
    args = record.get("args")
    shown = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items()) if args is not None else record.get("raw_arguments")
    return f"{record.get('name')}({shown})"


# --------------------------------------------------------------------------
# Live progress
# --------------------------------------------------------------------------

class LiveView:
    """Pass `on_event` to run_agent inside `with view.running(...)`."""

    def __init__(self, plain=False):
        self.rich = HAVE_RICH and not plain
        self.console = Console(highlight=False) if self.rich else None
        self._status = None

    def header(self, model, provider, toolset, question, notes=()):
        if self.rich:
            self.console.print(Panel(Text(question, style="bold"),
                                     title=f"[bold]{escape(model)}[/] @ {escape(provider)} · {escape(toolset)}",
                                     title_align="left"))
            for n in notes:
                self.console.print(f"[dim]note: {escape(n)}[/]")
        else:
            print(f"== {model} @ {provider} · {toolset}\nQ: {question}")
            for n in notes:
                print(f"note: {n}")

    def running(self):
        view = self

        class _Ctx:
            def __enter__(self):
                if view.rich:
                    view._status = view.console.status("model thinking…")
                    view._status.__enter__()
                return view

            def __exit__(self, *exc):
                if view._status is not None:
                    view._status.__exit__(*exc)
                    view._status = None
                return False
        return _Ctx()

    def on_event(self, kind, data):
        if kind == "model":
            text = f"model thinking… (round {data['round']}/{data['max_rounds']})"
            if self._status is not None:
                self._status.update(text)
            elif not self.rich:
                print(f"  ... {text}")
        elif kind == "tool":
            failed = bool(data.get("error"))
            if self.rich:
                mark = "[red]✗[/]" if failed else "[green]✓[/]"
                self.console.print(f"  {mark} [cyan]{escape(call_label(data))}[/]")
                self.console.print(f"      [{'red' if failed else 'dim'}]{escape(summarize_tool(data))}[/]")
            else:
                print(f"  {'x' if failed else '+'} {call_label(data)}\n      {summarize_tool(data)}")

    def error(self, message, hint=""):
        if self.rich:
            self.console.print(Panel(f"{escape(message)}\n[dim]{escape(hint)}[/]" if hint else escape(message),
                                     title="[red]could not complete[/]", border_style="red"))
        else:  # stderr, so --json output on stdout stays valid JSON
            print(f"ERROR: {message}" + (f"\n{hint}" if hint else ""), file=sys.stderr)


# --------------------------------------------------------------------------
# Final screen
# --------------------------------------------------------------------------

def governor_caught_nothing(trajectory, score):
    """The demo's point in one condition: the benchmark failed the decision
    but the risk governor let the trade through."""
    risk = trajectory.get("risk_result") or {}
    proposed = (score or {}).get("proposed") or {}
    return bool(score) and not score["passed"] and risk.get("approved") and proposed.get("action") in ("LONG", "SHORT")


def verdict_lines(task, score, trajectory):
    e = task["expected"]
    p = score.get("proposed") or {}
    lines = [f"expected: {' / '.join(e['acceptable_actions'])}"
             + (f" {' / '.join(e['acceptable_symbols'])}" if set(e["acceptable_actions"]) & {"LONG", "SHORT"} else ""),
             f"proposed: {p.get('action') or 'nothing'} {p.get('symbol') or ''}".rstrip()]
    for f in score["failures"]:
        lines.append(f"✗ {f}: {CHECK_MEANING.get(f, f)}")
    for hit in score.get("trap_hits") or []:
        lines.append(f"  trap value {hit['value']} from {hit['trap']} (in {hit['location']})")
    if score.get("stop_reason") in STOP_MEANING and not score["passed"]:
        lines.append(f"  ({STOP_MEANING[score['stop_reason']]})")
    if governor_caught_nothing(trajectory, score):
        lines.append("⚠ The risk governor APPROVED this trade. Only the benchmark caught it.")
    return lines


def print_summary(t):
    """The whole trajectory as plain text: the fallback when rich is
    missing or --plain is set."""
    print()
    print(f"outcome:      {t['outcome']}")
    if t["outcome"] == "no_proposal":
        print(f"agent said:   {t['final_text']}")
    elif t["outcome"] == "invalid_proposal":
        print(f"error:        {t['proposal_error']}")
    else:
        d = t["decision"]
        print(f"proposal:     {d['action']} {d['symbol']}  confidence={d['confidence']}  "
              f"size={d['position_size_pct']}  leverage={d['leverage']}x")
        print(f"reasoning:    {d['reasoning']}")
        for g in t["grounding"]:
            print(f"  evidence [{g['status']:>13}] {g['claim']} = {g['value']} "
                  f"(claimed {g['claimed_source'] or '-'}, verified {g['verified_source'] or '-'})")
        r = t["risk_result"]
        print(f"governor:     {'APPROVED' if r['approved'] else 'REJECTED'}  "
              f"size={r['adjusted_position_size_pct']}  leverage={r['adjusted_leverage']}x  "
              f"risk_score={r['risk_score']}")
        for v in r["violations"]:
            print(f"  [{v['severity']}] {v['rule_id']}: {v['message']}")
    if t["schema_compliant"] is not None:
        print(f"schema:       {'compliant' if t['schema_compliant'] else 'NON-COMPLIANT'}")
        for w in t["schema_warnings"]:
            print(f"  {w}")
    unverified = [n for n in t["text_numbers"] if n["status"] == "unverified"]
    print(f"text numbers: {len(t['text_numbers'])} checked, {len(unverified)} unverified")
    for n in unverified:
        print(f"  unverified '{n['token']}' in {n['location']}")
    print(f"final action: {t['final_action']}")
    if "saved_to" in t:
        print(f"trajectory:   {t['saved_to']}")


def render_result(view, trajectory, task=None, score=None):
    if not view.rich:
        print_summary(trajectory)
        if task is not None:
            print(f"\nBENCHMARK {task['id']}: {'PASS' if score['passed'] else 'FAIL'}")
            for line in verdict_lines(task, score, trajectory):
                print(f"  {line}")
            print(f"  why: {task['rationale']}")
        return

    c, t = view.console, trajectory
    if t["outcome"] != "proposal":
        why = STOP_MEANING.get(t.get("stop_reason"), t.get("proposal_error") or "")
        body = Text(f"No trade proposal ({why}).")
        if t.get("final_text"):
            body.append(f"\n\nThe agent said: {t['final_text']}")
        c.print(Panel(body, title="decision", border_style="yellow"))
    else:
        d = t["decision"]
        head = Text(f"{d['action']} {d['symbol']}", style="bold")
        if d["action"] in ("LONG", "SHORT"):
            head.append(f"   entry {_money(d['entry_price'])}  stop {_money(d['stop_loss_price'])}"
                        f"  target {_money(d['take_profit_price'])}", style="")
        head.append(f"\nconfidence {d['confidence']}   size {d['position_size_pct']}   leverage {d['leverage']}x")
        c.print(Panel(Group(head, Text(""), Text(d["reasoning"] or "", style="italic")),
                      title="decision and the agent's reasoning", title_align="left"))

        if t["grounding"]:
            table = Table(title="evidence: where each cited number came from", title_justify="left")
            for col in ("claim", "value", "cited tool", "found in", "status"):
                table.add_column(col)
            colour = {"grounded": "green", "misattributed": "yellow", "sign_flipped": "yellow", "ungrounded": "red"}
            for g in t["grounding"]:
                table.add_row(escape(str(g["claim"])), _num(g["value"]), escape(g["claimed_source"] or "-"),
                              escape(g["verified_source"] or "-"), f"[{colour.get(g['status'], 'white')}]{g['status']}[/]")
            c.print(table)

        r = t["risk_result"]
        verdict = "[green]APPROVED[/]" if r["approved"] else "[red]REJECTED[/]"
        lines = [f"{verdict}   final action {t['final_action']}   size {r['adjusted_position_size_pct']}"
                 f"   leverage {r['adjusted_leverage']}x"]
        lines += [escape(f"  [{v['severity']}] {v['rule_id']}: {v['message']}") for v in r["violations"]]
        c.print(Panel("\n".join(lines), title="risk governor (deterministic rules)", title_align="left"))

    if task is not None:
        ok = score["passed"]
        body = "\n".join(escape(line) for line in verdict_lines(task, score, t)) + f"\n\n[dim]why: {escape(task['rationale'])}[/]"
        c.print(Panel(body, title=f"[bold]benchmark {task['id']}: {'PASS' if ok else 'FAIL'}[/]",
                      title_align="left", border_style="green" if ok else "red"))
    if t.get("saved_to"):
        c.print(f"[dim]trajectory saved: {t['saved_to']}[/]")


def render_task_list(view, tasks):
    if view.rich:
        table = Table(title="benchmark tasks (run one with --task ID)", title_justify="left")
        for col in ("id", "suite", "category", "question"):
            table.add_column(col, overflow="fold")
        for t in tasks:
            table.add_row(t["id"], t.get("toolset", "rest"), t["category"], escape(t["question"]))
        view.console.print(table)
    else:
        width = max([24] + [len(t["id"]) for t in tasks])
        for t in tasks:
            print(f"{t['id']:{width}} {t.get('toolset', 'rest'):4} {t['category']:18} {t['question']}")


# --------------------------------------------------------------------------
# Shell banner
# --------------------------------------------------------------------------

_GLYPHS = {
    "L": ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    "M": ["███╗   ███╗", "████╗ ████║", "██╔████╔██║", "██║╚██╔╝██║", "██║ ╚═╝ ██║", "╚═╝     ╚═╝"],
    "U": ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
}
_GRADIENT = ((0, 212, 255), (140, 82, 255))  # cyan -> violet, left to right


def logo_lines(word="LITMUS"):
    return ["".join(_GLYPHS[c][row] for c in word) for row in range(6)]


def _can_draw_blocks():
    try:
        "█╗╚".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _version():
    try:
        from importlib.metadata import version
        return version("litmus-bench")
    except Exception:
        return "dev"


TAGLINE = "Real numbers. Wrong premise. Litmus catches it."


def status_lines(status):
    """(label, rich text, plain text) per line: model, CMC, mode, prompt.
    One line each, shared by the banner and /status."""
    reasoning = f" · reasoning {status['reasoning']}" if status.get("reasoning") else ""
    model = f"{status['model']} @ {status['provider']}{reasoning}"
    return [
        ("model", f"[bold]{escape(model)}[/]  " + ("[green]✓ key[/]" if status["model_key"]
                                                   else f"[red]✗ set {status['key_var']}[/]"),
         f"{model}  key {'yes' if status['model_key'] else 'NO'} ({status['key_var']})"),
        ("CMC", "[green]✓ key[/]" if status["cmc_key"] else "[red]✗ set CMC_API_KEY[/] [dim](needed for live questions)[/]",
         f"key {'yes' if status['cmc_key'] else 'NO'} (CMC_API_KEY)"),
        ("mode", escape(status["mode"]), status["mode"]),
        ("prompt", escape(status["prompt"]), status["prompt"]),
    ]


def print_status(view, status):
    for label, rich_text, plain_text in status_lines(status):
        if view.rich:
            view.console.print(f"  [dim]{label:<7}[/] {rich_text}")
        else:
            print(f"  {label:<7} {plain_text}")


def render_banner(view, status):
    """The shell's welcome screen: logo, what it is, the current model and
    keys, and where to start."""
    tips = [("/replay", "watch the headline failure: a saved real run, no keys needed"),
            ("/task mcp_eur_threshold", "the headline failure, graded on recorded data (no CMC credits)"),
            ("/report", "the published results, no keys needed"),
            ("/help", "every command")]
    if not view.rich:
        print(f"LITMUS {_version()} - {TAGLINE}\n")
        print_status(view, status)
        print("\nType a trading question, or /help. Try: " + " · ".join(cmd for cmd, _ in tips))
        return
    console = view.console
    if _can_draw_blocks():
        lines = logo_lines()
        width = max(len(line) for line in lines)
        logo = Text()
        for line in lines:
            for i, ch in enumerate(line):
                f = i / max(1, width - 1)
                r, g, b = (round(a + (z - a) * f) for a, z in zip(*_GRADIENT))
                logo.append(ch, style=f"bold #{r:02x}{g:02x}{b:02x}")
            logo.append("\n")
        console.print(logo, end="")
    else:
        console.print("[bold cyan]LITMUS[/]")
    console.print(f"{TAGLINE}  [dim]v{_version()}[/]\n")
    print_status(view, status)
    console.print("\n[bold]Ask a trading question[/], or type [bold cyan]/[/] for commands.")
    for cmd, text in tips:
        console.print(f"  [cyan]{cmd:<24}[/] [dim]{text}[/]")
    console.print()
