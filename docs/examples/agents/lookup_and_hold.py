"""
lookup_and_hold.py: a complete bring-your-own agent, and a baseline.

It looks up every coin the question names, cites each price it got, and
never trades. No model and no API key, so it runs anywhere:

    litmus-bench --agent "python docs/examples/agents/lookup_and_hold.py" --tasks tasks/mcp_benchmark.jsonl --trials 1

It is here to show the protocol end to end, in about 60 lines. It is also
the "cautious bot that never trades" baseline: on a suite where the right
answer is usually a trade, looking things up and holding scores badly.

To test your own agent, keep serve() and replace decide() with your loop:
pass task["tools"] to your model, and route each tool call it makes through
call_tool(name, arguments).
"""

import re

from litmus.agent_protocol import serve

# The coins this example understands, by ticker or name. A real agent would
# let its model read the question instead.
COINS = {"btc": "BTC", "bitcoin": "BTC", "eth": "ETH", "ethereum": "ETH", "ether": "ETH",
         "sol": "SOL", "solana": "SOL"}


def coins_in(question):
    """Tickers the question names, in order of appearance, no repeats."""
    return list(dict.fromkeys(COINS[w] for w in re.findall(r"[a-z]+", question.lower()) if w in COINS))


def price_of(symbol, tool_names, call_tool):
    """(price, tool it came from), or (None, tool) if the lookup failed."""
    if "get_crypto_quotes_latest" in tool_names:  # CMC's MCP tools: resolve the id first
        matches = call_tool("search_cryptos", {"query": symbol})
        coin = next((m for m in matches or [] if isinstance(m, dict) and m.get("symbol") == symbol), None)
        if coin is None:
            return None, "search_cryptos"
        quote = call_tool("get_crypto_quotes_latest", {"id": str(coin["id"])})
        rows = quote if isinstance(quote, list) else []
        return (rows[0].get("price") if rows else None), "get_crypto_quotes_latest"
    quote = call_tool("get_price", {"symbol": symbol})  # Litmus's REST tools
    return (quote.get("price") if isinstance(quote, dict) else None), "get_price"


def decide(task, call_tool):
    tool_names = {t["function"]["name"] for t in task["tools"]}
    coins = coins_in(task["question"]) or ["BTC"]
    evidence = []
    for symbol in coins:
        price, tool = price_of(symbol, tool_names, call_tool)
        if price is None:
            evidence.append({"claim": f"{symbol}'s price lookup failed", "source_tool": tool})
        else:
            evidence.append({"claim": f"{symbol} price in USD", "source_tool": tool, "value": price})
    print(f"looked up {coins}; holding", flush=True)  # goes to stderr: Litmus keeps it as the agent's log
    return {"action": "HOLD", "symbol": coins[0], "confidence": 0.5, "position_size_pct": 0, "leverage": 1,
            "time_horizon": "swing", "evidence": evidence,
            "reasoning": "Baseline agent: it looks the coins up and never opens a position."}


if __name__ == "__main__":
    serve(decide)
