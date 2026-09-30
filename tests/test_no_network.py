"""The suite's promise that no test can spend API quota, checked rather than
assumed."""

import os

import pytest

from litmus import agent, market
from conftest import NetworkBlocked


def test_real_keys_are_never_loaded():
    assert os.environ["GEMINI_API_KEY"] == "test-key-never-sent"
    assert os.environ["CMC_API_KEY"] == "test-key-never-sent"


def test_unscripted_model_call_is_blocked():
    with pytest.raises(NetworkBlocked):
        agent.client.chat.completions.create(model=agent.MODEL, messages=[], tools=[])


def test_market_http_is_blocked():
    with pytest.raises(NetworkBlocked):
        market.get_price("BTC")
