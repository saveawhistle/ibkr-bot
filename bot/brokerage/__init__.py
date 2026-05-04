"""IBKR connection + market data plumbing.

Layer that owns everything the rest of the bot uses to talk to TWS:
the async ``IBKRClient`` connection wrapper, live + historical bar
subscriptions, and the 5-sec → 1-min in-process aggregator.
"""
