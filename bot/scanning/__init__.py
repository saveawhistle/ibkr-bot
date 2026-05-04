"""Find + qualify candidate symbols for the morning watchlist.

Owns the IBKR ``TOP_PERC_GAIN`` scan, the catalyst classifier
(keyword + symbol-attribution gates), the operator catalyst-override
mechanism, and the external HTTP clients (Finnhub, yfinance) used to
enrich candidate symbols with float + news data.
"""
