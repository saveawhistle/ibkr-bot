"""One-shot diagnostic: subscribe to ``reqMktDepth`` + tick-by-tick prints
for a single liquid symbol, capture every event for ``--duration`` seconds,
and dump field shapes + sample values to stdout.

The detector implementations in :mod:`bot.exit_advisor.market.l2_events` need
to consume a canonical input event model. This probe confirms the actual
shape ``ib_async`` produces against live IBKR before we commit detectors
to a particular schema. Phase 9.4 found bar data had quirks the docs
didn't telegraph; assume L2 may too.

Usage:
    # Default: SPY for 30 seconds, paper TWS, NASDAQ TotalView eligible.
    python scripts/probe_reqmktdepth.py

    # Custom symbol / window:
    python scripts/probe_reqmktdepth.py --symbol AAPL --duration 60

This script is read-only:
- No orders placed
- No persistent state written (cache, config, manifests untouched)
- Cleanly disconnects after the duration window or on Ctrl+C

The output is intended for the operator to paste back into the spike
conversation so the detector input model can be designed against the
real shape, not the documented one.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

DEFAULT_SYMBOL = "SPY"
DEFAULT_DURATION_SECONDS = 30
DEFAULT_NUM_DEPTH_ROWS = 10
SAMPLE_RAW_EVENTS = 10
"""How many raw events of each kind to dump verbatim. Enough that the
operator can spot any non-obvious field while keeping the output
readable when SPY is firing thousands of updates."""


@dataclass
class FieldObservation:
    """One field's observed types + a sample value, accumulated across
    many events. Only the FIRST sample value is kept; the goal is shape,
    not statistics."""

    types: set[str] = field(default_factory=set)
    sample: Any = None
    sample_set: bool = False

    def record(self, value: Any) -> None:
        self.types.add(type(value).__name__)
        if not self.sample_set:
            self.sample = value
            self.sample_set = True


@dataclass
class ProbeCapture:
    """Mutable state the probe accumulates during the capture window."""

    depth_class_names: Counter[str] = field(default_factory=Counter)
    depth_field_observations: dict[str, FieldObservation] = field(default_factory=dict)
    depth_total: int = 0
    depth_raw_samples: list[dict[str, Any]] = field(default_factory=list)

    print_class_names: Counter[str] = field(default_factory=Counter)
    print_field_observations: dict[str, FieldObservation] = field(default_factory=dict)
    print_total: int = 0
    print_raw_samples: list[dict[str, Any]] = field(default_factory=list)

    def record_depth(self, obj: Any) -> None:
        self.depth_total += 1
        self.depth_class_names[type(obj).__name__] += 1
        snapshot = self._snapshot_object(obj, self.depth_field_observations)
        if len(self.depth_raw_samples) < SAMPLE_RAW_EVENTS:
            self.depth_raw_samples.append(snapshot)

    def record_print(self, obj: Any) -> None:
        self.print_total += 1
        self.print_class_names[type(obj).__name__] += 1
        snapshot = self._snapshot_object(obj, self.print_field_observations)
        if len(self.print_raw_samples) < SAMPLE_RAW_EVENTS:
            self.print_raw_samples.append(snapshot)

    @staticmethod
    def _snapshot_object(
        obj: Any, observations: dict[str, FieldObservation]
    ) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of ``obj``'s public fields,
        and mutate ``observations`` to record each field's type + first
        sample value. Tolerates non-serializable values by stringifying."""
        out: dict[str, Any] = {"_class": type(obj).__name__}
        for name in sorted(_public_attrs(obj)):
            try:
                value = getattr(obj, name)
            except Exception as exc:  # noqa: BLE001 — best-effort introspection
                value = f"<error: {exc!r}>"
            observations.setdefault(name, FieldObservation()).record(value)
            out[name] = _stringify(value)
        return out


def _public_attrs(obj: Any) -> list[str]:
    """Public, non-callable attributes of ``obj``. Skips dunders, methods,
    and ib_async event-emitter slots (which aren't useful here)."""
    out: list[str] = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001
            continue
        if callable(value):
            continue
        # ib_async Event objects (suffix "Event") are emitter slots; skip.
        if name.endswith("Event"):
            continue
        out.append(name)
    return out


def _stringify(value: Any) -> Any:
    """Reduce ``value`` to something json.dumps can handle. Leaves
    primitives intact, formats datetimes as ISO-8601, repr()s anything
    else (so the operator can read but the snapshot stays serializable)."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_stringify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    return repr(value)


async def _run_probe(symbol: str, duration: int, num_rows: int) -> ProbeCapture:
    """Subscribe to depth + tick-by-tick prints, capture for ``duration``
    seconds, return the accumulated capture."""
    from bot.brokerage.ibkr_client import IBKRClient

    capture = ProbeCapture()
    log = logging.getLogger("probe_reqmktdepth")

    client = IBKRClient()
    await client.connect()
    try:
        contract = await client.qualify_stock(symbol)
        log.info("qualified %s (conId=%s)", symbol, getattr(contract, "conId", "?"))

        depth_ticker = client.ib.reqMktDepth(contract, numRows=num_rows)
        prints_ticker = client.ib.reqTickByTickData(
            contract, "AllLast", numberOfTicks=0, ignoreSize=False
        )

        def _on_depth_update(t: Any) -> None:
            # ib_async exposes the per-update objects in ``t.domTicks``;
            # the list is cleared between TCP batches so iterate inline.
            for entry in getattr(t, "domTicks", []):
                capture.record_depth(entry)

        def _on_prints_update(t: Any) -> None:
            for tick in getattr(t, "tickByTicks", []):
                capture.record_print(tick)

        depth_ticker.updateEvent += _on_depth_update
        prints_ticker.updateEvent += _on_prints_update

        log.info(
            "capturing depth + prints for %s seconds (%s rows of depth requested)",
            duration,
            num_rows,
        )

        # Sleep with a graceful Ctrl+C escape — the operator can abort early
        # without leaving the subscription dangling.
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            log.info("capture cancelled — disconnecting")

        # Cancel the subscriptions before disconnect so IBKR releases the
        # market-data line cleanly.
        try:
            client.ib.cancelMktDepth(contract)
        except Exception as exc:  # noqa: BLE001
            log.warning("cancelMktDepth raised: %s", exc)
        try:
            client.ib.cancelTickByTickData(contract, "AllLast")
        except Exception as exc:  # noqa: BLE001
            log.warning("cancelTickByTickData raised: %s", exc)

    finally:
        await client.disconnect()

    return capture


def _print_section(title: str, lines: list[str]) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    for ln in lines:
        print(ln)


def _print_field_table(observations: dict[str, FieldObservation]) -> None:
    if not observations:
        print("  (no fields observed)")
        return
    name_width = max(len(name) for name in observations)
    type_width = max(
        len(", ".join(sorted(o.types))) for o in observations.values()
    ) if observations else 0
    type_width = max(type_width, len("type(s)"))
    print(f"  {'field'.ljust(name_width)}  {'type(s)'.ljust(type_width)}  sample")
    print(f"  {'-' * name_width}  {'-' * type_width}  ------")
    for name in sorted(observations):
        obs = observations[name]
        types_str = ", ".join(sorted(obs.types)) or "—"
        sample_repr = json.dumps(_stringify(obs.sample), default=str)
        if len(sample_repr) > 80:
            sample_repr = sample_repr[:77] + "..."
        print(f"  {name.ljust(name_width)}  {types_str.ljust(type_width)}  {sample_repr}")


def report(capture: ProbeCapture, symbol: str, duration: int) -> None:
    print(f"\nProbe: reqMktDepth + reqTickByTickData('AllLast') for {symbol}")
    print(f"Duration: {duration}s")
    print(f"Captured at: {datetime.now(UTC).isoformat(timespec='seconds')}")

    _print_section(
        "DEPTH SUMMARY",
        [
            f"Total depth events: {capture.depth_total}",
            f"ib_async class name(s): {dict(capture.depth_class_names)}",
            "",
            "Field shape:",
        ],
    )
    _print_field_table(capture.depth_field_observations)

    _print_section(
        "PRINT SUMMARY",
        [
            f"Total print events: {capture.print_total}",
            f"ib_async class name(s): {dict(capture.print_class_names)}",
            "",
            "Field shape:",
        ],
    )
    _print_field_table(capture.print_field_observations)

    _print_section(
        f"FIRST {SAMPLE_RAW_EVENTS} RAW DEPTH EVENTS",
        [
            json.dumps(s, indent=2, default=str)
            for s in capture.depth_raw_samples
        ],
    )
    _print_section(
        f"FIRST {SAMPLE_RAW_EVENTS} RAW PRINT EVENTS",
        [
            json.dumps(s, indent=2, default=str)
            for s in capture.print_raw_samples
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help="Seconds to capture (default 30).",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=DEFAULT_NUM_DEPTH_ROWS,
        help="Depth-of-book rows to request (default 10).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # SIGINT handling: let the inner await asyncio.sleep raise CancelledError,
    # which the probe catches and treats as "cancel early".
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _on_sigint(*_: Any) -> None:
        for task in asyncio.all_tasks(loop):
            task.cancel()

    if hasattr(signal, "SIGINT"):
        # Windows asyncio loops don't support signal handlers; the
        # default Ctrl+C handling propagates KeyboardInterrupt instead,
        # which we catch below.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal.SIGINT, _on_sigint)

    try:
        capture = loop.run_until_complete(
            _run_probe(args.symbol, args.duration, args.num_rows)
        )
    except KeyboardInterrupt:
        print("\ninterrupted — partial capture follows", file=sys.stderr)
        return 1
    finally:
        loop.close()

    report(capture, args.symbol, args.duration)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
