"""Layer L2-A detectors. Each consumes the canonical L2 event stream
plus :class:`book_state.BookState` and emits zero or more derived
:class:`events.Event` instances.

The ``L2Detector`` protocol is documented inline in each detector
module — no need for a separate Protocol declaration here, since the
harness invokes detectors by their ``consume`` method directly.
"""
