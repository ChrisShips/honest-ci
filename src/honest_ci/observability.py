"""Optional OpenTelemetry spans around gate execution.

Telemetry is a soft dependency: without the ``otel`` extra installed the span
helper degrades to a no-op so a private deployment never needs a collector.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_tracer: Any | None = None

try:
    from opentelemetry import trace as _otel

    _tracer = _otel.get_tracer("honest-ci")
except Exception:  # noqa: BLE001 - telemetry must never break a gate run.
    pass


@contextmanager
def span(name: str, **attrs: object) -> Iterator[None]:
    start = time.perf_counter()
    if _tracer:
        with _tracer.start_as_current_span(name) as active_span:
            for key, value in attrs.items():
                active_span.set_attribute(key, str(value))
            try:
                yield
            finally:
                active_span.set_attribute("latency_s", round(time.perf_counter() - start, 3))
    else:
        yield
