"""Central logging configuration for OMOP Atlas Agent.

Each CLI entry point calls :func:`setup_logging` once at the top of ``main()``.
Module code obtains its logger with the standard
``logger = logging.getLogger(__name__)`` pattern, so log records carry the
dotted module path and downstream filters / handlers can act on it.

The default level is ``INFO``, overridable via the ``LOG_LEVEL`` environment
variable (``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR`` | ``CRITICAL``).

A future migration to OpenTelemetry log export can wire an OTLP handler into
the root logger here; the rest of the codebase will pick it up unchanged.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def setup_logging(level: str | int | None = None) -> None:
    """Configure the root logger once. Idempotent — safe to call from every entry point."""
    global _configured
    if _configured:
        return

    resolved_level = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(level=resolved_level, format=_DEFAULT_FORMAT)
    _configured = True
