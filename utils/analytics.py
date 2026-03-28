import logging
import os
from typing import Any

_log = logging.getLogger("openrouter_ai.analytics")
_log.propagate = False

if not _log.handlers:
    if os.getenv("OPENROUTER_ANALYTICS_LOG", "").lower() in ("1", "true", "yes"):
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
        _log.addHandler(_h)
        _log.setLevel(logging.INFO)
    else:
        _log.addHandler(logging.NullHandler())


def event_track(event: str, payload: dict):
    _log.info("%s %s", event, payload)


class AnalyticsEngine:
    """Records pipeline events; delegates to logging. Swap for DB / metrics backend as needed."""

    def record_event(self, **kwargs: Any) -> None:
        payload = {k: getattr(v, "value", v) if hasattr(v, "value") else v for k, v in kwargs.items()}
        event_track("pipeline_request", payload)
