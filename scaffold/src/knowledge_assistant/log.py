import json
import logging
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

from knowledge_assistant.config import get_settings

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_configured = False

# Quiet noisy third-party loggers.
_NOISY_LOGGERS = {"pypdf": logging.ERROR, "httpx": logging.WARNING, "httpcore": logging.WARNING}


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def set_trace_id(tid: str) -> None:
    """Adopt a trace id from another process."""
    _trace_id.set(tid)


def current_trace_id() -> str:
    return _trace_id.get()


class _TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"trace_id"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", "-"),
            "message": record.getMessage(),
        }
        payload.update(
            {k: v for k, v in record.__dict__.items() if k not in self.RESERVED and not k.startswith("_")}
        )
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s")
    )
    console.addFilter(_TraceFilter())
    root.addHandler(console)

    log_dir = get_settings().log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    fileh = RotatingFileHandler(log_dir / "app.jsonl", maxBytes=5_000_000, backupCount=3)
    fileh.setFormatter(_JsonFormatter())
    fileh.addFilter(_TraceFilter())
    root.addHandler(fileh)

    for name, lvl in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
