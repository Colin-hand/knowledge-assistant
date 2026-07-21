"""In-process per-request stage tracker for live UI progress."""

import time

_STAGES: dict[str, tuple[str, float]] = {}
_TTL_S = 600


def set_stage(progress_id: str | None, stage: str) -> None:
    if not progress_id:
        return
    now = time.time()
    for pid, (_, ts) in list(_STAGES.items()):
        if now - ts > _TTL_S:
            del _STAGES[pid]
    _STAGES[progress_id] = (stage, now)


def get_stage(progress_id: str) -> str:
    entry = _STAGES.get(progress_id)
    return entry[0] if entry else ""
