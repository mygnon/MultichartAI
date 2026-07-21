"""pause_gate.py -- console pause/resume toggle for the long pipeline runs.

Press 'p' in the (elevated) console window where the pipeline log scrolls to
schedule a pause; the run actually pauses at the NEXT SAFE POINT (between MC64
operations, never mid-dialog). Press 'p' again to resume (or to cancel a pause
that has not taken effect yet).

Integration: mc_automation calls pause_gate.checkpoint() at the entry of its
MC64-operation entry points. The keyboard listener thread is started lazily on
the first checkpoint() call; if no interactive console is available (tests,
piped stdin), everything degrades to a no-op.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_paused = threading.Event()          # set => pause requested / active
_lock = threading.Lock()
_listener_started = False
_listener_ok = False


def _banner(msg: str) -> None:
    # print() + logger so the message is visible even between log lines
    line = f"\n######## {msg} ########"
    try:
        print(line, flush=True)
    except Exception:
        pass
    logger.info(msg)


def _listener() -> None:
    import msvcrt
    while True:
        try:
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"p", b"P"):
                    if _paused.is_set():
                        _paused.clear()
                        _banner("RESUME requested -- continuing")
                    else:
                        _paused.set()
                        _banner("PAUSE scheduled -- will pause at the next safe point "
                                "(current MC64 operation finishes first); press 'p' again to resume")
                # all other keys ignored
        except Exception:
            pass
        time.sleep(0.2)


def _ensure_listener() -> None:
    global _listener_started, _listener_ok
    with _lock:
        if _listener_started:
            return
        _listener_started = True
        try:
            import msvcrt
            msvcrt.kbhit()  # raises/fails if no console attached
            t = threading.Thread(target=_listener, name="pause-gate", daemon=True)
            t.start()
            _listener_ok = True
            logger.info("[pause] press 'p' in this window to pause/resume between runs")
        except Exception:
            _listener_ok = False  # no console -> permanent no-op


def checkpoint() -> None:
    """Called at safe points; blocks while a pause is active."""
    _ensure_listener()
    if not _listener_ok or not _paused.is_set():
        return
    t0 = time.time()
    _banner("PAUSED at safe point -- press 'p' in this window to resume")
    while _paused.is_set():
        time.sleep(0.5)
    _banner(f"RESUMED (paused {time.time() - t0:.0f}s)")
