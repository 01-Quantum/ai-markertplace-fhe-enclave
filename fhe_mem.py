import logging
import os
import sys
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger("fhe_vault")

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _rss_mb() -> float:
    """Current resident set size of this process, in MB."""
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * _PAGE_SIZE / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        try:
            import resource

            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return maxrss / (1024 * 1024) if sys.platform == "darwin" else maxrss / 1024
        except Exception:
            return 0.0


@contextmanager
def log_memory(operation: str, interval: float = 0.5):
    """
    Log process memory before/after an operation and sample the peak during it.
    Yields a stats dict with elapsed_ms (and memory fields) filled on exit.
    """
    stats: dict[str, float | int | None] = {
        "elapsed_ms": None,
        "peak_mb": None,
        "before_mb": None,
        "after_mb": None,
    }
    before = _rss_mb()
    logger.info("[mem] %s start: rss=%.1f MB", operation, before)

    peak = before
    stop = threading.Event()

    def _sample() -> None:
        nonlocal peak
        while not stop.wait(interval):
            peak = max(peak, _rss_mb())

    sampler = threading.Thread(target=_sample, name=f"mem-{operation}", daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        yield stats
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        sampler.join(timeout=1.0)
        after = _rss_mb()
        peak = max(peak, after)
        stats["elapsed_ms"] = int(round(elapsed * 1000))
        stats["before_mb"] = round(before, 1)
        stats["peak_mb"] = round(peak, 1)
        stats["after_mb"] = round(after, 1)
        logger.info(
            "[mem] %s done: took=%.2fs before=%.1f MB peak=%.1f MB after=%.1f MB (delta=%.1f MB)",
            operation,
            elapsed,
            before,
            peak,
            after,
            after - before,
        )
