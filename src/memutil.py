"""OS-level memory safety: query ACTUAL available RAM (not a guessed fixed cap) and refuse/bail before an
allocation would drive the system into a swap-death crash. Complements the systemd memguard (which hard-kills
at a fixed limit) — this lets code SEE real headroom and adapt (stream/offload/bail cleanly) instead.

Usage:
    from memutil import available_gb, ensure_ram, ram_ok
    ensure_ram(need_bytes, headroom_gb=8, label="down_proj Grams")   # raises BEFORE allocating if it won't fit
    if not ram_ok(headroom_gb=6): ...offload/flush...                # periodic check inside accumulation loops
"""
import os


def available_bytes():
    """OS-reported available RAM in bytes (psutil preferred; /proc/meminfo MemAvailable fallback; None if unknown)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            return None
    return None


def available_gb():
    b = available_bytes()
    return None if b is None else b / 1e9


def ensure_ram(need_bytes, headroom_gb=8.0, label=""):
    """Raise MemoryError BEFORE a big allocation if need_bytes + headroom exceeds OS-available RAM.
    Prefer this over trusting a fixed cap: a clean early raise beats a system-freezing OOM."""
    avail = available_bytes()
    if avail is None:
        return
    need = need_bytes + headroom_gb * 1e9
    if need > avail:
        raise MemoryError(
            f"[memutil] refusing to allocate {need_bytes/1e9:.1f}G for '{label}': only {avail/1e9:.1f}G "
            f"OS-available (want {headroom_gb:.0f}G headroom). Reduce the working set (stream/offload/chunk).")


def ram_ok(headroom_gb=6.0):
    """True while OS-available RAM stays above headroom. Call inside accumulation loops to spill/bail early
    rather than crashing the host."""
    avail = available_bytes()
    return avail is None or avail > headroom_gb * 1e9


def require_ram_ok(headroom_gb=6.0, label=""):
    if not ram_ok(headroom_gb):
        raise MemoryError(f"[memutil] OS-available RAM below {headroom_gb:.0f}G headroom during '{label}' "
                          f"({available_gb():.1f}G left) — bailing before a host OOM.")
