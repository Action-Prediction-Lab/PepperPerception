"""Shared helpers for the perception service guards.

Each guard runs as a script (exit 1 on failure) and under pytest (skipped when no service answers).
"""
import sys

import cv2
import numpy as np
import zmq

ADDRESS = "tcp://localhost:5557"
TIMEOUT_MS = 30000


def request(parts, address=ADDRESS, timeout_ms=TIMEOUT_MS):
    """Send one multipart REQ and return the decoded JSON reply, or None on timeout."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(address)
    try:
        sock.send_multipart(parts)
        return sock.recv_json()
    except zmq.Again:
        return None
    finally:
        sock.close()


def service_reachable():
    """True if the perception service answers a trivial request."""
    return request([encode(blank_image())], timeout_ms=5000) is not None


def blank_image(width=320, height=240):
    return np.zeros((height, width, 3), np.uint8)


def noise_image(width=320, height=240, seed=0):
    return np.random.RandomState(seed).randint(0, 255, (height, width, 3), dtype=np.uint8)


def encode(image):
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buffer.tobytes()


class Guard:
    """Accumulates named checks and fails once at the end with the full list."""

    def __init__(self, name):
        self.name = name
        self.failures = []

    def check(self, label, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures.append(label)

    def finish(self):
        if self.failures:
            raise AssertionError(
                f"{self.name}: {len(self.failures)} check(s) failed: {self.failures}")
        print(f"\n--- {self.name}: all checks passed ---")


def main(fn):
    """Run a guard function as a script, exiting 1 on failure."""
    try:
        fn()
    except AssertionError as exc:
        print(f"\n--- FAILURE ---\n{exc}")
        sys.exit(1)
