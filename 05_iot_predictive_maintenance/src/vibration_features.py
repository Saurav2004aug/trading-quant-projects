"""
Python mirror of the feature maths in firmware/sensor_stream.ino, so the
algorithm can be unit-tested on known signals (the firmware itself needs
the ESP32 toolchain to compile).

Input: an (N, 3) burst of accelerometer samples in m/s^2.
"""
from __future__ import annotations

import numpy as np

G = 9.80665


def vibration_features(acc: np.ndarray) -> dict:
    ac = acc - acc.mean(axis=0)                  # remove gravity + offset per axis
    m2 = (ac**2).sum(axis=1)                     # squared magnitude of AC vector
    ms = m2.mean()
    peak = m2.max()
    return {
        "rms_g": float(np.sqrt(ms) / G),
        "peak_g": float(np.sqrt(peak) / G),
        "crest": float(np.sqrt(peak / ms)) if ms > 0 else 0.0,
        "kurtosis": float((m2**2).mean() / ms**2) if ms > 0 else 0.0,
    }


def simulated_burst(fs: float = 1000.0, n: int = 1024, amp_g: float = 0.2, freq: float = 50.0,
                    gravity_axis=(0.3, 0.5, 0.81), noise_g: float = 0.0, impulses: int = 0,
                    seed: int = 0) -> np.ndarray:
    """Test signal: sinusoidal vibration on the x axis, gravity in an arbitrary
    orientation, optional white noise and periodic impulses (bearing-like)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    gdir = np.asarray(gravity_axis, float)
    acc = np.tile(gdir / np.linalg.norm(gdir) * G, (n, 1))
    acc[:, 0] += amp_g * G * np.sin(2 * np.pi * freq * t)
    acc += rng.normal(0, noise_g * G, acc.shape)
    if impulses:
        idx = np.linspace(0, n - 1, impulses).astype(int)
        acc[idx, 1] += 3 * amp_g * G
    return acc
