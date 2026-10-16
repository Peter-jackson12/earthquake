"""T31 computation measurement helpers (memory sampling, flags, idle check, clock check).

The measured fit runs in its own single-threaded child process. A separate watcher (the orchestrator process)
samples the child's resident set size every 20 ms; the child reports interval boundaries in perf_counter
seconds, a clock that is shared between processes on one machine (checked by clock_common_check).
"""
from __future__ import annotations
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

MIB = 2 ** 20


class TreeRss:
    """rss of a process plus all of its descendants.

    On Windows a venv's python.exe is a launcher that starts the real interpreter as a child process, so the pid
    returned by Popen is a tiny stub; summing the tree measures the interpreter that runs the fit. Descendants are
    (re)discovered only every `refresh` seconds because psutil.children() scans all processes."""

    def __init__(self, pid, refresh=0.5, clock=time.monotonic):
        self.root = psutil.Process(pid)
        self.procs = {pid: self.root}
        self.refresh, self.clock = refresh, clock
        self._start = clock()
        self._last = -1e9

    def pids(self):
        return sorted(self.procs)

    def __call__(self):
        now = self.clock()
        every = self.refresh if now - self._start < 5.0 else 5.0
        if now - self._last >= every:
            self._last = now
            try:
                for ch in self.root.children(recursive=True):
                    self.procs.setdefault(ch.pid, ch)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        total = 0
        for pid, proc in list(self.procs.items()):
            try:
                total += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                if pid == self.root.pid:
                    raise
                del self.procs[pid]
        return total


class RssSampler:
    """Samples the rss of a pid (and its descendants) at a fixed interval from a daemon thread."""

    def __init__(self, pid, interval=0.02, clock=time.perf_counter, reader=None):
        self.pid, self.interval, self.clock = pid, interval, clock
        self._tree = None if reader else TreeRss(pid)
        self._reader = reader or self._tree
        self._stop = threading.Event()
        self._t, self._rss = [], []
        self._thread = threading.Thread(target=self._loop, name=f'rss-sampler-{pid}', daemon=True)
        self.error = None

    @property
    def sampled_pids(self):
        return self._tree.pids() if self._tree else [self.pid]

    def _loop(self):
        while not self._stop.is_set():
            try:
                now = self.clock()
                rss = int(self._reader())
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError) as exc:
                self.error = type(exc).__name__
                break
            self._t.append(now)
            self._rss.append(rss)
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        return np.asarray(self._t, dtype=np.float64), np.asarray(self._rss, dtype=np.int64)


def interval_increment(t, rss, start, end):
    """Peak sampled rss inside [start, end] minus the sample at (or just before) start.

    The baseline sample is included in the maximum, so the increment is never negative; an interval with no
    sample reports 0 and n_samples=0."""
    t, rss = np.asarray(t), np.asarray(rss)
    if t.size == 0 or end < start:
        return dict(baseline_bytes=None, peak_bytes=None, increment_bytes=0, n_samples=0)
    before = np.flatnonzero(t <= start)
    base_i = int(before[-1]) if before.size else 0
    inside = np.flatnonzero((t >= start) & (t <= end))
    base = int(rss[base_i])
    peak = max([base] + [int(rss[i]) for i in inside])
    return dict(baseline_bytes=base, peak_bytes=peak, increment_bytes=int(peak - base), n_samples=int(inside.size))


def memory_intervals(t, rss, marks, sampling_limited_below=50):
    """Per-stage and combined sampled increments from the child's perf_counter marks."""
    spec = dict(encode=('enc_start', 'fit_start'), train=('fit_start', 'fit_end'), predict=('fit_end', 'pred_end'),
                encode_train=('enc_start', 'fit_end'))
    out = {}
    for name, (a, b) in spec.items():
        rec = interval_increment(t, rss, marks[a], marks[b])
        rec['duration_seconds'] = float(marks[b] - marks[a])
        rec['increment_mib'] = rec['increment_bytes'] / MIB
        rec['sampling_limited'] = bool(rec['n_samples'] < sampling_limited_below)
        out[name] = rec
    return out


def peak_counters():
    """Process high-water marks (Windows peak_wset / peak_pagefile); zeros elsewhere."""
    info = psutil.Process().memory_info()
    return dict(rss=int(info.rss), peak_wset=int(getattr(info, 'peak_wset', 0) or 0),
                peak_pagefile=int(getattr(info, 'peak_pagefile', 0) or 0))


def wset_floor(after_load, after_stage):
    """Aux metric: increase of the peak working set over the value right after data loading.

    peak_wset can never decrease, so an increase of 0 or less means the stage stayed below the loading
    high-water mark and the value is a floor (FLOOR_LIMITED)."""
    inc = int(after_stage) - int(after_load)
    return dict(increment_bytes=inc, increment_mib=inc / MIB, floor_limited=bool(inc <= 0))


def idle_check(threshold=10.0, window=5.0, max_wait=60.0, cpu_percent=None, clock=time.monotonic):
    """Wait for system CPU <= threshold%; after max_wait proceed and mark noisy (never discard)."""
    cpu_percent = cpu_percent or (lambda w: psutil.cpu_percent(interval=w))
    start = clock()
    samples = []
    while True:
        value = float(cpu_percent(window))
        samples.append(value)
        if value <= threshold:
            return dict(noisy=False, samples=samples, waited_seconds=float(clock() - start))
        if clock() - start >= max_wait:
            return dict(noisy=True, samples=samples, waited_seconds=float(clock() - start))


def top_cpu_processes(n=10, window=1.0):
    procs = []
    for p in psutil.process_iter(['pid', 'name']):
        try:
            p.cpu_percent(None)
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(window)
    rows = []
    for p in procs:
        try:
            rows.append(dict(pid=p.pid, name=p.info.get('name'), cpu_percent=float(p.cpu_percent(None))))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows.sort(key=lambda r: -r['cpu_percent'])
    return rows[:n]


def clock_common_check():
    """perf_counter read by a freshly started child must lie between two parent reads (shared clock)."""
    before = time.perf_counter()
    out = subprocess.run([sys.executable, '-c', 'import time;print(repr(time.perf_counter()))'],
                         capture_output=True, text=True, check=True).stdout.strip()
    after = time.perf_counter()
    child = float(out)
    return dict(parent_before=before, child=child, parent_after=after, common=bool(before <= child <= after))


def coefficient_of_variation(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2 or float(v.mean()) == 0.0:
        return None
    return float(v.std(ddof=1) / v.mean())
