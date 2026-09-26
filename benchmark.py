"""
benchmark.py
------------
Utilities for measuring and comparing sequential vs. parallel performance.

Definitions used throughout this project:

    Speedup    = T_sequential / T_parallel
                 (how many times faster the parallel version is)

    Efficiency = Speedup / Number_of_Workers
                 (what fraction of "perfect" linear speedup was achieved -
                 e.g. 4 workers giving exactly 4x speedup would be 100%
                 efficient; in practice overhead means it's usually less)
"""

import time
import multiprocessing

import sequential
import parallel


def time_sequential(arr, operation: str, params: dict):
    """Run and time the sequential implementation. Returns (result, seconds)."""
    start = time.perf_counter()
    result = sequential.process_sequential(arr, operation, params)
    elapsed = time.perf_counter() - start
    return result, elapsed


def time_parallel(arr, operation: str, params: dict, num_workers: int):
    """Run and time the parallel implementation. Returns (result, seconds)."""
    start = time.perf_counter()
    result = parallel.process_parallel(arr, operation, params, num_workers)
    elapsed = time.perf_counter() - start
    return result, elapsed


def compute_speedup(sequential_time: float, parallel_time: float) -> float:
    if parallel_time <= 0:
        return 0.0
    return sequential_time / parallel_time


def compute_efficiency(speedup: float, num_workers: int) -> float:
    if num_workers <= 0:
        return 0.0
    return speedup / num_workers


def run_comparison(arr, operation: str, params: dict, num_workers: int) -> dict:
    """Run BOTH sequential and parallel once each, for 'Compare Both' mode
    and for single ad-hoc benchmarks. Returns a dict with everything the UI
    needs to display.
    """
    seq_result, seq_time = time_sequential(arr, operation, params)
    par_result, par_time = time_parallel(arr, operation, params, num_workers)

    speedup = compute_speedup(seq_time, par_time)
    efficiency = compute_efficiency(speedup, num_workers)

    return {
        "sequential_result": seq_result,
        "parallel_result": par_result,
        "sequential_time": seq_time,
        "parallel_time": par_time,
        "speedup": speedup,
        "efficiency": efficiency,
        "workers": num_workers,
    }


def available_worker_options(candidates=(1, 2, 4, 8)):
    """Return only the worker counts from `candidates` that make sense on
    this machine (i.e. do not exceed the number of CPU cores available).
    """
    max_cores = multiprocessing.cpu_count()
    return [w for w in candidates if w <= max_cores] or [1]


def run_worker_sweep(arr, operation: str, params: dict, worker_counts=None) -> dict:
    """Benchmark the sequential version once, then the parallel version at
    every worker count in `worker_counts`, and return a full results table
    plus the raw sequential time (used as the speedup baseline).

    Nothing here is hard-coded - every number comes from an actual timed
    run on the image currently loaded in the app.
    """
    if worker_counts is None:
        worker_counts = available_worker_options()

    _, seq_time = time_sequential(arr, operation, params)

    rows = []
    for workers in worker_counts:
        _, par_time = time_parallel(arr, operation, params, workers)
        speedup = compute_speedup(seq_time, par_time)
        efficiency = compute_efficiency(speedup, workers)
        rows.append({
            "Workers": workers,
            "Time (s)": round(par_time, 4),
            "Speedup": round(speedup, 2),
            "Efficiency (%)": round(efficiency * 100, 1),
        })

    return {
        "sequential_time": seq_time,
        "rows": rows,
    }
