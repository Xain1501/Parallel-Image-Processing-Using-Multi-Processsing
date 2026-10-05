"""
parallel.py
-----------
The multi-process implementation. This is the heart of the PDC (Parallel &
Distributed Computing) demonstration in this project.

We use Python's built-in `multiprocessing` module (NOT threads) because
CPython has a Global Interpreter Lock (GIL): only one thread can execute
Python bytecode at a time, no matter how many CPU-bound threads you start.
`multiprocessing` sidesteps the GIL entirely by starting whole separate
OS processes, each with its own Python interpreter and its own GIL, which
the operating system can schedule onto different CPU cores simultaneously.

The workflow for every operation except Resize is:

    Original Image
          |
          v
    Split into N row-bands, each grown by a "halo" of extra rows
    (see image_operations.kernel_halo_radius / the module docstring there)
          |
    +-----+-----+-----+-----+
    v     v     v     v     v
  Work1 Work2 Work3 Work4  ...   <- separate OS processes, run concurrently
    |     |     |     |     |
    v     v     v     v     v
  Result Result Result Result    <- each worker crops its own halo back off
    +-----+-----+-----+-----+
          |
          v
   np.vstack(...) -> Final Image

Resize is handled slightly differently (see _process_parallel_resize below)
because "resize" is not a kernel/convolution operation - there's no fixed
"halo radius" for it in the same sense. That difference, and its
consequences, are explained in the README and in image_operations.resize_band.
"""

from multiprocessing import Pool
import numpy as np

import sequential


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def _row_boundaries(height: int, num_chunks: int):
    """Split `height` rows into `num_chunks` contiguous, non-overlapping
    bands as evenly as possible. Returns a list of (start_row, end_row).
    """
    num_chunks = max(1, min(num_chunks, height))
    base = height // num_chunks
    boundaries = []
    start = 0
    for i in range(num_chunks):
        end = start + base if i < num_chunks - 1 else height
        boundaries.append((start, end))
        start = end
    return boundaries


def get_chunk_boundaries(height: int, num_workers: int):
    """Public helper: the exact row boundaries each worker will be assigned
    for an image of the given height and worker count. Used by app.py to
    draw a visual overlay showing how the image was actually split, so the
    UI can *show* the chunking rather than describe it in text.
    """
    num_chunks = max(1, min(num_workers, height))
    return _row_boundaries(height, num_chunks)


def _split_with_halo(arr: np.ndarray, num_chunks: int, halo: int):
    """Split an image into `num_chunks` row-bands, each grown by `halo`
    extra rows above/below (clamped at the image edges).

    Returns a list of tuples: (chunk_with_halo, crop_top, crop_bottom)
    where chunk_with_halo[crop_top:crop_bottom] gives back exactly the rows
    belonging to that chunk's TRUE (non-overlapping) boundary, once the
    chunk has been processed.
    """
    height = arr.shape[0]
    boundaries = _row_boundaries(height, num_chunks)
    chunks = []
    for (start, end) in boundaries:
        halo_start = max(0, start - halo)
        halo_end = min(height, end + halo)
        chunk = arr[halo_start:halo_end]
        crop_top = start - halo_start
        crop_bottom = crop_top + (end - start)
        chunks.append((chunk, crop_top, crop_bottom))
    return chunks


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------
# These MUST be top-level (module-level) functions, not lambdas or methods,
# because multiprocessing.Pool needs to "pickle" (serialize) them to send
# to each worker process.

def _worker(args):
    """Run inside a worker process: process one (already-halo-padded)
    chunk using the exact same logic as sequential mode.
    """
    chunk, operation, params = args
    return sequential.process_sequential(chunk, operation, params)


def _resize_worker(args):
    """Run inside a worker process: resize one horizontal band."""
    import image_operations as ops
    band, new_width, new_height = args
    return ops.resize_band(band, new_width, new_height)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_parallel(arr: np.ndarray, operation: str, params: dict, num_workers: int):
    """Process a full image using `num_workers` worker processes.

    This is the parallel counterpart to sequential.process_sequential(),
    with the same signature plus `num_workers`.
    """
    num_workers = max(1, int(num_workers))

    if operation == "Resize":
        return _process_parallel_resize(arr, params, num_workers)

    halo = ops_halo_radius(operation, params)
    height = arr.shape[0]
    num_chunks = max(1, min(num_workers, height))

    chunk_specs = _split_with_halo(arr, num_chunks, halo)
    tasks = [(chunk, operation, params) for (chunk, _, _) in chunk_specs]

    # --- Step: Distribute + Process (this is the actual parallel part) ---
    with Pool(processes=num_chunks) as pool:
        results = pool.map(_worker, tasks)

    # --- Step: Synchronize + Combine ---
    # Pool.map() already blocks until every worker has finished (this is the
    # synchronization point - the main process waits here for all workers).
    cropped = [
        result[crop_top:crop_bottom]
        for result, (_, crop_top, crop_bottom) in zip(results, chunk_specs)
    ]
    return np.vstack(cropped)


def _process_parallel_resize(arr: np.ndarray, params: dict, num_workers: int):
    """Parallel resize: split the image into independent horizontal bands,
    resize each band on its own worker process, then stack the results.

    See image_operations.resize_band for a full explanation of why this is
    a reasonable-but-not-perfectly-seamless approach, and what its limits
    are.
    """
    scale = float(params.get("scale", 100)) / 100.0
    height, width = arr.shape[:2]
    num_chunks = max(1, min(num_workers, height))

    boundaries = _row_boundaries(height, num_chunks)
    bands = [arr[start:end] for (start, end) in boundaries]
    new_width = max(1, int(round(width * scale)))

    # The TRUE target total height is whatever a whole-image resize would
    # produce (same formula as image_operations.resize_whole - including
    # its own "at least 1 pixel" floor). Every band boundary is then mapped
    # into that exact target space via `row * target_h / height`, rounded.
    # Because this is one continuous, monotonic mapping from [0, height] to
    # [0, target_h], the boundaries always start at 0 and end at exactly
    # target_h - so the bands' heights always sum to precisely target_h,
    # matching resize_whole exactly, no matter how many workers are used or
    # how small the image/scale is. Individual bands are allowed to round
    # down to 0 rows in this process (handled by image_operations.resize_band)
    # rather than being padded to a minimum of 1, which would inflate the
    # combined total above the correct target height.
    target_h = max(1, int(round(height * scale)))
    new_heights = []
    for (start, end) in boundaries:
        new_start = int(round(start * target_h / height)) if height > 0 else 0
        new_end = int(round(end * target_h / height)) if height > 0 else 0
        new_heights.append(new_end - new_start)

    tasks = [
        (band, new_width, new_h) for band, new_h in zip(bands, new_heights)
    ]

    with Pool(processes=num_chunks) as pool:
        results = pool.map(_resize_worker, tasks)

    combined = np.vstack(results)

    # Extremely rare edge case: if every band rounded down to 0 rows (can
    # only happen on a pathologically tiny image at a very small scale),
    # fall back to a direct whole-image resize so the output is never
    # completely empty.
    if combined.shape[0] == 0:
        import image_operations as ops
        return ops.resize_whole(arr, scale)

    return combined


def ops_halo_radius(operation: str, params: dict) -> int:
    """Thin wrapper kept in this module so parallel.py's splitting logic
    doesn't need a direct top-level import cycle; delegates to
    image_operations.kernel_halo_radius.
    """
    import image_operations as ops
    return ops.kernel_halo_radius(operation, params)
