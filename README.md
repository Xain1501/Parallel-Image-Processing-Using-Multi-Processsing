# Parallel Image Processing System

A beginner-friendly Parallel & Distributed Computing (PDC) project that
processes images two ways — **sequentially** (one CPU core) and **in
parallel** (multiple CPU cores, via Python's `multiprocessing` module) — and
lets you directly compare their execution time, speedup, and efficiency.

---

## 1. Introduction

Image processing (blurring, sharpening, detecting edges, resizing) is
computationally expensive: every pixel needs to be visited, and often every
neighboring pixel too. This makes it a great, visual way to demonstrate a
core idea from Parallel & Distributed Computing: **data parallelism** —
splitting a large workload into independent pieces and processing those
pieces at the same time, on different CPU cores.

This project is a Streamlit web app where you upload an image, choose an
operation, and watch the sequential and parallel versions race.

> **App hanging or printing a huge wall of import errors ending in
> `KeyboardInterrupt` when you click Process/Benchmark on Windows?** This
> was a real bug caught and fixed in this project — see "Windows +
> multiprocessing note" under Section 18 for the full explanation of what
> was happening and how `app.py` now avoids it.

> **Tested status:** every operation (Grayscale, Blur, Edge Detection,
> Sharpen, Resize) has been run through the full app — upload, every
> processing mode (Sequential / Parallel / Compare Both), and the
> benchmark sweep — using Streamlit's own automated `AppTest` framework,
> with zero exceptions and zero errors across all 15
> operation × mode combinations. Sequential and parallel outputs were
> also diffed pixel-by-pixel: Grayscale, Blur, Edge Detection, and
> Sharpen are pixel-**identical** between sequential and parallel at
> every worker count (1, 2, 3, 4, 8) and on tiny (1x1, 1xN), odd-sized,
> and normal images alike. Resize matches sequential output's exact
> dimensions in every case, with the small pixel differences described
> in Section 10 below. Invalid/corrupted uploads are caught and shown
> as a friendly in-app error rather than crashing.

## 2. Problem Statement

A single CPU core can only do one thing at a time. As images get larger,
sequential (single-core) processing takes proportionally longer. Modern
computers have multiple CPU cores sitting idle unless software is explicitly
written to use them. This project answers: **how much faster can we make
image processing by spreading the work across multiple CPU cores, and what
are the limits of doing so?**

## 3. Objectives

- Implement 5 image operations: Grayscale, Blur, Edge Detection, Sharpen,
  Resize.
- Implement each operation both sequentially and in parallel.
- Correctly handle the "halo" / boundary problem that arises when you split
  an image for operations that depend on neighboring pixels.
- Measure real execution time, and compute **Speedup** and **Efficiency**
  from real, non-hard-coded measurements.
- Present everything in a clear, visual Streamlit interface.
- Clearly explain the underlying PDC concepts (data parallelism, the GIL,
  overhead, diminishing returns) rather than just showing numbers.

## 4. Technologies

- **Python 3.11+**
- **Pillow** — image loading/saving/resizing
- **NumPy** — fast array math (convolutions, pixel math)
- **Streamlit** — the web UI
- **multiprocessing** (Python standard library) — the actual parallelism
- **matplotlib** — benchmark charts

No AI/ML/deep learning and no external APIs are used anywhere in this
project — every operation is implemented from scratch using plain array
math.

## 5. Architecture

```text
parallel-image-processing/
│
├── app.py                <- Streamlit UI (entry point: `streamlit run app.py`)
├── sequential.py          <- Single-process ("baseline") implementation
├── parallel.py            <- Multi-process implementation (multiprocessing.Pool)
├── image_operations.py    <- The actual pixel math for all 5 operations
├── benchmark.py            <- Timing, speedup, efficiency, worker-count sweep
├── requirements.txt
├── README.md
└── sample_images/          <- A few generated test images to try the app with
```

**File responsibilities:**

- **`image_operations.py`** — the "what": pure functions that take a NumPy
  image array and return a processed NumPy image array. Contains
  Grayscale, Blur (Gaussian), Edge Detection (Sobel), Sharpen, and Resize,
  plus the convolution and halo-radius helpers they share. Neither
  sequential.py nor parallel.py contain any pixel math themselves — they
  only call into this file.
- **`sequential.py`** — dispatches to the right `image_operations` function
  for the whole image, on the current process, single-core.
- **`parallel.py`** — splits the image into chunks (with the correct halo
  for the chosen operation), hands each chunk to a `multiprocessing.Pool`
  worker (which in turn calls the same `sequential.process_sequential()`
  function — just on a smaller chunk), waits for every worker to finish,
  crops off the borrowed halo rows, and stitches the chunks back together.
- **`benchmark.py`** — wraps calls to `sequential.py` / `parallel.py` in
  `time.perf_counter()` timers and computes Speedup / Efficiency. Also runs
  the "sweep" across multiple worker counts for the benchmark chart.
- **`app.py`** — the Streamlit UI only. Collects user input, calls the
  three modules above, and displays images/metrics/charts. Contains no
  image-processing or multiprocessing logic of its own.

## 6. Data Parallelism (the core PDC concept demonstrated here)

> **Data parallelism** means dividing a large dataset into smaller portions
> and having multiple workers perform the **same operation** on different
> portions **simultaneously**.

```text
Large Image
     |
     v
Split into chunks
     |
Worker 1 -> Chunk 1
Worker 2 -> Chunk 2
Worker 3 -> Chunk 3
Worker 4 -> Chunk 4
     |
     v
Combine results
```

Every worker process in this project runs the **exact same function**
(e.g. `blur()`) — they just each see a different horizontal slice of the
image. This is what makes it data parallelism rather than task parallelism
(where different workers would do genuinely different jobs).

## 7. Sequential Processing

`sequential.py` calls the relevant `image_operations` function once, on the
whole image, in the single Python process that Streamlit itself is running
in:

```python
start = time.perf_counter()
result = process_sequential(image, "grayscale", {})
end = time.perf_counter()
execution_time = end - start
```

This is the baseline every parallel run is compared against.

## 8. Parallel Processing

`parallel.py` uses:

```python
from multiprocessing import Pool
```

The workflow:

```text
                 Original Image
                       |
                       v
                 Split Image
                       |
        +--------------+--------------+
        v              v              v
     Worker 1       Worker 2       Worker 3
        v              v              v
     Process        Process        Process
        +--------------+--------------+
                       v
                  Combine Chunks
                       |
                       v
                Final Image
```

You can choose **1, 2, 4, or 8 workers** in the UI — the app automatically
hides worker counts above what `multiprocessing.cpu_count()` reports for
your machine, since requesting more workers than physical cores just adds
overhead without adding real parallelism.

## 9. Why `multiprocessing` and Not Threads

Python (CPython) has a **Global Interpreter Lock (GIL)**: only one thread
can execute Python bytecode at any given instant, no matter how many
threads you start or how many CPU cores your machine has. This makes plain
Python *threads* a poor fit for **CPU-bound** work like image processing —
they don't run truly in parallel.

`multiprocessing` works around the GIL by starting entirely separate **OS
processes**, each with its own Python interpreter and its own GIL. The
operating system can then schedule these processes onto different physical
CPU cores, so they genuinely execute at the same time.

The tradeoff is **overhead**: data has to be divided, *pickled* (serialized)
and sent to each worker process, processed, and the results sent back and
combined. All of that takes real time, which is why:

> More workers does not automatically mean better performance.

## 10. Image Chunking and the Halo Problem

Grayscale is **pointwise**: each output pixel only depends on the
*same* input pixel, so splitting the image into chunks is trivial — no
extra information is needed.

Blur, Sharpen, and Edge Detection are **convolutions**: each output pixel
depends on a small neighborhood of surrounding input pixels (e.g. a 3x3 or
5x5 area). If you naively split the image into chunks and process each
chunk in total isolation, pixels near the top/bottom edge of a chunk would
be missing real neighbor data from the adjacent chunk — producing a visible
seam/stripe at every chunk boundary.

**The fix: a halo (a.k.a. "ghost region").** Before handing a chunk to a
worker, `parallel.py` grows it by `radius` extra rows borrowed from the
neighboring chunk(s) (clamped at the image's actual top/bottom edge). The
worker processes this enlarged chunk as normal. Afterwards, the borrowed
halo rows are cropped back off before the chunks are stitched together.
This means every real output pixel had access to its true neighbors, and
sequential vs. parallel output is **pixel-identical** for Grayscale, Blur,
Sharpen, and Edge Detection (verified in the app's "Compare Both" mode via
a mean-pixel-difference sanity check, and confirmed to be exactly `0.0` in
this project's own test runs).

```text
Original Image
┌───────────────┐
│               │
│               │
│               │
│               │
└───────────────┘

Split into chunks (with halo overlap)

┌───────┬───────┐
│ W1    │ W2    │
├───────┼───────┤
│ W3    │ W4    │
└───────┴───────┘
```

### Resize is different

Resize does **not** use the halo approach, and is called out separately in
the UI. A resize filter (even simple bilinear interpolation) blends
together neighboring pixels — including, potentially, pixels that live just
across a chunk boundary. This project's parallel resize instead splits the
image into independent horizontal **bands** and resizes each band on its
own worker, with each band's exact target height computed from cumulative,
rounded scaled boundaries (so the stacked bands always add up to the same
total height as a whole-image resize). This keeps the implementation simple
and avoids a second halo scheme, but it means pixels right at a band
boundary are interpolated using only the pixels available inside that band
— not the true neighbors from the next band. In practice this produces
output extremely close to (but not always pixel-identical to) a whole-image
resize, visible as a very faint seam at band boundaries for some scale
factors. This is a deliberate, documented simplification rather than a bug
— a perfectly seamless parallel resize would need halo handling as complex
as Blur/Sharpen/Edge Detection for comparatively little visual benefit.

## 11. Synchronization

`multiprocessing.Pool.map()` is a **blocking** call: the main process
submits all chunk-processing tasks to the pool and then waits until every
single worker has returned a result before continuing. This wait is the
project's synchronization point — it guarantees all chunks are finished
and ready before they're combined, matching the "Divide → Distribute →
Process → **Synchronize** → Combine" model described in the app's "How
Parallel Processing Works" section.

## 12. Speedup

```text
Speedup = Sequential Time / Parallel Time
```

Example:

```text
Sequential = 8 seconds
Parallel   = 3 seconds

Speedup = 8 / 3 = 2.67x
```

A Speedup of `1.0x` means parallel took exactly as long as sequential (no
benefit). Above `1.0x` means parallel was faster; below `1.0x` means
parallel was actually *slower* (this can happen on small images, where
process/overhead cost outweighs the work saved).

## 13. Efficiency

```text
Efficiency = Speedup / Number of Workers
```

Example:

```text
Speedup = 2.67
Workers = 4

Efficiency = 2.67 / 4 = 0.6675 = 66.75%
```

100% efficiency would mean "perfect" linear scaling (4 workers = exactly
4x faster). In practice, overhead (splitting, sending data, combining
results) means efficiency is almost always below 100%, and tends to drop
further as more workers are added.

## 14. Benchmarking

The **"Run Benchmark"** button in the app:

1. Times the sequential version once (the baseline).
2. Times the parallel version at every worker count supported by your
   machine (from `1, 2, 4, 8`, filtered by `multiprocessing.cpu_count()`).
3. Builds a results table:

```text
Workers | Time | Speedup | Efficiency
---------------------------------------
1       | 8.2s | 1.00x   | 100%
2       | 4.7s | 1.74x   | 87%
4       | 2.8s | 2.93x   | 73%
8       | 2.4s | 3.42x   | 43%
```

   (Illustrative numbers only — **every value the app actually shows you is
   computed live from real timed runs on your uploaded image**, nothing is
   hard-coded.)

4. Plots two charts:
   - **Number of Workers vs Execution Time**
   - **Number of Workers vs Speedup** (with an "ideal linear speedup"
     reference line, so you can visually see how far real performance
     falls short of the theoretical maximum)

## 15. Results / UI Overview

- **Sequential / Parallel / Compare Both** modes, each showing the
  original and processed image side by side.
- Live metrics: Sequential Time, Parallel Time, Speedup, Efficiency,
  Workers.
- In "Compare Both" mode, a consistency check reports the mean pixel-value
  difference between the sequential and parallel outputs, so you can see
  for yourself how close (or identical) they are.
- Expandable sections explaining: how parallel processing works, data
  parallelism, why `multiprocessing` (not threads) is used, and speedup
  limitations.

## 16. Limitations

- **Resize** parallel output is a documented approximation (see Section 10)
  — not pixel-identical to sequential resize in every case.
- Very small images may show speedup below `1.0x` (parallel slower than
  sequential) because process-creation and data-transfer overhead exceeds
  the actual work being done. This is expected and is itself a useful PDC
  lesson, not a bug.
- Performance numbers depend heavily on the host machine: CPU core count,
  current system load, image size, and operating system all affect the
  results you'll see.
- This project targets clarity and correctness of the PDC concepts over
  maximum raw performance; the convolution implementation is a simple,
  readable NumPy shift-and-add rather than a heavily optimized routine.

## 17. Installation

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## 18. Running the Application

From the project root:

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in
your browser.

### Windows + multiprocessing note (important — read if the app seems to hang)

Windows uses the **"spawn"** start method for new processes: every time
`multiprocessing.Pool()` creates a worker, Windows has to start a brand
new Python process and re-import your script into it from scratch (Linux
and macOS use "fork" instead, which copies the already-running process in
memory and doesn't have this issue — this is why the behavior below is
Windows-specific).

Streamlit runs `app.py` by exec-ing it as `sys.modules['__main__']`, so
that `if __name__ == "__main__":` works the way it would in a normal
script. Combined with Windows' spawn behavior, this means: **every worker
process re-imports this entire file**, with `__name__` set to
`"__mp_main__"` instead of `"__main__"`.

All of the Streamlit UI code in this project lives inside a `main()`
function, which is only called from:

```python
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
```

Because a spawned worker's `__name__` is `"__mp_main__"`, not
`"__main__"`, this condition is `False` in every worker — so workers skip
straight past the UI entirely. `streamlit` and `matplotlib` are also
imported *inside* `main()` rather than at the top of the file, so a
worker process never even imports them.

**If you ever edit this project and add a Streamlit call (`st.something(...)`)
outside of `main()`, or move a heavy import back to the top of the file,
you will reintroduce this bug**: every worker you spawn will try to
re-run the whole Streamlit app with no real server behind it, which
shows up as the app hanging on "Process Image" / "Run Benchmark" and,
if you hit Ctrl+C, a huge wall of import tracebacks ending in
`KeyboardInterrupt` (because the process was stuck mid-import of
streamlit/matplotlib/numpy, not because of an actual bug in your
calculation code). The fix is always the same: make sure *every* line of
Streamlit UI code is reachable only through `main()`, called only inside
the `if __name__ == "__main__":` guard.

## 19. Project Structure

```text
parallel-image-processing/
│
├── app.py
├── sequential.py
├── parallel.py
├── image_operations.py
├── benchmark.py
├── requirements.txt
├── README.md
└── sample_images/
    ├── sample_large.jpg
    ├── sample_medium.jpg
    └── sample_small.jpg
```

(`sample_images/` contains a few synthetically generated test images —
patterned, high-contrast, and available in a few sizes — so you have
something to experiment with immediately, without needing to source your
own large image. Larger images make the parallel speedup more visible,
since there's more real work to divide among workers.)

## 20. Future Improvements

- Add more operations (e.g. rotate, color channel adjustments) using the
  same chunk/halo pattern.
- Add a true halo-based parallel resize for pixel-perfect equivalence with
  sequential resize.
- Add a progress bar that updates per-chunk as workers complete, using
  `Pool.imap_unordered()` instead of `Pool.map()`.
- Persist benchmark history across runs so different images/operations can
  be compared on the same chart.
- Optionally support `ProcessPoolExecutor` from `concurrent.futures` as an
  alternate implementation, to compare API styles.

---

### Summary of where multiprocessing is actually used

Every call to `multiprocessing.Pool(...)` in this project lives in
`parallel.py`, inside either `process_parallel()` (Grayscale / Blur / Edge
Detection / Sharpen) or `_process_parallel_resize()` (Resize). Both are
only ever invoked from `benchmark.py`'s `time_parallel()` /
`run_comparison()` / `run_worker_sweep()` functions, which are in turn only
called from `app.py` inside button-click handlers — never automatically on
page load, and never at module import time.
