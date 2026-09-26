"""
app.py
------
The Streamlit user interface for the Parallel Image Processing System.

This file is intentionally "dumb": it does not contain any image-processing
logic or multiprocessing logic itself. It only:

  1. Collects input from the user (image, operation, parameters, mode,
     worker count).
  2. Calls into sequential.py / parallel.py / benchmark.py to do the actual
     work.
  3. Displays the results.

Run with:
    streamlit run app.py

WINDOWS NOTE: multiprocessing.Pool() is only ever created *inside* function
calls that are triggered by a button click (see benchmark.py / parallel.py),
never at import time / module top-level. This is important on Windows,
which uses the "spawn" start method for new processes: if Pool() were
created as soon as the module was imported, Streamlit re-running this script
on every interaction could spawn processes recursively. Because all Pool
creation happens lazily inside function bodies, this project is safe to run
with `streamlit run app.py` on Windows.
"""

import io
import multiprocessing
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Force a non-interactive backend: Streamlit has no
                        # GUI event loop, and this avoids any chance of
                        # matplotlib trying to open a window on machines
                        # where a different backend is the default.
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

import sequential
import parallel
import benchmark
import image_operations as ops

# On Windows, multiprocessing uses the "spawn" start method and re-imports
# this module in each worker process. freeze_support() is a no-op on
# Linux/macOS but is the officially recommended safety call for any script
# that uses multiprocessing.Pool and might be frozen into an .exe or run in
# unusual ways on Windows.
multiprocessing.freeze_support()


st.set_page_config(page_title="Parallel Image Processing System", layout="wide")

MAX_CORES = multiprocessing.cpu_count()
WORKER_CANDIDATES = [1, 2, 4, 8]
WORKER_OPTIONS = [w for w in WORKER_CANDIDATES if w <= MAX_CORES] or [1]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧩 Parallel Image Processing System")
st.caption("A Parallel & Distributed Computing Project")

st.write(
    "Upload an image, pick an operation, and compare how long it takes to "
    "process **sequentially** (one CPU core) versus **in parallel** "
    "(multiple CPU cores, using Python's `multiprocessing` module)."
)

with st.expander("ℹ️ About this project / how it works", expanded=False):
    st.markdown(
        """
This app demonstrates **data parallelism**: a large image is split into
smaller chunks, and multiple worker *processes* perform the same operation
on different chunks **at the same time**, on different CPU cores.

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
Combine results -> Final Image
```

We use `multiprocessing` (separate OS processes) instead of threads because
CPython's **Global Interpreter Lock (GIL)** prevents multiple threads from
executing Python bytecode at the same time on separate cores. Separate
*processes* each get their own interpreter and their own GIL, so they can
truly run in parallel.

Splitting and combining chunks takes time too (this is called **overhead**),
so more workers does not automatically mean better performance - especially
on small images. See the "How Parallel Processing Works" and "Speedup &
Efficiency" sections lower down for the full explanation.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

st.subheader("1. Upload an Image")
uploaded_file = st.file_uploader(
    "Upload Image", type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded_file is None:
    st.info(
        "No image uploaded yet. A few sample images are included in the "
        "`sample_images/` folder of this project if you'd like something "
        "to test with (large, computation-heavy images work best for "
        "showing off parallel speedup)."
    )
    st.stop()

try:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_arr = np.array(pil_image)
except Exception as exc:
    st.error(f"Could not read this file as an image. Details: {exc}")
    st.stop()

h, w = image_arr.shape[:2]
st.success(f"Loaded image: {w} x {h} pixels")

# ---------------------------------------------------------------------------
# Operation selection
# ---------------------------------------------------------------------------

st.subheader("2. Select Operation")
operation = st.selectbox("Select Operation", ops.OPERATIONS)

params = {}
col_a, col_b = st.columns(2)
with col_a:
    if operation == "Blur":
        params["radius"] = st.slider("Blur Radius", min_value=1, max_value=10, value=2)
    elif operation == "Resize":
        params["scale"] = st.select_slider(
            "Scale (%)", options=[25, 50, 75, 100, 150, 200], value=100
        )
    else:
        st.caption("This operation has no extra parameters.")

# ---------------------------------------------------------------------------
# Processing mode + workers
# ---------------------------------------------------------------------------

st.subheader("3. Processing Mode")
mode = st.radio(
    "Processing Mode", ["Sequential", "Parallel", "Compare Both"], horizontal=True
)

num_workers = 1
if mode in ("Parallel", "Compare Both"):
    st.caption(f"This machine has **{MAX_CORES}** CPU core(s) available.")
    num_workers = st.radio(
        "Number of Workers", WORKER_OPTIONS, horizontal=True,
        index=len(WORKER_OPTIONS) - 1,
    )

st.divider()

# ---------------------------------------------------------------------------
# Process button
# ---------------------------------------------------------------------------

st.subheader("4. Run")
run_clicked = st.button("▶️ Process Image", type="primary")

if run_clicked:
    try:
        if mode == "Sequential":
            with st.spinner("Processing sequentially..."):
                result, seq_time = benchmark.time_sequential(image_arr, operation, params)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_arr, caption="Original Image", width="stretch")
            with col2:
                st.image(result, caption="Processed Image", width="stretch")

            st.metric("Sequential Time", f"{seq_time:.3f} s")

        elif mode == "Parallel":
            with st.spinner(f"Processing in parallel with {num_workers} worker(s)..."):
                result, par_time = benchmark.time_parallel(
                    image_arr, operation, params, num_workers
                )

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_arr, caption="Original Image", width="stretch")
            with col2:
                st.image(result, caption="Processed Image", width="stretch")

            st.metric("Parallel Time", f"{par_time:.3f} s")
            st.metric("Workers", num_workers)

        else:  # Compare Both
            with st.spinner("Running sequential and parallel versions..."):
                comparison = benchmark.run_comparison(
                    image_arr, operation, params, num_workers
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image_arr, caption="Original Image", width="stretch")
            with col2:
                st.image(
                    comparison["sequential_result"],
                    caption="Sequential Result", width="stretch",
                )
            with col3:
                st.image(
                    comparison["parallel_result"],
                    caption="Parallel Result", width="stretch",
                )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Sequential Time", f"{comparison['sequential_time']:.3f} s")
            m2.metric("Parallel Time", f"{comparison['parallel_time']:.3f} s")
            m3.metric("Speedup", f"{comparison['speedup']:.2f}x")
            m4.metric("Efficiency", f"{comparison['efficiency'] * 100:.1f} %")
            m5.metric("Workers", comparison["workers"])

            # Sanity check: sequential and parallel outputs should be
            # visually equivalent (allowing for the small, documented
            # differences described for Resize / Blur near image edges).
            seq_arr = comparison["sequential_result"].astype(np.float64)
            par_arr = comparison["parallel_result"].astype(np.float64)
            if seq_arr.shape == par_arr.shape:
                mean_diff = float(np.mean(np.abs(seq_arr - par_arr)))
                st.caption(
                    f"Consistency check: mean pixel-value difference between "
                    f"sequential and parallel output = **{mean_diff:.3f}** "
                    f"(0 = pixel-identical; small non-zero values near image "
                    f"edges/chunk boundaries are expected and explained in "
                    f"the README)."
                )
            else:
                st.caption(
                    "Consistency check skipped: sequential and parallel "
                    "outputs have slightly different pixel dimensions, "
                    "which can happen with Resize due to independent "
                    "per-band rounding (see README for details)."
                )

    except Exception as exc:
        st.error(f"Processing failed: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------

st.subheader("5. Full Benchmark (all worker counts)")
st.caption(
    "Runs the sequential version once, then the parallel version at every "
    "supported worker count, and plots the results. Nothing here is "
    "hard-coded - every value comes from an actual timed run on your "
    "uploaded image."
)

if st.button("📊 Run Benchmark"):
    try:
        with st.spinner("Benchmarking... this runs the operation multiple times."):
            sweep = benchmark.run_worker_sweep(image_arr, operation, params, WORKER_OPTIONS)

        st.metric("Sequential Time (baseline)", f"{sweep['sequential_time']:.3f} s")

        st.table(sweep["rows"])

        workers_list = [row["Workers"] for row in sweep["rows"]]
        times_list = [row["Time (s)"] for row in sweep["rows"]]
        speedup_list = [row["Speedup"] for row in sweep["rows"]]

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig1, ax1 = plt.subplots()
            ax1.plot(workers_list, times_list, marker="o", color="#1f77b4")
            ax1.set_xlabel("Number of Workers")
            ax1.set_ylabel("Execution Time (s)")
            ax1.set_title("Workers vs Execution Time")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        with chart_col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(workers_list, speedup_list, marker="o", color="#ff7f0e", label="Actual Speedup")
            ax2.plot(workers_list, workers_list, linestyle="--", color="gray", label="Ideal (linear) Speedup")
            ax2.set_xlabel("Number of Workers")
            ax2.set_ylabel("Speedup (x)")
            ax2.set_title("Workers vs Speedup")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2)

    except Exception as exc:
        st.error(f"Benchmark failed: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Educational sections
# ---------------------------------------------------------------------------

with st.expander("📖 How Parallel Processing Works"):
    st.markdown(
        """
**Step 1 - Divide.** The image is divided into smaller row-chunks (with a
small overlapping "halo" region for operations that need neighboring
pixels, such as Blur, Sharpen and Edge Detection).

**Step 2 - Distribute.** Each chunk is assigned to a worker process.

**Step 3 - Process.** Workers process their chunks simultaneously, on
separate CPU cores.

**Step 4 - Synchronize.** The main process waits for every worker to
finish (`Pool.map()` blocks until all results are ready).

**Step 5 - Combine.** The processed chunks are stacked back together
(and any borrowed halo rows are cropped off first) to produce the final
image.

```text
                 Main Process
                      |
                  Split Image
                      |
       +--------------+--------------+
       v              v              v
   Process 1      Process 2      Process 3
       v              v              v
    Chunk 1         Chunk 2        Chunk 3
       +--------------+--------------+
                      |
                  Join Chunks
                      |
                      v
                 Final Image
```
        """
    )

with st.expander("🧵 Data Parallelism (the main PDC concept here)"):
    st.markdown(
        """
**Data parallelism** means dividing a large dataset into smaller portions
and having multiple workers perform the **same operation** on different
portions **simultaneously**.

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

Every worker in this project runs the exact same function (e.g. `blur()`)
- they just each see a different slice of the image. That's what makes
this data parallelism, as opposed to *task* parallelism (different workers
doing genuinely different jobs).
        """
    )

with st.expander("⚙️ Why multiprocessing instead of threads?"):
    st.markdown(
        """
Python (specifically CPython, the standard implementation) has a
**Global Interpreter Lock (GIL)**: only one thread can execute Python
bytecode at any given instant, even on a multi-core machine. This means
plain threads are a poor fit for CPU-bound work like image processing -
they won't actually run in parallel.

The `multiprocessing` module works around this by starting entirely
separate **OS processes**, each with its own Python interpreter and its
own GIL. The operating system can schedule these processes onto different
CPU cores, so they genuinely run at the same time.

The tradeoff is **overhead**: data has to be *divided*, *sent* (pickled and
transferred) to each worker process, *processed*, and the results have to
be *sent back* and *combined*. All of that takes time - so:

> More workers does not automatically mean better performance.
        """
    )

with st.expander("📉 Speedup limitations"):
    st.markdown(
        """
```text
Image
  |
Split
  |
Send chunks to workers
  |
Workers process
  |
Return results
  |
Combine
```

Every one of those steps costs time. For a **small** image, that overhead
can be larger than the time saved by processing in parallel - so a small
image may actually be *faster* sequentially. A sufficiently **large**
image provides enough real computational work for the benefits of
parallelism to outweigh the overhead.

Real-world speedup also depends on:

- CPU core count
- Image size
- Operation complexity (Blur/Sharpen/Edge Detection do more math per pixel
  than Grayscale)
- Number of workers requested
- Process creation overhead
- Data transfer overhead (sending image chunks between processes)
- Memory bandwidth

This is why **Efficiency** (`Speedup / Workers`) is a useful number to
track alongside raw Speedup: it tells you how much of the "theoretical
maximum" benefit you actually got from adding more workers.
        """
    )

st.caption(
    f"Detected {MAX_CORES} CPU core(s) on this machine. "
    f"Worker options are automatically limited to this maximum."
)
