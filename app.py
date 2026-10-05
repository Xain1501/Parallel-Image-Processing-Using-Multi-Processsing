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
  3. Displays the results - images, metrics, charts, and a visual overlay
     of the actual chunk boundaries used, rather than paragraphs of text.
     The project is meant to be understood by using it, not by reading an
     in-app explanation.

Run with:
    streamlit run app.py

--------------------------------------------------------------------------
IMPORTANT WINDOWS NOTE (read this if you ever see the app hang, or a huge
wall of import tracebacks, when you click "Process Image" or
"Run Benchmark"):
--------------------------------------------------------------------------
Streamlit runs your script by exec-ing it as a module that it registers as
`sys.modules['__main__']`, specifically so that `if __name__ == "__main__":`
blocks work the way a normal script's would. On Windows, multiprocessing's
default "spawn" start method bootstraps every new worker process by
RE-RUNNING whatever is currently `sys.modules['__main__']` - in this case,
this entire file - but with its `__name__` set to `"__mp_main__"` instead
of `"__main__"`.

If the Streamlit UI code below were *not* wrapped in a guard, every one of
the worker processes you spawn would re-run the ENTIRE Streamlit app from
scratch (re-importing streamlit/matplotlib/numpy and re-executing
st.set_page_config(), the file uploader, etc.) with no real Streamlit
server behind it - which is exactly what produces a storm of import
tracebacks and an app that appears to hang.

The fix: every single line of UI code lives inside `main()`, and `main()`
is only called when `__name__ == "__main__"` (true for the real Streamlit
process, false for spawned workers). Workers then do nothing but define
functions/import lightweight modules - they never touch Streamlit at all.
`streamlit` and `matplotlib` are also imported *inside* main() (not at
module level) so that a spawned worker process never has to import them
either, keeping each worker's startup fast.

If you are extending this app, keep this pattern: don't add any top-level
Streamlit calls outside of `main()`.
--------------------------------------------------------------------------
"""

import multiprocessing

import numpy as np

import sequential
import parallel
import benchmark
import image_operations as ops


CHUNK_COLORS = [
    "#FF3B30", "#34C759", "#007AFF", "#FF9500",
    "#AF52DE", "#00C7BE", "#FFD60A", "#FF2D55",
]


def _chunk_overlay(arr: np.ndarray, boundaries, Image, ImageDraw):
    """Return a copy of `arr` with colored bands drawn over each chunk's
    row range, so the actual split used by the parallel run is visible
    directly on the image instead of being described in text.
    """
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for i, (start, end) in enumerate(boundaries):
        color = CHUNK_COLORS[i % len(CHUNK_COLORS)]
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        draw.rectangle([0, start, img.width - 1, max(start, end - 1)], fill=(r, g, b, 40))
        draw.line([0, start, img.width - 1, start], fill=color, width=3)
    draw.line([0, boundaries[-1][1] - 1, img.width - 1, boundaries[-1][1] - 1], fill=CHUNK_COLORS[(len(boundaries) - 1) % len(CHUNK_COLORS)], width=3)
    return np.array(img)


def main():
    # These imports are intentionally INSIDE main(), not at module level.
    # See the module docstring above: this keeps them out of every spawned
    # multiprocessing worker process's startup path on Windows.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import streamlit as st
    from PIL import Image, ImageDraw

    st.set_page_config(page_title="Parallel Image Processing System", layout="wide")

    MAX_CORES = multiprocessing.cpu_count()
    WORKER_CANDIDATES = [1, 2, 4, 8]
    WORKER_OPTIONS = [w for w in WORKER_CANDIDATES if w <= MAX_CORES] or [1]

    st.title("🧩 Parallel Image Processing System")

    uploaded_file = st.file_uploader(
        "Upload Image", type=["png", "jpg", "jpeg", "bmp", "webp"]
    )

    if uploaded_file is None:
        st.stop()

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_arr = np.array(pil_image)
    except Exception as exc:
        st.error(f"Could not read this file as an image. Details: {exc}")
        st.stop()

    h, w = image_arr.shape[:2]

    top_col1, top_col2, top_col3 = st.columns(3)
    with top_col1:
        operation = st.selectbox("Operation", ops.OPERATIONS)
    with top_col2:
        params = {}
        if operation == "Blur":
            params["radius"] = st.slider("Blur Radius", min_value=1, max_value=10, value=2)
        elif operation == "Resize":
            params["scale"] = st.select_slider(
                "Scale (%)", options=[25, 50, 75, 100, 150, 200], value=100
            )
        else:
            st.caption(f"{w} x {h} px")
    with top_col3:
        mode = st.radio("Mode", ["Sequential", "Parallel", "Compare Both"], horizontal=True)

    num_workers = 1
    if mode in ("Parallel", "Compare Both"):
        num_workers = st.radio(
            f"Workers (max {MAX_CORES})", WORKER_OPTIONS, horizontal=True,
            index=len(WORKER_OPTIONS) - 1,
        )

    run_clicked = st.button("▶️ Process Image", type="primary")

    if run_clicked:
        try:
            if mode == "Sequential":
                with st.spinner("Processing..."):
                    result, seq_time = benchmark.time_sequential(image_arr, operation, params)

                col1, col2 = st.columns(2)
                col1.image(image_arr, caption="Original", width="stretch")
                col2.image(result, caption="Processed", width="stretch")
                st.metric("Sequential Time", f"{seq_time:.3f} s")

            elif mode == "Parallel":
                with st.spinner("Processing..."):
                    result, par_time = benchmark.time_parallel(
                        image_arr, operation, params, num_workers
                    )

                boundaries = parallel.get_chunk_boundaries(h, num_workers)
                overlay = _chunk_overlay(image_arr, boundaries, Image, ImageDraw)

                col1, col2, col3 = st.columns(3)
                col1.image(image_arr, caption="Original", width="stretch")
                col2.image(overlay, caption=f"{len(boundaries)} chunks", width="stretch")
                col3.image(result, caption="Processed", width="stretch")

                m1, m2 = st.columns(2)
                m1.metric("Parallel Time", f"{par_time:.3f} s")
                m2.metric("Workers", num_workers)

            else:  # Compare Both
                with st.spinner("Processing..."):
                    comparison = benchmark.run_comparison(
                        image_arr, operation, params, num_workers
                    )

                boundaries = parallel.get_chunk_boundaries(h, num_workers)
                overlay = _chunk_overlay(image_arr, boundaries, Image, ImageDraw)

                col1, col2, col3 = st.columns(3)
                col1.image(overlay, caption=f"{len(boundaries)} chunks", width="stretch")
                col2.image(comparison["sequential_result"], caption="Sequential", width="stretch")
                col3.image(comparison["parallel_result"], caption="Parallel", width="stretch")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Sequential", f"{comparison['sequential_time']:.3f} s")
                m2.metric("Parallel", f"{comparison['parallel_time']:.3f} s")
                m3.metric("Speedup", f"{comparison['speedup']:.2f}x")
                m4.metric("Efficiency", f"{comparison['efficiency'] * 100:.1f}%")
                m5.metric("Workers", comparison["workers"])

                seq_arr = comparison["sequential_result"].astype(np.float64)
                par_arr = comparison["parallel_result"].astype(np.float64)
                if seq_arr.shape == par_arr.shape:
                    mean_diff = float(np.mean(np.abs(seq_arr - par_arr)))
                    st.metric("Pixel Diff (seq vs par)", f"{mean_diff:.3f}")

        except Exception as exc:
            st.error(f"Processing failed: {exc}")

    st.divider()

    if st.button("📊 Run Benchmark"):
        try:
            with st.spinner("Benchmarking..."):
                sweep = benchmark.run_worker_sweep(image_arr, operation, params, WORKER_OPTIONS)

            st.metric("Sequential Time", f"{sweep['sequential_time']:.3f} s")
            st.table(sweep["rows"])

            workers_list = [row["Workers"] for row in sweep["rows"]]
            times_list = [row["Time (s)"] for row in sweep["rows"]]
            speedup_list = [row["Speedup"] for row in sweep["rows"]]

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig1, ax1 = plt.subplots()
                ax1.plot(workers_list, times_list, marker="o", color="#1f77b4")
                ax1.set_xlabel("Workers")
                ax1.set_ylabel("Time (s)")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)

            with chart_col2:
                fig2, ax2 = plt.subplots()
                ax2.plot(workers_list, speedup_list, marker="o", color="#ff7f0e", label="Actual")
                ax2.plot(workers_list, workers_list, linestyle="--", color="gray", label="Ideal")
                ax2.set_xlabel("Workers")
                ax2.set_ylabel("Speedup (x)")
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                st.pyplot(fig2)

        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")


if __name__ == "__main__":
    # On Windows, multiprocessing's "spawn" start method re-imports this
    # file for every worker process, but with __name__ set to
    # "__mp_main__" instead of "__main__" - so this block (and therefore
    # all of the Streamlit UI inside main()) only ever runs in the real,
    # top-level Streamlit process. freeze_support() is the officially
    # recommended call to make right after this guard for any script that
    # uses multiprocessing on Windows.
    multiprocessing.freeze_support()
    main()
