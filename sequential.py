"""
sequential.py
-------------
The "baseline" implementation. Everything runs in a single Python process,
on a single CPU core, operating on the WHOLE image at once.

This file is deliberately tiny - its only job is to dispatch to the right
function in image_operations.py. It exists as its own file (rather than
being inlined into app.py) so that:

  1. It's obvious, when reading the project, exactly what "sequential mode"
     does.
  2. parallel.py can import and reuse it: each parallel worker calls
     process_sequential() on its own small chunk of the image. Parallelism
     in this project is really just "run the sequential logic many times,
     at once, on different pieces of the image."
"""

import image_operations as ops


def process_sequential(arr, operation: str, params: dict):
    """Process a full image (or a chunk of one) using a single process.

    Parameters
    ----------
    arr : np.ndarray
        An (H, W, 3) uint8 RGB image array.
    operation : str
        One of: "Grayscale", "Blur", "Edge Detection", "Sharpen", "Resize".
    params : dict
        Extra parameters the operation needs, e.g. {"radius": 3} for Blur
        or {"scale": 150} for Resize (scale is a PERCENTAGE here, e.g. 150
        means 150%).

    Returns
    -------
    np.ndarray
        The processed image array.
    """
    if operation == "Grayscale":
        return ops.grayscale(arr)

    elif operation == "Blur":
        radius = int(params.get("radius", 2))
        return ops.blur(arr, radius)

    elif operation == "Edge Detection":
        return ops.edge_detection(arr)

    elif operation == "Sharpen":
        return ops.sharpen(arr)

    elif operation == "Resize":
        scale = float(params.get("scale", 100)) / 100.0
        return ops.resize_whole(arr, scale)

    else:
        raise ValueError(f"Unknown operation: {operation}")
