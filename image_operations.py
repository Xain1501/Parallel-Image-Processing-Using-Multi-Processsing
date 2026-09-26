"""
image_operations.py
--------------------
This file contains the actual "pixel math" for every image operation used in
this project:

    - Grayscale
    - Blur (Gaussian)
    - Edge Detection (Sobel)
    - Sharpen
    - Resize

Both sequential.py and parallel.py call these SAME functions. The only
difference between "sequential" and "parallel" is HOW MUCH of the image is
handed to these functions at once (the whole image vs. one small chunk of
it), and whether multiple worker processes call them at the same time.

This is intentional: it makes it very clear that parallelism does not change
*what* work is done, only *how* the work is split up and scheduled.

Every function here works on NumPy arrays of shape (height, width, 3) with
dtype uint8 (standard RGB image data), except where noted.
"""

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------
# "Blur", "Sharpen" and "Edge Detection" are all implemented as a small
# matrix (a "kernel") that is slid over every pixel of the image. Each output
# pixel is a weighted sum of that pixel and its neighbors.
#
# Because these operations look at NEIGHBORING pixels, if you cut the image
# into chunks and process each chunk in isolation, pixels near the top/bottom
# edge of a chunk would be missing neighbors that live in the NEXT chunk.
# This produces visible seams/stripes at chunk boundaries.
#
# The fix used throughout this project is a "halo" (also called a "ghost
# region" or "overlap region"): before giving a chunk to a worker, we grow it
# by `radius` extra rows above and below (borrowed from the neighboring
# chunk). The worker processes the enlarged chunk, and afterwards we crop the
# borrowed rows back off. This way every real output pixel had access to its
# true neighbors, and there are no seams.
# ---------------------------------------------------------------------------


def gaussian_kernel(radius: int) -> np.ndarray:
    """Build a 2D Gaussian blur kernel with the given radius.

    A radius of 2 produces a 5x5 kernel, radius 4 produces a 9x9 kernel, etc.
    Larger radius = stronger/wider blur = larger halo needed for chunking.
    """
    radius = max(1, int(radius))
    sigma = max(radius / 2.0, 0.6)
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


# Fixed 3x3 kernels (radius = 1) used by Sharpen and Edge Detection.
SHARPEN_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]], dtype=np.float64
)

SOBEL_X = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], dtype=np.float64
)

SOBEL_Y = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]], dtype=np.float64
)


def _convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D kernel to a single-channel (H, W) array via convolution.

    Uses 'reflect' padding so edge pixels are handled sensibly, and a simple
    shift-and-add implementation which is fast enough for small kernels
    (3x3, 5x5, 9x9) without needing extra dependencies like scipy.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(channel, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight == 0:
                continue
            out += weight * padded[i:i + channel.shape[0], j:j + channel.shape[1]]
    return out


def _apply_kernel_rgb(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a kernel to every channel of an (H, W, 3) RGB image."""
    channels = [_convolve_channel(arr[:, :, c].astype(np.float64), kernel)
                for c in range(arr.shape[2])]
    result = np.stack(channels, axis=2)
    return np.clip(result, 0, 255).astype(np.uint8)


def kernel_halo_radius(operation: str, params: dict) -> int:
    """Return how many extra rows of neighboring pixels a given operation
    needs above/below a chunk in order to produce seamless output.

    Grayscale and Resize are pointwise / band-based, so they need no halo.
    """
    if operation == "Blur":
        radius = int(params.get("radius", 2))
        return radius
    if operation in ("Edge Detection", "Sharpen"):
        return 1  # 3x3 kernels
    return 0  # Grayscale, Resize


# ---------------------------------------------------------------------------
# The five operations
# ---------------------------------------------------------------------------

def grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using the standard luminosity
    formula, then replicate the single channel back into 3 channels so the
    result can still be displayed/compared as a normal RGB image.

    This is a pointwise operation: every output pixel only depends on the
    SAME pixel of the input, never on its neighbors. That makes it the
    simplest possible case for parallel chunking (no halo required).
    """
    arr = arr.astype(np.float64)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def blur(arr: np.ndarray, radius: int = 2) -> np.ndarray:
    """Apply a Gaussian blur. Needs a halo of `radius` rows when chunked."""
    kernel = gaussian_kernel(radius)
    return _apply_kernel_rgb(arr, kernel)


def sharpen(arr: np.ndarray) -> np.ndarray:
    """Sharpen the image using a standard 3x3 sharpening kernel.
    Needs a halo of 1 row when chunked.
    """
    return _apply_kernel_rgb(arr, SHARPEN_KERNEL)


def edge_detection(arr: np.ndarray) -> np.ndarray:
    """Detect edges using the Sobel operator.

    Steps:
      1. Convert to grayscale (edges are computed on intensity, not color).
      2. Convolve with a horizontal Sobel kernel (Gx) and a vertical Sobel
         kernel (Gy) to estimate the intensity gradient in each direction.
      3. Combine them into a gradient magnitude: sqrt(Gx^2 + Gy^2).
      4. Clip to the 0-255 range so it can be displayed as an image.

    Needs a halo of 1 row when chunked (3x3 kernels).

    NOTE ON DESIGN: we deliberately clip the raw magnitude to 0-255 rather
    than rescaling it based on this image's own min/max gradient value.
    An adaptive, data-dependent rescale would need to know the max gradient
    across the *entire* image - but in parallel mode each worker only ever
    sees its own chunk, so it would compute a different max and rescale
    each chunk differently, producing visible brightness seams between
    chunks. Using a fixed 0-255 clip keeps every pixel's math independent
    of everything else in the image, which is what makes the halo approach
    (and chunking in general) produce pixel-identical sequential and
    parallel output.
    """
    gray = grayscale(arr)[:, :, 0].astype(np.float64)
    gx = _convolve_channel(gray, SOBEL_X)
    gy = _convolve_channel(gray, SOBEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return np.stack([magnitude, magnitude, magnitude], axis=2)


def resize_whole(arr: np.ndarray, scale: float) -> np.ndarray:
    """Resize the ENTIRE image at once (used by the sequential version and
    as the 'ground truth' to compare the parallel band-based resize against).

    scale is a fraction, e.g. 0.5 = 50%, 1.5 = 150%.
    """
    h, w = arr.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(arr.astype(np.uint8))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.array(resized)


def resize_band(band: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Resize a single horizontal BAND of the image independently to an
    already-computed target size (see parallel._process_parallel_resize for
    how new_height is chosen so that all bands stacked together add up to
    exactly the same total height as a whole-image resize).

    A band can legitimately be asked to resize down to 0 rows (this happens
    when downscaling a very small image with many workers, so that a given
    band's own share of the output has no rows at all) - in that case we
    return an empty array of the right width/dtype rather than forcing a
    minimum of 1 row, which would make the combined output taller than a
    whole-image resize.

    IMPORTANT LIMITATION (explained in the README and the app's UI):
    Resizing is not a pointwise operation - a proper resize filter (even
    "simple" bilinear interpolation) blends together neighboring pixels,
    including pixels that may sit just across a chunk boundary. If we split
    the image into bands and resize each one completely independently
    (which is what this function does), pixels right at a band's top/bottom
    edge are interpolated using only the pixels available *inside that
    band*, not the true neighboring pixels from the adjacent band.

    In practice this produces output that is extremely close to, but not
    always pixel-identical to, resizing the whole image at once - most
    noticeably as a very faint seam at band boundaries. We accept this as a
    reasonable, clearly-documented approximation rather than pretending it
    is perfectly equivalent (a truly seamless parallel resize would need a
    halo like Blur/Sharpen/Edge Detection, adding real complexity for a
    fairly small visual benefit).
    """
    new_width = max(1, int(new_width))
    new_height = int(new_height)
    if new_height <= 0:
        return np.zeros((0, new_width, 3), dtype=np.uint8)
    img = Image.fromarray(band.astype(np.uint8))
    resized = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.array(resized)


OPERATIONS = ["Grayscale", "Blur", "Edge Detection", "Sharpen", "Resize"]
