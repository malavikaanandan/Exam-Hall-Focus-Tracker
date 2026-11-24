"""Helpers for tiled detection: splitting a frame into tiles and merging boxes.

Face detectors like BlazeFace shrink the whole frame to a small input size,
so faces far from the camera become too small to find. Cutting the frame into
overlapping tiles and enlarging each tile makes those faces big enough again.
"""


def tile_grid(width, height, rows, cols, overlap=0.2):
    """Return (x0, y0, x1, y1) pixel boxes covering the frame.

    Neighbouring tiles overlap by `overlap` of a tile's size, so a face on a
    tile border is still fully inside at least one tile.
    """
    tw, th = width / cols, height / rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(max(0, c * tw - tw * overlap))
            y0 = int(max(0, r * th - th * overlap))
            x1 = int(min(width, (c + 1) * tw + tw * overlap))
            y1 = int(min(height, (r + 1) * th + th * overlap))
            tiles.append((x0, y0, x1, y1))
    return tiles


def iou(a, b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(faces, iou_thresh=0.3):
    """Remove duplicate detections of the same face (e.g. from overlapping tiles).

    faces: list of dicts with "box" (x, y, w, h) and "score". Keeps the
    highest-scoring face of every group of overlapping boxes.
    """
    kept = []
    for f in sorted(faces, key=lambda f: f["score"], reverse=True):
        if all(iou(f["box"], k["box"]) < iou_thresh for k in kept):
            kept.append(f)
    return kept
