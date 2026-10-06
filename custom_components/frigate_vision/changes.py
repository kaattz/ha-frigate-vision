"""Block-based change analysis over low-resolution frames.

This module is the "which moments are worth a cell" half of the evidence build,
and it is deliberately pure: it takes grayscale arrays, returns numbers and
segments, and imports nothing from Home Assistant. That keeps the selection
algorithm unit-testable without a runtime, and it keeps the expensive part
(decoding, HTTP) in the caller where the executor already is.

Why blocks rather than one whole-frame average
---------------------------------------------
A door opening on the left of the frame changes a few percent of the pixels. The
mean difference over the whole frame is a small number that sits near the noise
of a camera's own exposure drift, so a whole-frame average cannot tell "the door
moved" from "the sun came out". Measured on this deployment's own sheets, the
decisive moments are local: the entrance door and the lift door each occupy
roughly a fifth of the frame while the rest is a still corridor.

So the score is the mean of the *busiest* few blocks, not the mean of all of
them. A block that changed a lot counts even when its neighbours did not, which
is exactly the signal the frame-level average throws away.

Two comparisons, not one
------------------------
Each frame is compared against the previous frame AND against its segment's
anchor. Adjacent-frame comparison alone misses a slow change: a door opening over
four preview frames moves little between any two of them, so every pairwise score
stays under the threshold while the door ends up wide open. The anchor comparison
is what catches the accumulation, and it is why a frame can be scored as changed
even when its immediate predecessor looked identical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Whole-frame analysis resolution. 160x90 is a sixteenth of the 640x360 detect
# frame on each axis: enough to keep the doors legible as shapes, small enough
# that a hundred candidates cost a few million byte operations rather than a
# video decode.
WHOLE_WIDTH = 160
WHOLE_HEIGHT = 90

# 8 columns x 6 rows = 48 blocks of 20x15. The block count is the resolution of
# the "where" question: finer blocks localise a change better but each holds
# fewer pixels, and a 20x15 block still has 300 samples for a proportion
# estimate, which is ample.
BLOCK_COLUMNS = 8
BLOCK_ROWS = 6

# Grayscale levels a pixel must move by to count as changed. Below this the
# difference is camera noise rather than scene change. Deliberately a module
# constant rather than a literal so it can be calibrated against real day, night
# and light-switch samples.
PIXEL_THRESHOLD = 12.0

# How many of the busiest blocks carry the frame's score. Three of forty-eight is
# the smallest count that survives a block boundary landing mid-door: with one,
# a change straddling two blocks halves its own score; with three, the two halves
# both contribute.
TOP_BLOCKS = 3

# Focus regions are resampled to this square before the same block analysis runs.
# 96x96 on a region that occupies ~1/5 of a 320x180 preview is an upscale, which
# invents no detail -- what it buys is that the SAME block grid applies to a small
# region as to the whole frame, so one threshold works for both.
FOCUS_SIZE = 96


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """The grid a frame is split into."""

    columns: int = BLOCK_COLUMNS
    rows: int = BLOCK_ROWS

    @property
    def count(self) -> int:
        return self.columns * self.rows


# A module-level singleton rather than a call in each default: BlockSpec() as a
# default would be evaluated at import and shared, which is fine for a frozen
# dataclass but reads as a mutable default to both linters and the next reader.
DEFAULT_SPEC = BlockSpec()


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of consecutive frames the analysis judged alike."""

    start: float
    end: float
    kind: str  # "stable" or "change"
    score: float
    indices: tuple[int, ...]

    @property
    def span(self) -> float:
        return self.end - self.start


def residual(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """The brightness-compensated absolute difference between two frames.

    Subtracting the median before taking the absolute value is the global
    illumination compensation: a frame that got uniformly brighter has a large
    raw difference everywhere and a median that moves with it, so the residual
    stays near zero. A door opening moves part of the frame and leaves the median
    where it was, so its residual survives.

    `int16` rather than the arrays' own dtype: both are unsigned, and an 8-bit
    subtraction wraps -- a dark pixel going darker would come back as a large
    positive difference and read as a change.
    """
    delta = current.astype(np.int16) - reference.astype(np.int16)
    return np.abs(delta - np.median(delta))


def global_change(reference: np.ndarray, current: np.ndarray) -> float:
    """How much the frame changed *uniformly*.

    The absolute median of the raw difference. Compensation subtracts exactly
    this, so it is discarded by `residual` -- but discarding it silently would
    mean an exposure swing and a genuine scene change became indistinguishable in
    the metadata. Recording it keeps the two apart for whoever reads the artifact
    later, and it is the number that says "the light changed" rather than "the
    door moved".
    """
    delta = current.astype(np.int16) - reference.astype(np.int16)
    return float(abs(np.median(delta)))


def block_scores(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    spec: BlockSpec = DEFAULT_SPEC,
    pixel_threshold: float = PIXEL_THRESHOLD,
) -> np.ndarray:
    """Per-block fraction of pixels that changed, as a flat array.

    Each block reports the *proportion* of its pixels above the threshold rather
    than their mean magnitude. A proportion is bounded and comparable across
    blocks, so "the door block" and "the corridor block" can be ranked against
    each other; a mean magnitude would let one block with a few very bright
    changed pixels outrank a block where half the pixels moved a little.

    The frame is cropped to the largest exact multiple of the grid, so a preview
    whose height is not divisible by the row count is analysed rather than
    rejected -- the discarded edge is a few pixels of a deliberately low-resolution
    image, and rejecting the frame would lose the whole candidate instead.
    """
    height, width = residual(reference, current).shape
    usable_h = (height // spec.rows) * spec.rows
    usable_w = (width // spec.columns) * spec.columns
    if usable_h == 0 or usable_w == 0:
        raise ValueError("frame_too_small_for_grid")
    grid = residual(reference, current)[:usable_h, :usable_w]
    changed = (grid > pixel_threshold).astype(np.float32)
    blocks = changed.reshape(
        spec.rows, usable_h // spec.rows, spec.columns, usable_w // spec.columns
    )
    return np.asarray(blocks.mean(axis=(1, 3)).reshape(-1))


def frame_score(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    spec: BlockSpec = DEFAULT_SPEC,
    pixel_threshold: float = PIXEL_THRESHOLD,
    top_blocks: int = TOP_BLOCKS,
) -> float:
    """The frame's change score: the mean of its busiest blocks.

    See the module docstring for why this is not a whole-frame average.
    """
    scores = block_scores(
        reference, current, spec=spec, pixel_threshold=pixel_threshold
    )
    if scores.size == 0:
        return 0.0
    busiest = np.sort(scores)[-min(top_blocks, scores.size) :]
    return float(busiest.mean())


def segment_frames(
    times: Sequence[float],
    scores: Sequence[float],
    *,
    change_threshold: float,
) -> list[Segment]:
    """Group consecutive scored frames into stable and change segments.

    A frame at or above `change_threshold` extends a change segment; below it
    extends a stable one. The grouping is strictly local -- adjacent frames only,
    never a global pass -- and that is a requirement rather than an implementation
    detail. "The door closed, opened, then closed again" contains two frames of a
    closed door that are nearly identical to each other; merging them would
    collapse the two ends of the activity into one cell and lose the opening
    between them entirely.

    `scores[i]` describes the change from `times[i-1]` to `times[i]`, so the
    first score belongs to the second timestamp. The first frame therefore joins
    whatever segment the second one starts: a single frame cannot be its own
    segment, and inventing a zero score for it would claim a comparison nobody
    made.
    """
    if not times:
        return []
    if len(times) != len(scores) + 1:
        raise ValueError("scores_must_describe_adjacent_pairs")

    segments: list[Segment] = []
    current_kind = "stable" if not scores or scores[0] < change_threshold else "change"
    start_index = 0
    for index, score in enumerate(scores):
        kind = "change" if score >= change_threshold else "stable"
        if kind != current_kind:
            segments.append(
                _build_segment(times, scores, start_index, index, current_kind)
            )
            start_index = index + 1
            current_kind = kind
    segments.append(
        _build_segment(times, scores, start_index, len(times) - 1, current_kind)
    )
    return segments


def _build_segment(
    times: Sequence[float],
    scores: Sequence[float],
    start_index: int,
    end_index: int,
    kind: str,
) -> Segment:
    """Build one segment. `end_index` is inclusive and indexes `times`."""
    indices = tuple(range(start_index, end_index + 1))
    # The score that classified a segment is the one that opened it, except for
    # the first segment whose opening comparison belongs to its second frame.
    relevant = [
        scores[index]
        for index in range(max(0, start_index - 1), min(len(scores), end_index))
    ]
    mean_score = float(np.mean(relevant)) if relevant else 0.0
    return Segment(
        start=float(times[start_index]),
        end=float(times[end_index]),
        kind=kind,
        score=mean_score,
        indices=indices,
    )


def polygon_bounds(
    polygon: Sequence[tuple[float, float]], *, width: int, height: int
) -> tuple[int, int, int, int]:
    """Return the pixel bounding box of a normalized polygon, clamped.

    Frigate stores zone coordinates as fractions of the frame with diagonals, so
    the box is derived from the vertices rather than assumed. Clamped to the frame
    because a zone drawn at the edge can round one pixel outside it, and a crop
    reaching past the array would raise for a region the user can see is valid.

    Raises ValueError for an empty polygon: a zone with no vertices is a parse
    failure upstream, not an empty region.
    """
    if not polygon:
        raise ValueError("empty_polygon")
    xs = [point[0] * width for point in polygon]
    ys = [point[1] * height for point in polygon]
    left = max(0, min(int(np.floor(min(xs))), width - 1))
    top = max(0, min(int(np.floor(min(ys))), height - 1))
    right = min(width, max(int(np.ceil(max(xs))), left + 1))
    bottom = min(height, max(int(np.ceil(max(ys))), top + 1))
    return left, top, right, bottom


def mask_for_polygon(
    polygon: Sequence[tuple[float, float]],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize a normalized polygon into a boolean mask of the whole frame.

    A mask rather than a plain crop, because a Frigate zone is a polygon with
    diagonal edges: cropping its bounding box would include the wall outside the
    zone, and a change in that wall would then score as if it happened inside the
    region the user drew. Measured on this deployment, `home_door` fills 18.2% of
    the frame while its bounding box fills 30.8% -- so more than a third of the
    box is not the zone.

    The even-odd rule is implemented directly rather than pulling in a drawing
    dependency: the polygons are small and the test is a handful of comparisons
    per pixel, all in numpy.

    Vertices are validated here, not only at the Frigate boundary. `frigate.py`
    already rejects a non-finite coordinate when it parses `/api/config`, so a NaN
    cannot arrive from that path today -- but this function is public, and with a
    NaN it does not fail: it quietly returns a wrong area (measured: a polygon
    with one NaN vertex produced 4608 masked pixels instead of raising). A region
    the user drew would then be applied to the wrong pixels with nothing to
    notice it by. The check is cheap and belongs where the assumption is used.
    """
    if len(polygon) < 3:
        raise ValueError("polygon_needs_three_vertices")
    for point in polygon:
        if (
            not isinstance(point, (tuple, list))
            or len(point) != 2
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in point
            )
        ):
            raise ValueError("polygon_vertex_invalid")
    xs = np.array([point[0] * width for point in polygon], dtype=np.float64)
    ys = np.array([point[1] * height for point in polygon], dtype=np.float64)
    columns = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(columns, rows)

    inside = np.zeros((height, width), dtype=bool)
    count = len(polygon)
    for index in range(count):
        next_index = (index + 1) % count
        x1, y1 = xs[index], ys[index]
        x2, y2 = xs[next_index], ys[next_index]
        if y1 == y2:
            continue
        # Half-open in y so a vertex is counted once, which is what keeps a
        # horizontal edge from flipping the parity twice.
        crosses = ((y1 > grid_y) != (y2 > grid_y))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (x2 - x1) * (grid_y - y1) / (y2 - y1) + x1
        inside ^= crosses & (grid_x < x_at)
    return inside


def focus_region_change(
    reference: np.ndarray,
    current: np.ndarray,
    polygon: Sequence[tuple[float, float]],
    *,
    pixel_threshold: float = PIXEL_THRESHOLD,
) -> float:
    """The proportion of a zone's pixels that changed.

    Reported as a proportion of the *zone*, not of the frame: the question this
    answers is "how much of the region the user drew has changed", and dividing by
    the frame would make a small zone's total change unreachable no matter how
    completely it changed.
    """
    height, width = reference.shape
    mask = mask_for_polygon(polygon, width=width, height=height)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    difference = residual(reference, current)
    return float(((difference > pixel_threshold) & mask).sum() / total)


@dataclass(frozen=True, slots=True)
class ScoredFrame:
    """One analysed instant: what it looks like, and how much it differs."""

    timestamp: float
    # Change against the previous frame -- catches a sudden event.
    sudden: float
    # Change against the current segment's first frame -- catches a slow one that
    # no adjacent pair would notice.
    cumulative: float
    # Uniform brightness movement since the previous frame, recorded rather than
    # compensated away so an exposure swing is visible in the metadata.
    global_shift: float
    # Change inside each configured zone, keyed by zone name.
    zones: Mapping[str, float]

    @property
    def score(self) -> float:
        """The number the segmenter thresholds on.

        The larger of the two comparisons, so a moment qualifies if EITHER kind of
        change happened there. Taking the sum would let two half-changes stack into
        a change neither comparison supports.
        """
        return max(self.sudden, self.cumulative)

    @property
    def best_zone(self) -> float:
        """The busiest configured zone, or 0.0 when none are configured."""
        return max(self.zones.values()) if self.zones else 0.0


def analyze_sequence(
    frames: Sequence[tuple[float, np.ndarray]],
    *,
    zones: Mapping[str, Sequence[tuple[float, float]]] = {},
    spec: BlockSpec = DEFAULT_SPEC,
    pixel_threshold: float = PIXEL_THRESHOLD,
    anchor_reset_threshold: float | None = None,
) -> list[ScoredFrame]:
    """Score an ordered run of grayscale frames.

    Two references per frame, for the reason in the module docstring: the previous
    frame answers "did something just happen", and the segment's first frame
    answers "has something been happening". The second reference is reset whenever
    a change segment begins, so a long slow opening accumulates against the frame
    the segment started from rather than against a window that keeps sliding away
    from it.

    `anchor_reset_threshold` is what keeps that second reference meaningful: when a
    frame's cumulative score reaches it, that frame becomes the new anchor. Without
    the reset the anchor stays at the start of the run, so a minute of standing
    still after one door movement would keep reporting a large cumulative
    difference against a moment that is no longer relevant.

    Frames must arrive in timestamp order; that is the caller's contract rather
    than something re-sorted here, because the segment anchors depend on which
    frame opened a segment and silently reordering would change the result.
    """
    if not frames:
        return []
    scored: list[ScoredFrame] = []
    anchor = frames[0][1]
    previous = frames[0][1]
    for index, (timestamp, frame) in enumerate(frames):
        if index == 0:
            scored.append(
                ScoredFrame(
                    timestamp=timestamp,
                    sudden=0.0,
                    cumulative=0.0,
                    global_shift=0.0,
                    zones={name: 0.0 for name in zones},
                )
            )
            continue
        sudden = frame_score(
            previous, frame, spec=spec, pixel_threshold=pixel_threshold
        )
        cumulative = frame_score(
            anchor, frame, spec=spec, pixel_threshold=pixel_threshold
        )
        scored.append(
            ScoredFrame(
                timestamp=timestamp,
                sudden=sudden,
                cumulative=cumulative,
                global_shift=global_change(previous, frame),
                zones={
                    name: focus_region_change(
                        anchor,
                        frame,
                        polygon,
                        pixel_threshold=pixel_threshold,
                    )
                    for name, polygon in zones.items()
                },
            )
        )
        if anchor_reset_threshold is not None and cumulative >= anchor_reset_threshold:
            # The change has been seen; the following frames should accumulate
            # against this new state rather than against the one before it.
            anchor = frame
        previous = frame
    return scored


def segment_scored_frames(
    scored: Sequence[ScoredFrame], *, change_threshold: float
) -> list[Segment]:
    """Group scored frames into segments using their combined score.

    A thin adapter over `segment_frames` so callers do not have to unzip the
    timestamps and scores themselves -- and, more importantly, so the threshold
    is applied to `ScoredFrame.score` in exactly one place.
    """
    times = [item.timestamp for item in scored]
    scores = [item.score for item in scored[1:]]
    return segment_frames(times, scores, change_threshold=change_threshold)
