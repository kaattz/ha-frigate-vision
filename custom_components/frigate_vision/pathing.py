"""Pure helpers for Frigate person motion paths.

This module deliberately has no Home Assistant imports so the selection
algorithm can be unit tested without a Home Assistant runtime.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class InvalidPathData(ValueError):
    """Frigate path_data exists but violates the public contract."""


@dataclass(frozen=True, slots=True)
class MotionPath:
    """One person event window with normalized (timestamp, x, y) points."""

    start_time: float
    end_time: float
    points: tuple[tuple[float, float, float], ...]


def parse_path_data(
    payload: Mapping[str, Any],
) -> tuple[tuple[float, float, float], ...] | None:
    """Return normalized path points, or None when path_data is absent."""
    data = payload.get("data")
    if not isinstance(data, Mapping) or "path_data" not in data:
        return None
    path_data = data["path_data"]
    if not isinstance(path_data, (list, tuple)):
        raise InvalidPathData("invalid_path_data")
    points: list[tuple[float, float, float]] = []
    for point in path_data:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise InvalidPathData("invalid_path_data")
        coordinates, timestamp = point
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
            raise InvalidPathData("invalid_path_data")
        raw_x, raw_y = coordinates
        if any(isinstance(value, bool) for value in (raw_x, raw_y, timestamp)):
            raise InvalidPathData("invalid_path_data")
        try:
            x = float(raw_x)
            y = float(raw_y)
            occurred_at = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise InvalidPathData("invalid_path_data") from exc
        if not all(math.isfinite(value) for value in (x, y, occurred_at)):
            raise InvalidPathData("invalid_path_data")
        if occurred_at < 0:
            raise InvalidPathData("invalid_path_data")
        points.append((occurred_at, x, y))
    return tuple(points)


def select_motion_times(
    paths: Sequence[MotionPath],
    *,
    count: int = 3,
    lower: float | None = None,
    upper: float | None = None,
    extra: Sequence[tuple[float, float]] = (),
) -> tuple[float, ...] | None:
    """Pick high-motion timestamps that also stay spread across the activity.

    Two properties matter, and they pull against each other:

    * **Motion.** A frame is worth taking where the picture changed, so the
      candidate instants are ranked by how far the person moved.
    * **Coverage.** The three picks become the middle of a six-cell sheet whose
      outer cells are the person's first and last frames. If the picks cluster,
      everything between them is invisible to the model -- and a door opening,
      a hand-off, or a turn is exactly the kind of short event that lands in
      such a hole.

    Measured on this deployment, ranking by motion alone failed the second
    property badly. An activity whose path points were dense in its first half
    got two picks inside 3.8s of each other while the next sat 33.3s later; the
    moment the entry door opened fell in that 33.3s gap and the model, with no
    frame showing it, answered `home_arrival` for someone walking *out*.

    Splitting the movement list into equal groups **by index** is what allowed
    that: index thirds only equal time thirds when the points are spread
    evenly, and Frigate emits points as the detector reports them, which is
    anything but even.

    `extra` carries instants discovered by *probing* a gap for visible change.
    The detector reports where the person moved, which is not the same as where
    the picture changed: a door opening while the person stands still produces
    no path movement at all. Measured here, every hole wider than 12s that was
    probed contained real change (mean greyscale difference 2.45-85.66, against
    a threshold of 1.0), so those holes are not empty corridors. Offering the
    probed instants as ordinary candidates lets the same worst-gap rule decide
    whether to trade a motion pick for one, instead of a second heuristic
    guessing which cell to sacrifice.

    Returns None when the path data cannot fill every group, which is the
    caller's signal to fall back to image-change selection.
    """
    if count <= 0:
        return None
    points: list[tuple[float, float, float]] = []
    for path in paths:
        for occurred_at, x, y in path.points:
            if path.start_time <= occurred_at <= path.end_time:
                points.append((occurred_at, x, y))
    if len(points) < count + 1:
        return None
    ordered = sorted(points)
    movements: list[tuple[float, float]] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        _, previous_x, previous_y = previous
        occurred_at, x, y = current
        movements.append((occurred_at, math.hypot(x - previous_x, y - previous_y)))
    if len(movements) < count:
        return None
    eligible_movements = [
        item
        for item in movements
        if (lower is None or item[0] > lower) and (upper is None or item[0] < upper)
    ]
    if len(eligible_movements) < count:
        return None

    # Choose the combination that minimises the worst unobserved stretch, and
    # break ties towards more movement.
    #
    # Equal thirds of the window were tried first and are not good enough.
    # Measured on activity 19:05, whose movements sit at relative seconds 0.2,
    # 1.8, 5.6, 21.4, 23.3, ...: index thirds picked 1.8/34.5/42.5 (a 32.7s
    # hole), and time thirds picked 1.8/23.7/35.1 (a 21.9s hole). Both left the
    # entry door opening -- 6.2s in -- unobserved, and the model, shown no frame
    # of it, called a person walking *out* a `home_arrival`.
    #
    # Minimising the maximum gap instead picks 5.6/21.4/35.1 on the same data:
    # a 17.0s worst gap with a pick 0.58s from the door. That is the difference
    # between a rule that merely spreads picks and one that actually aims at
    # covering the activity.
    #
    # The search is exhaustive: `count` is 3 and the candidate list is bounded
    # by Frigate's own path-point density (measured: 14-18 movements per
    # activity), so this is a few hundred to a few thousand combinations of
    # arithmetic -- negligible beside a single snapshot fetch.
    span_start = lower if lower is not None else eligible_movements[0][0]
    span_end = upper if upper is not None else eligible_movements[-1][0]
    if span_end <= span_start:
        return None

    def worst_gap(combo: Sequence[tuple[float, float]]) -> float:
        edges = [span_start, *(item[0] for item in combo), span_end]
        return max(b - a for a, b in zip(edges, edges[1:], strict=False))

    # Probed instants join as ordinary candidates. Their "distance" is the
    # visible change measured there, which is the same *idea* -- how much
    # happened at that instant -- but not the same *scale*: a path movement is
    # normalised frame coordinates (measured here at 0.01-0.15) while a
    # greyscale change is 0-255 (measured at 2-85). Comparing them directly
    # would let the tie-break below always prefer a probe over a motion pick,
    # regardless of merit.
    #
    # So both sources are scaled to [0, 1] against their own peers first. The
    # tie-break then means "prefer the instant where relatively more happened",
    # which is a claim each source can actually support.
    by_instant: dict[float, float] = {}
    for occurred_at, weight in _relative_weights(eligible_movements).items():
        if weight > by_instant.get(occurred_at, -1.0):
            by_instant[occurred_at] = weight
    for occurred_at, weight in _relative_weights(extra).items():
        if lower is not None and occurred_at <= lower:
            continue
        if upper is not None and occurred_at >= upper:
            continue
        if weight > by_instant.get(occurred_at, -1.0):
            by_instant[occurred_at] = weight
    candidates = sorted(by_instant.items())
    if len(candidates) < count:
        return None

    def score(combo: Sequence[tuple[float, float]]) -> tuple[tuple[float, ...], float]:
        edges = [span_start, *(item[0] for item in combo), span_end]
        gaps = sorted(
            (b - a for a, b in zip(edges, edges[1:], strict=False)), reverse=True
        )
        # Compare the gaps *worst-first*, not just the worst one. Minimising the
        # single largest gap leaves ties, and an arbitrary tie-break spends two
        # cells on adjacent instants while a quieter stretch keeps its hole --
        # measured on a fixture whose movements span 104-109 inside a 103-117
        # window, where the largest gap is set by the quiet tail either way and
        # only the second-largest gap distinguishes a good spread from a
        # clustered one.
        #
        # Ties then go to the combination that saw more movement, so a pick
        # prefers a moment something happened over a moment nothing did.
        return (tuple(gaps), -sum(distance for _, distance in combo))

    best_combo = min(itertools.combinations(candidates, count), key=score)
    selected = sorted(occurred_at for occurred_at, _ in best_combo)
    if len(set(selected)) != count:
        return None
    return tuple(selected)


def _relative_weights(
    items: Sequence[tuple[float, float]],
) -> dict[float, float]:
    """Scale one source's magnitudes to [0, 1] within itself.

    Movement distances and greyscale differences are measured in different units
    and cannot be compared directly, but each answers the same question inside
    its own source: how much happened here, relative to the rest of this source.
    Normalising per source makes the two comparable without inventing a
    conversion between coordinate distance and pixel brightness.
    """
    if not items:
        return {}
    highest = max(value for _, value in items)
    if highest <= 0:
        # Nothing moved or nothing changed: every instant is equally
        # unremarkable, so they all weigh the same.
        return {occurred_at: 0.0 for occurred_at, _ in items}
    return {occurred_at: value / highest for occurred_at, value in items}


def probe_gap_times(
    sheet_times: Sequence[float],
    *,
    max_gap: float,
    probes: int = 3,
    within: tuple[float, float] | None = None,
) -> tuple[float, ...]:
    """Instants worth fetching inside the sheet's widest unobserved stretch.

    The sheet has a fixed six cells, so a hole cannot be closed by adding one:
    the picks have to be re-aimed, and that needs frames to re-aim at.

    Returns instants to fetch, not frames -- the caller fetches them and scores
    the result, because only real pixels can say whether anything happened.

    Empty when no hole is wider than `max_gap`. That bound is the whole economy
    of this function: every probe costs a snapshot fetch, and a hole shorter than
    the shortest action worth seeing cannot be hiding one.

    `within` excludes the stretch beyond the activity. The sheet's last two
    cells are the person's final frame and the emptied scene, which are *meant*
    to be far apart -- nothing happens between them by definition -- so probing
    there would spend fetches on an empty corridor and could displace a pick
    that was covering the activity itself.
    """
    if probes <= 0 or len(sheet_times) < 2:
        return ()
    ordered = sorted(sheet_times)
    widest = 0.0
    window: tuple[float, float] | None = None
    for start, end in zip(ordered, ordered[1:], strict=False):
        if within is not None:
            # Clip the candidate hole to the activity, then judge what remains.
            clipped_start = max(start, within[0])
            clipped_end = min(end, within[1])
            if clipped_end <= clipped_start:
                continue
        else:
            clipped_start, clipped_end = start, end
        span = clipped_end - clipped_start
        if span > widest:
            widest = span
            window = (clipped_start, clipped_end)
    if window is None or widest <= max_gap:
        return ()
    start, end = window
    # Evenly spaced and strictly inside: an instant on an existing cell would
    # duplicate a frame the sheet already has.
    return tuple(
        start + (end - start) * (index + 1) / (probes + 1)
        for index in range(probes)
    )
