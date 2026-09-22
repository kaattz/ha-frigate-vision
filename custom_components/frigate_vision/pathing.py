"""Pure helpers for Frigate person motion paths.

This module deliberately has no Home Assistant imports so the selection
algorithm can be unit tested without a Home Assistant runtime.
"""

from __future__ import annotations

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
) -> tuple[float, ...] | None:
    """Pick highest-motion timestamps, one per equal movement third.

    Movement intervals are grouped by index so the picks span the whole
    activity instead of clustering on one burst. Returns None when the path
    data cannot fill every group. Ties prefer the later timestamp.
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
    selected: list[float] = []
    total = len(movements)
    for group in range(count):
        group_start = total * group // count
        group_end = total * (group + 1) // count
        chunk = movements[group_start:group_end]
        eligible = [
            item
            for item in chunk
            if (lower is None or item[0] > lower) and (upper is None or item[0] < upper)
        ]
        if not eligible:
            return None
        selected.append(max(eligible, key=lambda item: (item[1], item[0]))[0])
    unique = sorted(set(selected))
    if len(unique) != count:
        return None
    return tuple(unique)
