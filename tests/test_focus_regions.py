"""Tests for focus-region masking and cropping.

A Frigate zone is a POLYGON. Cropping its bounding box instead would include the
wall outside the zone, and a change in that wall would then score as though it
happened inside the region the user drew. Measured against this deployment's own
zones at preview resolution, the box is 30.8% of the frame while the polygon is
18.2% -- so more than a third of the box is not the zone at all.

The polygons below are copied verbatim from `GET /api/config` on the real
Frigate, which is why they have six and seven vertices rather than four.
"""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.frigate_vision.changes import (
    focus_region_change,
    mask_for_polygon,
    polygon_bounds,
)

# The deployment's two zones, normalized.
HOME_DOOR = [
    (0.181, 0.175),
    (0.407, 0.125),
    (0.442, 0.677),
    (0.456, 0.738),
    (0.533, 0.981),
    (0.286, 1.0),
]
ELEVATOR = [
    (0.411, 0.083),
    (0.68, 0.048),
    (0.662, 0.553),
    (0.713, 0.768),
    (0.516, 0.909),
    (0.471, 0.766),
    (0.445, 0.672),
]

WIDTH, HEIGHT = 320, 180


def _flat(value: int = 100) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), value, dtype=np.uint8)


class TestPolygonBounds:
    """The bounding box must come from the vertices, clamped to the frame."""

    def test_the_box_of_a_real_zone(self) -> None:
        left, top, right, bottom = polygon_bounds(
            HOME_DOOR, width=WIDTH, height=HEIGHT
        )
        # Derived from the vertices: min x is 0.181*320=57.9 -> 57; min y is
        # 0.125*180=22.5 -> 22 (the second vertex, not the first -- 0.175*180=31.5
        # is higher on screen); max x is 0.533*320=170.6 -> 171; max y 1.0*180=180.
        assert (left, top, right, bottom) == (57, 22, 171, 180)

    def test_a_zone_at_the_edge_is_clamped_into_the_frame(self) -> None:
        """A zone drawn at the border can round one pixel outside it."""
        box = polygon_bounds(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            width=WIDTH,
            height=HEIGHT,
        )
        assert box == (0, 0, WIDTH, HEIGHT)

    def test_an_empty_polygon_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty_polygon"):
            polygon_bounds([], width=WIDTH, height=HEIGHT)


class TestTheMaskFollowsThePolygon:
    """The mask is the zone, not its box."""

    def test_the_mask_is_smaller_than_its_bounding_box(self) -> None:
        """THE reason a box crop is not enough.

        Measured on the real zone: the box is 31.3% of the frame and the polygon
        18.2%, so 41.7% of the box is wall. A change in that wall would score as
        a change inside the region the user drew.
        """
        mask = mask_for_polygon(HOME_DOOR, width=WIDTH, height=HEIGHT)
        left, top, right, bottom = polygon_bounds(
            HOME_DOOR, width=WIDTH, height=HEIGHT
        )
        box_area = (right - left) * (bottom - top)
        assert mask.sum() < box_area * 0.7, (
            f"多边形必须明显小于其包围盒：mask={mask.sum()} box={box_area}"
        )
        assert mask.sum() > 0

    def test_a_full_frame_rectangle_masks_everything(self) -> None:
        mask = mask_for_polygon(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            width=WIDTH,
            height=HEIGHT,
        )
        assert mask.all()

    def test_a_vertex_inside_the_polygon_is_masked(self) -> None:
        mask = mask_for_polygon(ELEVATOR, width=WIDTH, height=HEIGHT)
        # The lift door's own centroid, in normalized terms.
        xs = [point[0] for point in ELEVATOR]
        ys = [point[1] for point in ELEVATOR]
        column = int(sum(xs) / len(xs) * WIDTH)
        row = int(sum(ys) / len(ys) * HEIGHT)
        assert mask[row, column], "多边形质心必须落在掩码内"

    def test_a_point_far_outside_is_not_masked(self) -> None:
        mask = mask_for_polygon(ELEVATOR, width=WIDTH, height=HEIGHT)
        assert not mask[5, 10], "画面左上角远在电梯 zone 之外"

    def test_fewer_than_three_vertices_is_refused(self) -> None:
        with pytest.raises(ValueError, match="polygon_needs_three_vertices"):
            mask_for_polygon([(0.1, 0.1), (0.2, 0.2)], width=WIDTH, height=HEIGHT)

    def test_a_concave_polygon_is_handled(self) -> None:
        """An L-shape must not fill its own notch.

        Frigate zones are drawn freehand, so concave shapes are ordinary. The
        even-odd rule handles them; a convex-hull shortcut would not.
        """
        # An L: the vertical arm is x in [0, 0.5), the horizontal arm is
        # y in [0.5, 1). The notch is x >= 0.5 AND y < 0.5.
        elbow = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (1.0, 0.5),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        mask = mask_for_polygon(elbow, width=WIDTH, height=HEIGHT)
        assert mask[80, 40], "竖直臂内部"
        assert mask[160, 260], "水平臂内部"
        assert not mask[45, 240], "L 形的缺口（x>=0.5 且 y<0.5）不能被填上"


class TestFocusRegionChange:
    """The score answers "how much of the zone changed"."""

    def test_a_change_inside_the_zone_scores_high(self) -> None:
        before = _flat()
        after = before.copy()
        # A block well inside the home-door zone's mask.
        after[110:150, 80:150] = 220
        assert focus_region_change(before, after, HOME_DOOR) > 0.2

    def test_a_change_outside_every_zone_scores_zero(self) -> None:
        """The property that makes the regions worth configuring."""
        before = _flat()
        after = before.copy()
        after[100:140, 250:310] = 220
        assert focus_region_change(before, after, HOME_DOOR) == 0.0
        assert focus_region_change(before, after, ELEVATOR) == 0.0

    def test_a_uniform_brightness_shift_scores_zero(self) -> None:
        """Compensation applies inside a zone exactly as it does frame-wide."""
        assert focus_region_change(_flat(100), _flat(160), HOME_DOOR) == 0.0

    def test_the_denominator_is_the_zone_not_the_frame(self) -> None:
        """A small zone that changes completely must reach a high score.

        Dividing by the frame instead would cap a 18% zone at 0.18 no matter how
        completely it changed, which is the opposite of what the region is for.
        """
        before = _flat()
        mask = mask_for_polygon(ELEVATOR, width=WIDTH, height=HEIGHT)
        after = before.copy()
        after[mask] = 220
        assert focus_region_change(before, after, ELEVATOR) > 0.95

    def test_a_zone_covering_nothing_scores_zero(self) -> None:
        """A degenerate polygon cannot divide by zero."""
        sliver = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
        assert focus_region_change(_flat(), _flat(), sliver) == 0.0


class TestMalformedVerticesAreRefused:
    """A NaN vertex must raise, not silently produce a wrong region.

    rigate.py already rejects a non-finite coordinate when it parses the camera
    config, so this cannot arrive from Frigate today. But mask_for_polygon is
    public and a NaN does not fail in it: measured, a polygon with one NaN vertex
    returned 4608 masked pixels instead of raising, which would apply a region the
    user drew to the wrong pixels with nothing to notice it by.
    """

    def test_a_nan_vertex_is_refused(self) -> None:
        nan = float("nan")
        with pytest.raises(ValueError, match="polygon_vertex_invalid"):
            mask_for_polygon(
                [(0.1, 0.1), (0.5, 0.1), (nan, 0.5), (0.1, 0.9)],
                width=320,
                height=180,
            )

    def test_an_infinite_vertex_is_refused(self) -> None:
        with pytest.raises(ValueError, match="polygon_vertex_invalid"):
            mask_for_polygon(
                [(0.1, 0.1), (0.5, 0.1), (float("inf"), 0.5)],
                width=320,
                height=180,
            )

    def test_a_boolean_vertex_is_refused(self) -> None:
        """True is an int in Python and would pass a naive numeric check."""
        with pytest.raises(ValueError, match="polygon_vertex_invalid"):
            mask_for_polygon(
                [(0.1, 0.1), (0.5, 0.1), (True, 0.5)],
                width=320,
                height=180,
            )

    def test_a_wrong_shaped_vertex_is_refused(self) -> None:
        with pytest.raises(ValueError, match="polygon_vertex_invalid"):
            mask_for_polygon(
                [(0.1, 0.1), (0.5, 0.1), (0.2, 0.3, 0.4)],
                width=320,
                height=180,
            )

    def test_a_well_formed_polygon_still_works(self) -> None:
        mask = mask_for_polygon(HOME_DOOR, width=WIDTH, height=HEIGHT)
        assert mask.sum() > 0
