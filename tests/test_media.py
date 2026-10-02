from __future__ import annotations

import asyncio
import json
import time
from io import BytesIO
from pathlib import Path

import pytest
from homeassistant.core import HomeAssistant
from PIL import Image, ImageChops

from custom_components.frigate_vision.const import (
    CONF_PERSON_HIGHLIGHT,
    HIGHLIGHT_SEAM,
)
from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import (
    EvidencePlan,
    MediaError,
    MediaManager,
    build_contact_sheet,
    frame_is_infrared,
    frame_is_overexposed,
    largest_person_box,
    pair_times_with_roles,
    recordings_cover,
    select_review_change_frames,
    validate_unique_frames,
)
from custom_components.frigate_vision.media_source import DATA_MEDIA_REGISTRY
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
)
from custom_components.frigate_vision.store import ActivityStore
from custom_components.frigate_vision.vision import (
    evidence_width_budget,
    resize_for_provider,
    vision_config_from,
)


def _jpeg(value: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 36), (value, value, value)).save(output, "JPEG")
    return output.getvalue()


def _ir_jpeg(value: int = 160) -> bytes:
    """A frame with the camera's night-vision cast.

    Measured on production frames: an IR-lit cell has a strongly green-dominant
    profile (R-G around -150) while colour frames sit near R-G = +2.
    """
    output = BytesIO()
    Image.new("RGB", (64, 36), (5, value, 105)).save(output, "JPEG")
    return output.getvalue()


def _blown_jpeg(shade: int = 250) -> bytes:
    """A saturated frame, as when the lift door opens into the lens.

    Measured on production sheets: such cells are 42%-98% near-white pixels
    while ordinary cells stay under 1%, so the condition separates cleanly.
    Nearly white but not pure, matching the real frames rather than an ideal.

    `shade` varies the near-white value. A camera's saturated frames are similar
    but never byte-identical, and returning one fixed image for every blown
    instant made several cells exactly equal -- which the uniqueness guard then
    rejected, correctly. A test holding several blown frames must vary the shade,
    or it asserts something a real camera cannot produce.
    """
    output = BytesIO()
    value = max(236, min(254, shade))
    Image.new("RGB", (64, 36), (value, value, value - 4)).save(output, "JPEG")
    return output.getvalue()


def _blown_at(timestamp: float) -> bytes:
    """A saturated frame whose exact shade depends on the instant."""
    return _blown_jpeg(240 + int(timestamp * 3) % 12)


def _changing_jpeg(timestamp: float) -> bytes:
    """A frame whose brightness varies smoothly and monotonically with time.

    Fixtures must return a *different* picture for a different instant, because
    that is what a camera does. The obvious `int(t * 10) % 255` does not: it is a
    sawtooth, so instants 25.5s apart (and, at coarser multipliers, 0.1s apart)
    collapse onto the same grey and the uniqueness guard then rejects a sheet the
    fixture wrongly made identical. Deriving the value from a bounded,
    monotonic function of the timestamp keeps every distinct instant distinct.
    """
    # 12 levels per second, saturating smoothly, over the grey range a camera
    # would show here. Distinct instants differ by at least one level.
    level = int((timestamp * 12) % 200) + 20
    return _jpeg(level)


def test_probe_cells_keep_their_own_role_through_the_sort() -> None:
    """Every role must stay attached to the instant it describes.

    Probes land *inside* the motion span, so sorting the timestamps and leaving
    the roles in insertion order silently pairs each role with the wrong cell. It
    matters because `_nudge_uncovered_frames` treats `first`/`last`/`postroll` as
    required and a probe as droppable -- so a mispaired role lets a frame the
    sheet cannot be built without be discarded.

    Tested on the pure helper, not through `async_build`: a mutation that sorted
    the times while leaving roles in insertion order passed the whole suite when
    the pairing lived inside the async method, because nothing observed the two
    lists as a pair.
    """

    plan = EvidencePlan(
        mode="review_six",
        first_time=100.0,
        last_time=160.0,
        postroll_time=163.0,
        selection_source="path_motion",
        motion_times=(110.0, 130.0, 150.0),
    )
    # Two probes fall between motion picks, one after the last -- the arrangement
    # that makes sorting dangerous.
    times, roles = pair_times_with_roles(plan, (115.0, 135.0, 155.0))

    assert len(times) == len(roles) == 9
    assert list(times) == sorted(times)
    expected = {
        100.0: "first",
        110.0: "motion",
        115.0: "probe",
        130.0: "motion",
        135.0: "probe",
        150.0: "motion",
        155.0: "probe",
        160.0: "last",
        163.0: "postroll",
    }
    assert dict(zip(times, roles, strict=True)) == expected


def test_contact_sheet_accepts_nine_cells_as_three_rows(tmp_path) -> None:
    """Nine cells must lay out as 3x3, not be rejected or silently truncated.

    Measured on the owner's labelled sheets: a nine-cell sheet answered direction
    correctly 48% of the time against 28% for six cells (pooled over two models,
    n=40), and the mechanism is not in doubt -- the three extra frames are chosen
    by visible change and cut the worst unobserved gap from 17.1s to 11.5s, better
    on 29 of 48 activities and worse on none.

    The layout must follow the frame count rather than a fixed 2x3, or the extra
    row would be dropped without error -- which is exactly how a malformed
    nine-cell fixture of mine once produced an inflated result.
    """
    frames = []
    for index in range(9):
        path = tmp_path / f"n{index}.jpg"
        path.write_bytes(_jpeg(index * 20))
        frames.append(path)
    output = tmp_path / "nine.jpg"
    build_contact_sheet(frames, output, cell_size=(160, 90))
    with Image.open(output) as sheet:
        assert sheet.size == (480, 270), "three rows of 160x90 was expected"


def test_contact_sheet_rejects_a_count_that_is_not_whole_rows(tmp_path) -> None:
    """A partial row would silently drop frames, so it must fail loudly."""
    frames = []
    for index in range(7):
        path = tmp_path / f"p{index}.jpg"
        path.write_bytes(_jpeg(index * 20))
        frames.append(path)
    with pytest.raises(MediaError, match="invalid_frame_count"):
        build_contact_sheet(frames, tmp_path / "bad.jpg", cell_size=(160, 90))


def test_contact_sheet_validates_frames_and_dimensions(tmp_path) -> None:
    frames = []
    for index in range(6):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(_jpeg(index * 30))
        frames.append(path)
    output = tmp_path / "sheet.jpg"
    build_contact_sheet(frames, output, cell_size=(160, 90))
    with Image.open(output) as sheet:
        assert sheet.size == (480, 180)
        assert sheet.format == "JPEG"
    frames[0].write_text("broken")
    with pytest.raises(MediaError, match="frame_decode_failed"):
        build_contact_sheet(frames[:3], output)


# The composed sheet: a three-across, three-down grid of 640x360 cells with the
# highlight column appended on the right. The numbers are the pipeline's own --
# a 767-wide provider target scales 1920x1080 to 767x431 (1080 * 767 / 1920).
_GRID_WIDTH = 3 * 640
_GRID_HEIGHT = 3 * 360
_HIGHLIGHT_WIDTH = 448
_TARGET_WIDTH = 767
_SCALED_HEIGHT = 431
# JPEG encodes luma in 8x8 blocks but chroma at half resolution, so the chroma
# blocks straddling the seam between the grid and the column are 16px wide and
# the column's presence perturbs up to one macroblock of grid to its left.
# Measured on this fixture: the first differing column is 16px from the seam.
# The margin is set to two macroblocks so a different libjpeg build cannot make
# this assertion flaky; everything further left must be identical.
_SEAM_MARGIN = 32


def _solid_jpeg(
    size: tuple[int, int], colour: tuple[int, int, int] = (200, 30, 40)
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, colour).save(output, "JPEG")
    return output.getvalue()


def _nine_frames(tmp_path: Path) -> list[Path]:
    frames = []
    for index in range(9):
        path = tmp_path / f"g{index}.jpg"
        path.write_bytes(_jpeg(index * 20))
        frames.append(path)
    return frames


def _sealed_review_record(activity_id: str) -> ActivityRecord:
    """A SEALED record from a Frigate *review*, which carries no box of its own.

    This is the shape that matters in practice: every activity this household has
    recorded came from a review, not from a raw event. A review message lists
    `detections` but no box -- the box only exists on the detection -- so this
    record starts with `box_updates` empty and the manager has to go and ask.
    """
    return ActivityRecord(
        activity_id=activity_id,
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=130,
        camera="front",
        review_ids=("review_1",),
        detection_ids=("event_1",),
        association_deadline=140,
        finalization_deadline=220,
    )


def _sealed_record_with_boxes(
    activity_id: str,
    *,
    boxes: tuple[tuple[float, tuple[float, float, float, float]], ...] = (
        (102.0, (0.20, 0.30, 0.10, 0.20)),
        (110.0, (0.30, 0.35, 0.32, 0.55)),
    ),
) -> ActivityRecord:
    """A SEALED review record, the shape the media manager can build from.

    The review layout derives its sample times from the event window; `boxes`
    is the per-instant person box the close-up is chosen from. The detection
    zone updates a door cycle once planned from are carried along verbatim:
    the review planner must ignore them, and keeping them here proves it does.
    """
    return ActivityRecord(
        activity_id=activity_id,
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=130,
        camera="front",
        detection_ids=("event_1",),
        detection_zone_updates=(
            ("event_1", 102, ("near",)),
            ("event_1", 110, ("mid",)),
            ("event_1", 120, ("far",)),
        ),
        box_updates=boxes,
        association_deadline=140,
        finalization_deadline=220,
    )


def _close_up_strip_box(
    sheet: Image.Image, grid_height: int
) -> tuple[int, int, int, int]:
    """Return the bounding box of the lit content in the strip below the grid.

    The strip sits under the grid, separated by `HIGHLIGHT_SEAM` blank rows, and is
    filled black where the crop does not reach. A luma threshold separates crop from
    background; it sits above the JPEG ringing at the crop's edge and far below the
    crop's own luma.

    An empty result carries the sheet's size, because the usual cause is that the
    sheet is not taller than the grid -- no strip was appended at all.
    """
    strip = sheet.crop((0, grid_height + HIGHLIGHT_SEAM, sheet.width, sheet.height))
    mask = strip.convert("L").point(lambda value: 255 if value > 40 else 0)
    box = mask.getbbox()
    assert box is not None, (
        f"no content in the strip below y={grid_height}; sheet is {sheet.size}"
    )
    return box


def _expected_strip_size(
    crop_size: tuple[int, int], sheet_width: int, grid_height: int
) -> tuple[int, int]:
    """The strip size the builder should produce for a crop of this shape.

    The crop is fitted to (the sheet's width, the grid's height), so a wide crop is
    capped by the grid height and a tall one by the sheet width. The cap is the
    grid's height rather than a single row's: one row was tried and made the
    close-up SMALLER than the side column it replaced -- 137x144 drawn against
    275x288 -- which defeats the point of moving it below the grid at all.
    """
    crop_width, crop_height = crop_size
    scale = min(sheet_width / crop_width, grid_height / crop_height)
    return max(1, round(crop_width * scale)), max(1, round(crop_height * scale))


def _close_up_span(sheet: Image.Image, rows: int) -> tuple[int, int]:
    """The close-up's drawn width and height, measured from the image itself.

    Taking the expectation from the composed sheet rather than computing it keeps
    the provider-path tests honest about their own subject: they exist to prove the
    close-up does not lose pixels on the way to the provider, and predicting the
    builder's rounding in a second place would only add a way for the two to
    disagree for reasons that have nothing to do with the provider.
    """
    grid_height = round((sheet.width / 3) / (640 / 360)) * rows
    box = _close_up_strip_box(sheet.convert("RGB"), grid_height)
    return box[2] - box[0], box[3] - box[1]


def _assert_has_close_up_strip(
    sheet: Image.Image, rows: int, total_width: int = _TARGET_WIDTH
) -> int:
    """Assert the sheet carries a close-up strip below the grid; return its height.

    What these end-to-end tests care about is invariant across crop shapes:

    * the sheet is taller than the grid alone, so a strip was appended;
    * the strip is not taller than one grid row, so the close-up never outgrows the
      frames it accompanies;
    * the crop is centred in the strip rather than jammed against an edge.

    `rows` is the frame count divided by three, which the caller knows; the grid's
    height in the output is the sheet's width divided by the cells' 16:9 aspect and
    multiplied by those rows. Deriving it here rather than taking a constant keeps
    the helper correct for both the six- and nine-frame fixtures.

    The strip's own width cannot be predicted from the frame: the crop's shape
    decides it, and the crop is derived from the detection box.
    """
    cell_width = total_width / 3
    row_height = round(cell_width / (640 / 360))
    grid_height = row_height * rows
    assert sheet.height > grid_height, (
        f"no close-up strip below the grid: sheet {sheet.height} tall, "
        f"grid {grid_height} ({rows} rows of {row_height})"
    )
    strip_height = sheet.height - grid_height - HIGHLIGHT_SEAM
    # The cap is the grid's own height, not a single row's. One row was tried and
    # made the close-up smaller than the side column it replaced, so anything up to
    # the grid's height is legitimate; past that the close-up would outgrow the
    # frames it accompanies.
    assert strip_height <= grid_height + 1, (
        f"the strip is {strip_height} tall, taller than the grid ({grid_height})"
    )
    box = _close_up_strip_box(sheet.convert("RGB"), grid_height)
    left_gap = box[0]
    right_gap = sheet.width - box[2]
    assert abs(left_gap - right_gap) <= 2, (
        f"the crop is not centred: {left_gap} left against {right_gap} right"
    )
    return strip_height


def test_a_sheet_without_a_highlight_is_unchanged(tmp_path) -> None:
    """不传特写时必须与现在逐字节一致 —— 零回归的保证。

    默认参数下既有部署的产物不能有任何变化。
    """
    frames = _nine_frames(tmp_path)
    default_output = tmp_path / "default.jpg"
    explicit_output = tmp_path / "explicit.jpg"
    build_contact_sheet(frames, default_output)
    build_contact_sheet(frames, explicit_output, highlight=None)
    assert default_output.read_bytes() == explicit_output.read_bytes()
    with Image.open(default_output) as sheet:
        assert sheet.size == (_GRID_WIDTH, _GRID_HEIGHT)


def test_a_sheet_with_a_highlight_is_wider_and_keeps_the_cells_intact(
    tmp_path,
) -> None:
    """有特写时总宽增加一栏，但九宫格那部分像素不变。

    九宫格若被特写挤小，时间轴的可读性会下降，而那正是九宫格的全部价值。
    """
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "person.jpg"
    highlight.write_bytes(_solid_jpeg((800, 400)))
    plain = tmp_path / "plain.jpg"
    combo = tmp_path / "combo.jpg"
    build_contact_sheet(frames, plain, target_width=_TARGET_WIDTH)
    build_contact_sheet(frames, combo, highlight=highlight, target_width=_TARGET_WIDTH)
    with Image.open(plain) as plain_sheet, Image.open(combo) as combo_sheet:
        _strip_w, strip_h = _expected_strip_size(
            (800, 400), _TARGET_WIDTH, _SCALED_HEIGHT
        )
        assert combo_sheet.size == (
            _TARGET_WIDTH,
            _SCALED_HEIGHT + HIGHLIGHT_SEAM + strip_h,
        )
        # The grid is the top block, unchanged pixel for pixel: appending a strip
        # must not cost the frames anything, which is the whole reason it goes
        # below rather than beside them.
        intact = (0, 0, _TARGET_WIDTH, _SCALED_HEIGHT - _SEAM_MARGIN)
        assert (
            ImageChops.difference(
                combo_sheet.convert("RGB").crop(intact),
                plain_sheet.convert("RGB").crop(intact),
            ).getbbox()
            is None
        ), "the grid must keep its pixels when a strip is appended"


def test_the_highlight_keeps_its_aspect_ratio(tmp_path) -> None:
    """特写必须等比缩放，不能拉伸变形 —— 变形会让模型误判人体比例。"""
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "tall.jpg"
    highlight.write_bytes(_solid_jpeg((100, 300)))
    output = tmp_path / "combo.jpg"
    build_contact_sheet(frames, output, highlight=highlight, target_width=_TARGET_WIDTH)
    with Image.open(output) as sheet:
        box = _close_up_strip_box(sheet.convert("RGB"), _SCALED_HEIGHT)
    assert (box[2] - box[0]) / (box[3] - box[1]) == pytest.approx(100 / 300, abs=0.01)


def test_the_highlight_keeps_its_pixels_when_the_provider_scales_the_sheet(
    tmp_path,
) -> None:
    """特写不能被 target_width 一起缩小 —— 这是本功能的成败点。

    `resize_for_provider` 会缩小任何宽于 target_width 的图。拼图先缩到 target_width
    再贴特写，所以特写保留自己的像素；若顺序反了，收益从 7.6 倍掉到 3 倍。

    放下方之后这条同样成立，而且更容易成立：拼图本身就是 target_width 宽，贴上的
    条也是同一宽度，整图不再「比 target 宽」。
    """
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "person.jpg"
    # Wide, so it is the strip's height that bounds it -- the case where a naive
    # implementation would lose pixels to the provider's own downscale.
    highlight.write_bytes(_solid_jpeg((1600, 400)))
    output = tmp_path / "combo.jpg"
    build_contact_sheet(frames, output, highlight=highlight, target_width=_TARGET_WIDTH)
    expected_w, expected_h = _expected_strip_size(
        (1600, 400), _TARGET_WIDTH, _SCALED_HEIGHT
    )
    with Image.open(output) as sheet:
        assert sheet.size == (
            _TARGET_WIDTH,
            _SCALED_HEIGHT + HIGHLIGHT_SEAM + expected_h,
        )
        box = _close_up_strip_box(sheet.convert("RGB"), _SCALED_HEIGHT)
    assert box[2] - box[0] == pytest.approx(expected_w, abs=1)
    assert box[3] - box[1] == pytest.approx(expected_h, abs=1)


def test_a_tall_crop_is_bounded_by_the_sheet_width(tmp_path) -> None:
    """A person-shaped crop is limited by the sheet's width, not by the row height.

    A real crop is taller than it is wide -- the reference person box is 147x185 in
    the 640x360 frame, and the measured one is 249x261. Fitting that into a strip
    one grid row tall would shrink it to a fraction of the width available, so the
    box it is fitted to is (sheet width, one row height) and the width binds first.
    """
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "person.jpg"
    highlight.write_bytes(_solid_jpeg((147, 185)))
    output = tmp_path / "combo.jpg"
    build_contact_sheet(frames, output, highlight=highlight, target_width=_TARGET_WIDTH)
    expected_w, expected_h = _expected_strip_size(
        (147, 185), _TARGET_WIDTH, _SCALED_HEIGHT
    )
    with Image.open(output) as sheet:
        assert sheet.size == (
            _TARGET_WIDTH,
            _SCALED_HEIGHT + HIGHLIGHT_SEAM + expected_h,
        )
        box = _close_up_strip_box(sheet.convert("RGB"), _SCALED_HEIGHT)
    assert box[2] - box[0] == pytest.approx(expected_w, abs=1)
    assert box[3] - box[1] == pytest.approx(expected_h, abs=1)
    # Centred, so the bars are split evenly rather than the person sitting against
    # one edge of a strip it does not fill.
    left = box[0]
    right = _TARGET_WIDTH - box[2]
    assert abs(left - right) <= 2, f"not centred: {left} left against {right} right"


def test_a_wide_crop_is_capped_by_the_grid_height(tmp_path) -> None:
    """A very wide crop is bounded by the grid's height, not by its own width.

    Letting the strip grow past the grid would make the close-up outgrow the frames
    it accompanies, and the close-up is an aid to the grid rather than the subject.
    Capping it at the grid's height also keeps the two halves in proportion: a strip
    as tall as the frames is already twice the area the old side column showed.
    """
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "wide.jpg"
    highlight.write_bytes(_solid_jpeg((4000, 200)))
    output = tmp_path / "combo.jpg"
    build_contact_sheet(frames, output, highlight=highlight, target_width=_TARGET_WIDTH)
    expected_w, expected_h = _expected_strip_size(
        (4000, 200), _TARGET_WIDTH, _SCALED_HEIGHT
    )
    with Image.open(output) as sheet:
        assert sheet.width == _TARGET_WIDTH, "the strip must not widen the sheet"
        strip_height = sheet.height - _SCALED_HEIGHT - HIGHLIGHT_SEAM
    assert strip_height == expected_h
    assert strip_height <= _SCALED_HEIGHT + 1, "the strip outgrew the grid"
    assert expected_w == _TARGET_WIDTH, "a very wide crop should fill the width"


def test_a_tall_crop_is_not_shrunk_smaller_than_the_old_side_column(
    tmp_path,
) -> None:
    """A tall crop must still be BIGGER than what the side column used to show.

    This pins the regression that one-row capping caused. Measured on the real
    shapes, a 249x261 crop in the strip capped at one grid row draws 137x144, while
    the side column it replaced drew 275x288 -- so the placement the owner asked for
    would have made the close-up smaller than before. Capped at the grid's height it
    draws 275x288, and at the display width that is about 2.5x the old area.
    """
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "person.jpg"
    highlight.write_bytes(_solid_jpeg((249, 261)))
    output = tmp_path / "combo.jpg"
    build_contact_sheet(frames, output, highlight=highlight, target_width=_TARGET_WIDTH)
    expected_w, expected_h = _expected_strip_size(
        (249, 261), _TARGET_WIDTH, _SCALED_HEIGHT
    )
    with Image.open(output) as sheet:
        box = _close_up_strip_box(sheet.convert("RGB"), _SCALED_HEIGHT)
    drawn_w, drawn_h = box[2] - box[0], box[3] - box[1]
    assert (drawn_w, drawn_h) == pytest.approx((expected_w, expected_h), abs=1)
    # The old column drew the crop at 275x288 (fitted into 448 x the grid height).
    assert drawn_h >= _SCALED_HEIGHT - 1, (
        f"the strip drew the crop {drawn_h} tall against the grid's {_SCALED_HEIGHT}; "
        "capping at one row would shrink it below the column it replaced"
    )


def test_a_broken_highlight_fails_like_a_broken_frame(tmp_path) -> None:
    """An unreadable crop raises MediaError rather than leaking a Pillow error.

    Every other unreadable image in this module is reported the same way, and a
    raw `UnidentifiedImageError` escaping through the executor would reach the
    caller as an unhandled crash instead of a recoverable media failure.
    """
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "person.jpg"
    highlight.write_bytes(b"not an image")
    with pytest.raises(MediaError, match="frame_decode_failed"):
        build_contact_sheet(
            frames, tmp_path / "combo.jpg", highlight=highlight, target_width=767
        )


def test_a_highlight_without_a_target_width_is_rejected(tmp_path) -> None:
    """A column composed over an unscaled grid would be shrunk as a whole.

    `resize_for_provider` shrinks any sheet wider than the target, so a sheet
    built at the grid's full 1920 plus 448 would lose the very pixels the column
    exists to protect -- measured: 448 becomes 145, and the gain drops from 7.6x
    to 3x. Failing loudly here keeps that loss from being reintroduced silently.
    """
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "person.jpg"
    highlight.write_bytes(_solid_jpeg((800, 400)))
    with pytest.raises(MediaError, match="highlight_requires_target_width"):
        build_contact_sheet(frames, tmp_path / "combo.jpg", highlight=highlight)


def test_a_composed_sheet_reaches_the_provider_unshrunk(tmp_path) -> None:
    """The composed sheet must reach the provider with the close-up's pixels intact.

    This is the acceptance point of the whole feature. `resize_for_provider` shrinks
    anything wider than the target width, and if that happened to the close-up the
    measured gain over the grid would collapse.

    With the strip below the grid the widths agree by construction -- the sheet is
    exactly `target_width` wide -- so the only way to lose pixels is to get the
    budget wrong and shrink on height. The assertion is on the close-up's own size
    rather than on the sheet's, because that is what the feature is for.
    """
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "person.jpg"
    highlight.write_bytes(_solid_jpeg((1600, 400)))
    sheet = tmp_path / "combo.jpg"
    build_contact_sheet(frames, sheet, highlight=highlight, target_width=_TARGET_WIDTH)
    expected_w, expected_h = _expected_strip_size(
        (1600, 400), _TARGET_WIDTH, _SCALED_HEIGHT
    )
    data = resize_for_provider(sheet, _TARGET_WIDTH, already_sized=True)
    with Image.open(BytesIO(data)) as sent:
        box = _close_up_strip_box(sent.convert("RGB"), _SCALED_HEIGHT)
        assert box[2] - box[0] == pytest.approx(expected_w, abs=1), (
            "the close-up was scaled with the sheet and its pixels were lost"
        )
        assert box[3] - box[1] == pytest.approx(expected_h, abs=1)


def test_the_budget_keeps_a_composed_sheet_whole(tmp_path) -> None:
    """With the close-up on, the budget admits a composed sheet unscaled.

    Same acceptance point as the test above, but driven the way the provider path
    actually drives it -- through the budget rather than the boolean -- so the
    wiring cannot silently regress while the boolean stays correct.
    """
    config = vision_config_from(
        {},
        {"llm_base_url": "https://example.test/v1", "llm_api_key": "k",
         "llm_model": "m", CONF_PERSON_HIGHLIGHT: True},
    )
    frames = _nine_frames(tmp_path)
    highlight = tmp_path / "person.jpg"
    highlight.write_bytes(_solid_jpeg((1600, 400)))
    sheet = tmp_path / "combo.jpg"
    build_contact_sheet(
        frames, sheet, highlight=highlight, target_width=config.target_width
    )
    with Image.open(sheet) as composed:
        expected_w, expected_h = _close_up_span(composed, rows=3)
    data = resize_for_provider(sheet, evidence_width_budget(config))
    with Image.open(BytesIO(data)) as sent:
        assert sent.size[0] == config.target_width, "the strip must not widen the sheet"
        # Measured against the composed sheet rather than a predicted number: the
        # close-up's scale is the builder's business, and what this test exists to
        # prove is that the provider path does not shrink it further.
        box = _close_up_strip_box(sent.convert("RGB"), _SCALED_HEIGHT)
        assert box[2] - box[0] == pytest.approx(expected_w, abs=1), (
            "the close-up lost pixels on the way to the provider"
        )
        assert box[3] - box[1] == pytest.approx(expected_h, abs=1)


def test_the_budget_still_shrinks_an_artifact_built_before_the_option_was_on(
    tmp_path,
) -> None:
    """A grid-only sheet reused from before the option was enabled must still shrink.

    This is why the provider path uses a budget rather than the `already_sized`
    boolean. The option describes what *new* sheets look like, but evidence is
    reused once built (`_async_build_locked` returns early when the stage is
    EVIDENCE_READY), so an artifact can predate the switch. Deriving a boolean
    from the current setting would pass that 1920-wide grid-only sheet through
    unscaled -- 6.3x the pixels, billed, for a close-up the image does not
    contain. The budget is safe for both shapes.
    """
    config = vision_config_from(
        {},
        {"llm_base_url": "https://example.test/v1", "llm_api_key": "k",
         "llm_model": "m", CONF_PERSON_HIGHLIGHT: True},
    )
    # A stale artifact: grid only, never scaled by build_contact_sheet.
    frames = _nine_frames(tmp_path)
    stale = tmp_path / "stale.jpg"
    build_contact_sheet(frames, stale)
    with Image.open(stale) as image:
        assert image.size[0] > evidence_width_budget(config), (
            "fixture must exceed the budget for this test to mean anything"
        )
    data = resize_for_provider(stale, evidence_width_budget(config))
    with Image.open(BytesIO(data)) as sent:
        assert sent.size[0] == evidence_width_budget(config), (
            "a stale grid-only sheet was sent unscaled"
        )


def test_resizing_still_shrinks_a_grid_only_sheet(tmp_path) -> None:
    """The opt-out must not become the default for the plain grid path.

    `resize_for_provider` exists because the provider bills by pixel area and a
    full-size sheet costs several times more to send. A deployment with no
    close-up must keep being scaled down, or every analysis silently gets more
    expensive.
    """
    frames = _nine_frames(tmp_path)
    sheet = tmp_path / "plain.jpg"
    build_contact_sheet(frames, sheet, target_width=_TARGET_WIDTH)
    data = resize_for_provider(sheet, 640)
    with Image.open(BytesIO(data)) as sent:
        assert sent.size == (640, 360)


def test_recording_coverage_rejects_any_gap() -> None:
    assert recordings_cover((1, 2, 3), [{"start_time": 0, "end_time": 4}])
    assert not recordings_cover(
        (1, 2, 3),
        [{"start_time": 0, "end_time": 1.5}, {"start_time": 2.5, "end_time": 4}],
    )


def test_zero_change_review_candidates_are_rejected(tmp_path) -> None:
    frames = []
    for index in range(5):
        path = tmp_path / f"same_{index}.jpg"
        path.write_bytes(_jpeg(0))
        frames.append((float(index), path))
    with pytest.raises(MediaError, match="insufficient_visible_change"):
        select_review_change_frames(frames)


def test_too_few_candidates_is_distinct_from_no_visible_change(tmp_path) -> None:
    """Not enough frames and no movement in the frames are different failures.

    They previously shared one code, so an operator could not tell a missing
    download from a genuinely static scene. The first is a frame-availability
    problem, the second means the camera saw nothing worth analysing.
    """
    # Only two frames: the selector needs count+1 to score anything at all.
    too_few = []
    for index in range(2):
        path = tmp_path / f"few_{index}.jpg"
        path.write_bytes(_jpeg(index * 40))
        too_few.append((float(index), path))
    with pytest.raises(MediaError, match="insufficient_review_candidates"):
        select_review_change_frames(too_few)

    # Enough frames, but every one is identical.
    static = []
    for index in range(5):
        path = tmp_path / f"static_{index}.jpg"
        path.write_bytes(_jpeg(0))
        static.append((float(index), path))
    with pytest.raises(MediaError, match="insufficient_visible_change"):
        select_review_change_frames(static)


def test_final_frames_reject_duplicate_fixed_and_change_images(tmp_path) -> None:
    paths = []
    for index, value in enumerate((0, 100, 0)):
        path = tmp_path / f"duplicate_{index}.jpg"
        path.write_bytes(_jpeg(value))
        paths.append(path)
    with pytest.raises(MediaError, match="duplicate_evidence_frame"):
        validate_unique_frames(paths)


def test_a_localized_change_is_not_mistaken_for_a_duplicate(tmp_path) -> None:
    """局部的真实变化不能被判成重复——旧指标正是在这里判反了。

    缺陷实测（640x360，签名 64x36 = 2304 px）：

    | 帧对 | 旧指标（全帧均值） | 旧判定 | 应判定 |
    |---|---|---|---|
    | 同一张图重新编码（quality 30 vs 95） | 0.193 | 不同 | 重复 |
    | 20x20 物体出现（0.17%，真变化） | 0.049 | **重复** | 不同 |
    | 48x48 物体出现（1.0%） | 0.227 | 不同 | 不同 |

    均值被**面积**稀释：一个真实物体只占 0.17% 的画面，得分 0.049，反而**低于**
    "同一张图重新编码"的 0.193。后果不是少一次分析——被判重复会让整条活动失败，
    而 `duplicate_evidence_frame` 既不可重试也不可补跑（生产上已发生 3 次）。

    修复是**再加**一个判据而不是替换：均值负责"整片轻微变化"，变化像素计数负责
    "小面积剧烈变化"，两者**都**说没事才算重复。所以新判据只会比旧的更宽松，
    不可能把原本接受的帧对变成拒绝——这一点由下面的断言钉住。
    """
    from custom_components.frigate_vision.media import (
        _grayscale_signature,
        change_fraction,
        frames_are_duplicates,
    )

    def _frame(path, pixels) -> object:
        image = Image.new("L", (640, 360))
        image.putdata(pixels)
        image.save(path, "JPEG", quality=85)
        return _grayscale_signature(path)

    count = 640 * 360
    base = [110] * count
    reference = _frame(tmp_path / "a.jpg", base)

    # Same scene, every pixel one level brighter: a different picture for the mean
    # measure, so the combined predicate must not become stricter than the old one.
    gain_shift = _frame(tmp_path / "b.jpg", [111] * count)
    assert not frames_are_duplicates(reference, gain_shift), (
        "整帧亮度变化被判成同一画面——新判据不该比旧的更严格"
    )

    # A real object appearing: 30x30 px (0.39% of the frame) at +140 levels.
    changed = list(base)
    for y in range(180, 210):
        for x in range(320, 350):
            changed[y * 640 + x] = 250
    object_appears = _frame(tmp_path / "c.jpg", changed)
    assert change_fraction(reference, object_appears) > 0, (
        "真出现的物体必须产生变化像素"
    )
    assert not frames_are_duplicates(reference, object_appears), (
        "真的有东西出现了却被判成重复——这会让整条活动失败且不可补跑"
    )


def test_a_candidate_the_repair_accepts_cannot_be_rejected_by_the_guard(
    tmp_path,
) -> None:
    """修复说"我接受这个候选"，守卫就不能反悔——两侧必须用同一个判据、同一组比较。

    缺陷实测：`_collides_with_neighbours` 的 docstring 承诺 *"a candidate this accepts
    cannot later be rejected by it"*，但它只比较**相邻**两帧，而
    `validate_unique_frames` 比较**所有帧对**。于是一个与**非相邻**帧重复的候选会被
    修复放行、再被守卫拒绝——正是那句承诺说不会发生的事。

    `_async_replace_duplicates` 没有这个洞（它的 `acceptable` 比较所有帧），所以两个
    修复对"什么算重复"的理解本来就不一致。这条把两者钉在同一个 `frames_are_duplicates`
    上：非相邻重复也必须被拒。
    """
    from custom_components.frigate_vision.media import (
        _grayscale_signature,
        frames_are_duplicates,
    )

    def _plain(path, value, size=(64, 36)) -> Path:
        Image.new("L", size, value).save(path, "JPEG", quality=90)
        return path

    # Five frames, so a NON-adjacent duplicate has neighbours that are genuinely
    # different: the trial sits at index 2, and duplicates index 0.
    #
    #   0 = 20   <-- the picture the trial repeats (non-adjacent)
    #   1 = 200  <-- neighbour, different
    #   2 = placeholder; trial goes here
    #   3 = 200  <-- neighbour, different
    #   4 = 60
    first = _plain(tmp_path / "first.jpg", 20)
    left = _plain(tmp_path / "left.jpg", 200)
    placeholder = _plain(tmp_path / "placeholder.jpg", 90)
    right = _plain(tmp_path / "right.jpg", 200)
    tail = _plain(tmp_path / "tail.jpg", 60)
    trial = _plain(tmp_path / "trial.jpg", 20)

    result = [
        (1.0, first),
        (2.0, left),
        (3.0, placeholder),
        (4.0, right),
        (5.0, tail),
    ]

    assert frames_are_duplicates(
        _grayscale_signature(first), _grayscale_signature(trial)
    ), "precondition: the trial repeats frame 0"

    # Neither neighbour repeats the trial, so only a whole-sheet comparison finds it.
    for neighbour in (1, 3):
        assert not frames_are_duplicates(
            _grayscale_signature(result[neighbour][1]),
            _grayscale_signature(trial),
        ), f"precondition: neighbour {neighbour} must differ from the trial"

    assert MediaManager._collides_with_neighbours(2, trial, result), (
        "候选与**非相邻**帧重复却被放行——守卫随后会拒绝它，"
        "正是 docstring 承诺不会发生的事"
    )


def test_a_byte_identical_pair_is_still_a_duplicate(tmp_path) -> None:
    """修好方向感不能放过"同一张图"——那正是这个守卫存在的理由。

    这是必须保留的一半：Frigate 会把两个相邻请求吸附到同一张存储帧上，返回逐字节
    相同的图。把它当成两个时刻就是在骗模型，也正是 `validate_unique_frames` 的
    原意（见它上面的注释）。
    """
    from custom_components.frigate_vision.media import (
        _grayscale_signature,
        frames_are_duplicates,
    )

    path = tmp_path / "same.jpg"
    Image.new("L", (640, 360), 128).save(path, "JPEG", quality=85)
    # Re-open so the two sides are independent decodes of the same bytes, as a
    # duplicate pair is in production.
    assert frames_are_duplicates(
        _grayscale_signature(path), _grayscale_signature(path)
    )


async def test_media_manager_builds_once_and_atomically_registers(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="door_entry_1_100000",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=130,
        camera="front",
        detection_ids=("event_1",),
        detection_zone_updates=(
            ("event_1", 102, ("near",)),
            ("event_1", 110, ("mid",)),
            ("event_1", 120, ("far",)),
        ),
        association_deadline=140,
        finalization_deadline=220,
    )
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 101,
                "end_time": 121,
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _jpeg(int(timestamp) % 255)

    client = Client()
    manager = MediaManager(
        hass,
        store,
        client,
        tmp_path,
        ZoneRoles(
            near=frozenset({"near"}),
            transition=frozenset({"mid"}),
            far=frozenset({"far"}),
        ),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert completed.evidence_path is not None
    # The review layout fetches first + last + 9 change candidates + postroll.
    assert client.snapshot_calls == 12
    again = await manager.async_build(record.activity_id)
    assert again == completed
    assert client.snapshot_calls == 12


async def test_media_manager_serializes_concurrent_builds(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="door_entry_1_100000",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=130,
        camera="front",
        detection_ids=("event_1",),
        detection_zone_updates=(
            ("event_1", 102, ("near",)),
            ("event_1", 110, ("mid",)),
            ("event_1", 120, ("far",)),
        ),
        finalization_deadline=220,
    )
    await store.async_create(record)

    class Client:
        calls = 0

        async def async_get_event(self, event_id, camera):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 101,
                "end_time": 121,
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": 0, "end_time": 300}]

        async def async_get_snapshot(self, camera, timestamp, height):
            self.calls += 1
            await asyncio.sleep(0)
            return _jpeg(int(timestamp))

    client = Client()
    manager = MediaManager(
        hass,
        store,
        client,
        tmp_path,
        ZoneRoles(frozenset({"near"}), frozenset({"mid"}), frozenset({"far"})),
    )
    first, second = await asyncio.gather(
        manager.async_build(record.activity_id), manager.async_build(record.activity_id)
    )
    assert first == second
    # The review layout fetches first + last + 9 change candidates + postroll;
    # a concurrent second build must not re-fetch any of them.
    assert client.calls == 12


async def test_evidence_ready_requires_valid_registered_artifact(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.EVIDENCE_READY,
        created_at=1,
        updated_at=2,
        camera="front",
        evidence_mode="review_six",
        evidence_revision=1,
        evidence_path=str(tmp_path / "entry_1" / "activity_1.jpg"),
        evidence_media_url="media-source://frigate_vision/entry_1/activity_1",
        sample_times=(1, 1.5, 2),
        claimed_side_effects=("media:activity_1:1",),
    )
    await store.async_create(record)
    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    with pytest.raises(MediaError, match="artifact_missing"):
        await manager.async_build(record.activity_id)


async def test_evidence_ready_rejects_valid_file_outside_canonical_root(
    hass: HomeAssistant, tmp_path
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_jpeg(1))
    outside.with_suffix(".json").write_text(
        '{"activity_id":"activity_1","plan_version":1,"mode":"review_six","sample_times":[1,2,3]}'
    )
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            created_at=1,
            updated_at=2,
            camera="front",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(outside),
            evidence_media_url="media-source://frigate_vision/entry_1/activity_1",
            sample_times=(1, 2, 3),
            claimed_side_effects=("media:activity_1:1",),
        )
    )
    manager = MediaManager(
        hass, store, object(), root, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    with pytest.raises(MediaError, match="invalid_output_path"):
        await manager.async_build("activity_1")


async def test_media_cleanup_deletes_only_expired_registered_files(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    path = tmp_path / "entry_1" / "old.jpg"
    path.parent.mkdir()
    path.write_bytes(_jpeg(1))
    path.with_suffix(".json").write_text("{}")
    await store.async_create(
        ActivityRecord(
            activity_id="old",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.COMPLETED,
            created_at=1,
            updated_at=1,
            camera="front",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(path),
            evidence_media_url="media-source://frigate_vision/entry_1/old",
            sample_times=(1, 2, 3),
            claimed_side_effects=("media:old:1",),
        )
    )
    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    removed = await manager.async_cleanup(retention_days=1, now=200000)
    assert removed == ("old",)
    assert not path.exists()
    assert not path.with_suffix(".json").exists()
    expired = store.get("old")
    assert expired is not None and expired.evidence_expired_at == 200000
    await manager.async_restore_registry()


async def test_cleanup_removes_registry_before_physical_delete_failure(
    hass: HomeAssistant, tmp_path, monkeypatch
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    path = tmp_path / "entry_1" / "old.jpg"
    path.parent.mkdir()
    path.write_bytes(_jpeg(1))
    await store.async_create(
        ActivityRecord(
            activity_id="old",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.COMPLETED,
            created_at=1,
            updated_at=1,
            camera="front",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(path),
            evidence_media_url="media-source://frigate_vision/entry_1/old",
            sample_times=(1, 2, 3),
            claimed_side_effects=("media:old:1",),
        )
    )
    hass.data["frigate_vision_media_registry"] = {"entry_1/old": path}
    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    monkeypatch.setattr(
        "custom_components.frigate_vision.media._delete_artifact",
        lambda *args: (_ for _ in ()).throw(OSError("delete failed")),
    )
    with pytest.raises(OSError, match="delete failed"):
        await manager.async_cleanup(retention_days=1, now=200000)
    assert "entry_1/old" not in hass.data["frigate_vision_media_registry"]
    assert store.get("old").evidence_expired_at == 200000  # type: ignore[union-attr]


def _standalone_record() -> ActivityRecord:
    return ActivityRecord(
        activity_id="review_entry_1_front_review_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=120,
        camera="front",
        review_ids=("review_1",),
        detection_ids=("event_1",),
        finalization_deadline=120,
    )


def _motion_points() -> list[list[object]]:
    return [[[float(index), 0.0], 103.0 + index] for index in range(7)]


async def test_review_builds_a_nine_cell_sheet_when_a_hole_can_be_probed(
    hass: HomeAssistant, tmp_path
) -> None:
    """Nine cells = first | motion x3 | probed x3 | last | postroll.

    The three extra cells are the whole point: they are chosen by visible change
    rather than by where the person moved, which cuts the worst unobserved gap
    from 17.1s to 11.5s (better on 29 of 48 activities, worse on none). A nine-cell
    sheet is therefore not "six motion picks plus three more of the same" -- that
    arrangement measured 16.4s and buys almost nothing.

    Built from the measured 19:05 movement shape, whose clustered points are what
    leaves a hole wide enough to probe at all.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_nine",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=102,
        updated_at=152,
        camera="front",
        review_ids=("review_nine",),
        detection_ids=("event_1",),
        finalization_deadline=152,
    )
    await store.async_create(record)

    class Client:
        def __init__(self) -> None:
            self.snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            relative = (
                (0.20, 0.0127),
                (1.79, 0.0853),
                (5.60, 0.0611),
                (21.36, 0.0581),
                (23.34, 0.0851),
                (23.70, 0.0902),
                (34.08, 0.0797),
                (34.48, 0.0906),
                (35.08, 0.1516),
                (41.27, 0.1361),
                (41.49, 0.0093),
                (42.47, 0.1236),
                (42.88, 0.0672),
                (43.27, 0.0612),
                (45.05, 0.0706),
            )
            points: list[list[object]] = [[[0.0, 0.0], 103.0]]
            x = 0.0
            for offset, distance in relative:
                x += distance
                points.append([[x, 0.0], 103.0 + offset])
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 150,
                "data": {"path_data": points},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 9, (
        f"a nine-cell sheet was expected: {completed.sample_times}"
    )
    assert list(completed.sample_times) == sorted(completed.sample_times)
    assert len(set(completed.sample_times)) == 9, "no cell may repeat another"
    with Image.open(completed.evidence_path) as sheet:
        assert sheet.size == (640 * 3, 360 * 3), sheet.size


async def test_review_keeps_six_cells_when_no_hole_is_wide_enough(
    hass: HomeAssistant, tmp_path
) -> None:
    """A tightly covered activity stays at six -- nine is not unconditional.

    Probing only applies where a hole exists. Every probe costs a snapshot fetch,
    and a hole shorter than the shortest action worth seeing cannot hide one, so
    the sheet must stay at its six-cell size rather than pad itself with frames
    chosen for no reason.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6, completed.sample_times


async def test_probe_cells_keep_their_role_aligned_with_their_time(
    hass: HomeAssistant, tmp_path
) -> None:
    """A role must describe the frame it is paired with, after sorting.

    `_nudge_uncovered_frames` reads `roles[index]` to decide which frames the
    sheet cannot be built without -- `first`, `last` and `postroll` are required,
    a probe is not. Probes land *inside* the motion span, so sorting the times
    without carrying the roles along would pair a probe with a required cell's
    role. The sheet would then survive a recording hole it cannot actually be
    built from, or reject one it could.

    Verified through the nudge report, which records the role of every frame it
    had to move -- the one place the pairing becomes observable.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_roles",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=102,
        updated_at=152,
        camera="front",
        review_ids=("review_roles",),
        detection_ids=("event_1",),
        finalization_deadline=152,
    )
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            relative = (
                (0.20, 0.0127),
                (1.79, 0.0853),
                (5.60, 0.0611),
                (21.36, 0.0581),
                (23.34, 0.0851),
                (23.70, 0.0902),
                (34.08, 0.0797),
                (34.48, 0.0906),
                (35.08, 0.1516),
                (41.27, 0.1361),
                (41.49, 0.0093),
                (42.47, 0.1236),
                (42.88, 0.0672),
                (43.27, 0.0612),
                (45.05, 0.0706),
            )
            points: list[list[object]] = [[[0.0, 0.0], 103.0]]
            x = 0.0
            for offset, distance in relative:
                x += distance
                points.append([[x, 0.0], 103.0 + offset])
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 150,
                "data": {"path_data": points},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            # A hole covering the last stretch only, so the postroll frame must be
            # nudged and the report names its role.
            return [
                {"start_time": after - 1, "end_time": 149.0},
                {"start_time": 152.0, "end_time": before + 1},
            ]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 9
    # Every nudge the builder reported must name a role the sheet actually uses,
    # which is only true if roles stayed paired with their own times through the
    # sort. An unknown role would mean a frame was labelled with another's role.
    metadata = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    payload = json.loads(metadata)
    for nudge in payload["recording_nudges"]:
        assert nudge["role"] in {
            "first",
            "last",
            "postroll",
            "motion",
            "probe",
        }, f"a frame carried another cell's role: {nudge}"
    # And the sheet's own ordering must survive: nine strictly increasing cells.
    assert list(completed.sample_times) == sorted(completed.sample_times)
    assert len(set(completed.sample_times)) == 9


async def test_review_probes_the_widest_hole_and_reselects(
    hass: HomeAssistant, tmp_path
) -> None:
    """The fill step must actually reach the sheet, not just exist.

    The unit tests cover `probe_gap_times` and the `extra` candidates in
    isolation. Neither proves the pipeline calls them: the earlier scene-
    description work shipped a capability that was defined and then never wired,
    and it failed silently for days.

    The fixture gives the camera a scene that changes *only* in the stretch where
    the person was not moving, which is the measured failure mode -- an entry
    door opening while the detector had nothing to track. The path points stay
    still, so only probing can find it.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    # A window wide enough for the real 46s span: the hole only exists because
    # the activity is long and the movements cluster in its first half.
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_hole",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=102,
        updated_at=152,
        camera="front",
        review_ids=("review_hole",),
        detection_ids=("event_1",),
        finalization_deadline=152,
    )
    await store.async_create(record)

    # Built from the measured movement list of activity 19:05 on this
    # deployment, at its real 46s span. Compressing it into the record's usual
    # 14s window destroys the very thing under test: the selector spreads picks
    # to minimise the largest gap, so a short window simply has no hole left to
    # probe. The hole only exists because the activity is long and the movements
    # cluster early.
    class Client:
        snapshot_calls = 0
        probed: list[float] = []

        async def async_get_event(self, event_id: str, camera: str):
            relative = (
                (0.20, 0.0127),
                (1.79, 0.0853),
                (5.60, 0.0611),
                (21.36, 0.0581),
                (23.34, 0.0851),
                (23.70, 0.0902),
                (34.08, 0.0797),
                (34.48, 0.0906),
                (35.08, 0.1516),
                (41.27, 0.1361),
                (41.49, 0.0093),
                (42.47, 0.1236),
                (42.88, 0.0672),
                (43.27, 0.0612),
                (45.05, 0.0706),
            )
            points: list[list[object]] = [[[0.0, 0.0], 103.0]]
            x = 0.0
            for offset, distance in relative:
                x += distance
                points.append([[x, 0.0], 103.0 + offset])
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 150,
                "data": {"path_data": points},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            # A flash standing for the entry door opening: the person was inside
            # and still, so no path point reports it. It sits inside the hole the
            # motion picks leave (roughly 108.6..124.4), which is the point.
            if 110.0 < timestamp < 122.0:
                return _jpeg(235)
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    # Nine cells: the probe cells are *added* rather than replacing a motion pick,
    # which is the change this test now covers. An earlier revision re-aimed the
    # picks and stayed at six; that spent the extra evidence on nothing, since
    # more motion picks measured 16.4s worst gap against 11.5s for probed cells.
    assert len(completed.sample_times) == 9
    # Probing costs extra fetches; without them the count would be exactly six.
    assert client.snapshot_calls > 6, (
        "no probe was fetched, so the fill step never ran"
    )
    # The property that matters, and the one the reported failure violated: some
    # cell now sits strictly *inside* the hole. Without probing the three middle
    # picks are 108.6, 124.4 and 137.1 -- the first two exactly at the hole's
    # edges -- so nothing observed the 15.8s in between. Asserting on the hole
    # rather than on a specific instant keeps this true wherever inside it the
    # probes land.
    assert any(112.0 < moment < 122.0 for moment in completed.sample_times), (
        f"the hole is still unobserved: {completed.sample_times}"
    )


async def test_review_keeps_its_picks_when_probing_finds_nothing(
    hass: HomeAssistant, tmp_path
) -> None:
    """A failed probe must not lose the sheet.

    Probing is an improvement attempt. When it yields nothing usable the original
    motion picks have to survive intact -- a sheet built from a hole full of
    static frames would be worse than one that never looked.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            relative = (
                (0.20, 0.0127),
                (1.79, 0.0853),
                (5.60, 0.0611),
                (21.36, 0.0581),
                (23.34, 0.0851),
                (34.08, 0.0797),
                (35.08, 0.1516),
                (41.27, 0.1361),
                (42.47, 0.1236),
                (45.05, 0.0706),
            )
            scale = 14.0 / 46.26
            points: list[list[object]] = [[[0.0, 0.0], 103.0]]
            x = 0.0
            for offset, distance in relative:
                x += distance
                points.append([[x, 0.0], 103.0 + offset * scale])
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": points},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # Fail only the probes. The real cells are at the motion picks and at
            # the window edges, so a band strictly inside the hole hits probes
            # without touching a required frame.
            if 105.0 < timestamp < 108.5:
                raise OSError("no recording at this instant")
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert completed.selection_source == "path_motion"


async def test_review_path_motion_downloads_exactly_six_frames(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _jpeg(int(timestamp) % 255)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert completed.selection_source == "path_motion"
    assert client.snapshot_calls == 6


async def test_review_missing_path_data_falls_back_to_image_change(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.selection_source == "image_change"
    assert client.snapshot_calls == 12


async def test_review_invalid_path_data_fails_without_download(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": "not-a-list"},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": 0, "end_time": 300}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _jpeg(1)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    with pytest.raises(MediaError, match="invalid_path_data"):
        await manager.async_build(record.activity_id)
    assert client.snapshot_calls == 0


async def test_legacy_image_change_artifact_is_not_kept_as_path_motion(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)
    final = tmp_path / "entry_1" / f"{record.activity_id}.jpg"
    final.parent.mkdir(parents=True)
    final.write_bytes(_jpeg(10))
    final.with_suffix(".json").write_text(
        '{"activity_id":"review_entry_1_front_review_1","plan_version":1,'
        '"mode":"review_six","sample_times":[103.2,104.0,106.0,108.0,116.8,119.8]}'
    )

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.selection_source == "path_motion"
    assert client.snapshot_calls == 6
    payload = final.with_suffix(".json").read_text()
    assert '"plan_version": 4' in payload
    assert '"selection_source": "path_motion"' in payload


def test_infrared_frames_are_separated_from_colour_frames(tmp_path) -> None:
    """The IR discriminator must not fire on ordinary colour frames.

    Production measurements: IR frames sit near R-G = -150, transition frames
    near -35, and colour frames near +2. The threshold must reject the IR frames
    while keeping every transition and colour frame.
    """
    ir = tmp_path / "ir.jpg"
    ir.write_bytes(_ir_jpeg())
    assert frame_is_infrared(ir)

    colour = tmp_path / "colour.jpg"
    colour.write_bytes(_jpeg(128))
    assert not frame_is_infrared(colour)

    warm = tmp_path / "warm.jpg"
    output = BytesIO()
    Image.new("RGB", (64, 36), (130, 120, 100)).save(output, "JPEG")
    warm.write_bytes(output.getvalue())
    assert not frame_is_infrared(warm)

    transition = tmp_path / "transition.jpg"
    output = BytesIO()
    Image.new("RGB", (64, 36), (98, 128, 113)).save(output, "JPEG")
    transition.write_bytes(output.getvalue())
    assert not frame_is_infrared(transition)


def test_blown_frames_are_separated_from_usable_frames(tmp_path) -> None:
    """A saturated frame must be detected while ordinary frames pass.

    Measured across the shadow-mode sheets: blown cells are 42%-98% near-white
    while every usable cell is under 1%, so the discriminator has a wide margin.
    Bright-but-detailed frames must NOT be rejected -- a white door or wall is
    normal in this corridor and carries real evidence.
    """
    blown = tmp_path / "blown.jpg"
    blown.write_bytes(_blown_jpeg())
    assert frame_is_overexposed(blown)

    for value in (0, 64, 128, 200):
        normal = tmp_path / f"normal_{value}.jpg"
        normal.write_bytes(_jpeg(value))
        assert not frame_is_overexposed(normal), f"grey {value} must be usable"

    # A bright frame with structure is still evidence. Measured on production
    # cells, usable frames reach 14.7% near-white pixels (a sunlit doorway) while
    # blown ones start at 42.7%; a quarter-lit frame must therefore survive, or
    # the frames showing the white door in this corridor would be discarded.
    structured = tmp_path / "structured.jpg"
    output = BytesIO()
    image = Image.new("RGB", (64, 36), (60, 58, 55))
    for x in range(48, 64):
        for y in range(0, 36):
            image.putpixel((x, y), (242, 241, 239))
    image.save(output, "JPEG")
    structured.write_bytes(output.getvalue())
    assert not frame_is_overexposed(structured)


async def test_review_replaces_blown_frames_with_neighbouring_frame(
    hass: HomeAssistant, tmp_path
) -> None:
    """A saturated first frame must be replaced, as IR frames already are.

    When the lift door opens into the lens the sensor saturates. Measured on the
    shadow sheets, the `home_departure` sheet had two of six cells at 95.7% and
    81.5% near-white, so a third of that classification rested on frames holding
    no evidence at all.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            # Only the first requested timestamp is blown out.
            if abs(timestamp - 103.2) < 0.01:
                return _blown_at(timestamp)
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert 103.2 not in completed.sample_times
    assert client.snapshot_calls > 6
    assert tuple(sorted(set(completed.sample_times))) == completed.sample_times


async def test_blown_replacement_is_recorded_in_the_evidence_metadata(
    hass: HomeAssistant, tmp_path
) -> None:
    """The sheet must say which frames could not be recovered.

    Without the record, a classification resting on a still-saturated frame looks
    identical to one resting on six usable frames.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return (
                _blown_jpeg()
                if abs(timestamp - 103.2) < 0.01
                else _changing_jpeg(timestamp)
            )

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    metadata = Path(completed.evidence_path).with_suffix(".json").read_text()
    assert "overexposure_replacements" in metadata
    assert '"reason": "exposure"' in metadata


async def test_review_keeps_blown_frame_when_no_clear_alternative(
    hass: HomeAssistant, tmp_path
) -> None:
    """An unrecoverable frame must not fail the activity.

    Blown frames cluster where the lift door is open, so a whole window can be
    saturated. Failing would drop a real visit; the sheet is still more useful
    with five good frames than with none, and the metadata records the shortfall.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # Everything before 108 is saturated, and the ladder reaches only 5s
            # from 103.2, so no candidate for that frame is clear. Distinct grey
            # values elsewhere keep the uniqueness guard satisfied, so this test
            # isolates the overexposure behaviour.
            if timestamp < 108:
                return _blown_at(timestamp)
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6


async def test_review_replaces_infrared_frames_with_neighbouring_frame(
    hass: HomeAssistant, tmp_path
) -> None:
    """An IR first frame must be replaced by a later colour frame.

    The camera lags ~2s behind person detection when leaving night mode, so the
    first sample is reliably IR. Replacing it keeps six real frames instead of
    spending one cell on a washed-out frame.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            # Only the very first requested timestamp is IR.
            if abs(timestamp - 103.2) < 0.01:
                return _ir_jpeg()
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    # Six frames still, plus at least one extra fetch for the replacement.
    assert len(completed.sample_times) == 6
    assert 103.2 not in completed.sample_times
    assert client.snapshot_calls > 6
    assert tuple(sorted(set(completed.sample_times))) == completed.sample_times


async def test_review_reaches_colour_beyond_the_old_three_second_ladder(
    hass: HomeAssistant, tmp_path
) -> None:
    """A frame must be able to wait out a long night-mode exit.

    Measured on the 2026-09-20 08:27 and 10:24 reviews: the camera needed 3.5s
    to return colour, while the ladder stopped at +3.0s. Both sheets shipped
    with two full green frames and an empty `infrared_replacements`. The ladder
    now reaches 5.0s so a slow exit is still recoverable.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    # The first motion frame is IR until +3.5s; frame 2 sits far enough away
    # that +3.5s is in bounds.
    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # 105.0 is the first motion frame; it stays IR until 108.5.
            if 105.0 <= timestamp < 108.5:
                return _ir_jpeg()
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    # The IR frame was moved out to a colour instant, not left green.
    assert 105.0 not in completed.sample_times
    meta = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    assert "infrared_replacements" in meta
    assert '"to": 108.5' in meta


async def test_review_keeps_infrared_frame_when_no_colour_alternative(
    hass: HomeAssistant, tmp_path
) -> None:
    """A review must still produce evidence when every frame is IR.

    Failing the whole activity would discard real (if ugly) frames and lose the
    activity entirely, so the frame is kept and the substitution recorded.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _ir_jpeg(int(timestamp) % 200 + 20)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6


async def test_evidence_metadata_records_infrared_replacements(
    hass: HomeAssistant, tmp_path
) -> None:
    """Substitutions must be observable in the artifact metadata."""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if abs(timestamp - 103.2) < 0.01:
                return _ir_jpeg()
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    metadata = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    assert '"plan_version": 4' in metadata
    assert "infrared_replacements" in metadata
    assert "103.2" in metadata
    assert completed.evidence_path is not None


async def test_review_retries_a_frame_that_duplicates_another(
    hass: HomeAssistant, tmp_path
) -> None:
    """A frame colliding with another must be re-fetched instead of failing.

    The three fixed frames (first/last/postroll) are computed from the person
    window and are never scored by the change selector, so on a static corridor
    they can land on a frame identical to another selection and abort the whole
    activity. Uniqueness is the only signal that reliably detects this, so the
    collision is resolved by fetching a neighbouring frame that differs from
    every other selected frame.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    # 103.2 is the plan's first frame; the motion planner will also select
    # 105.0, so the two render identically and the sheet cannot be unique.
    colliding = 103.2
    partner = 105.0

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            # The colliding frame stays flat grey for the first retry step, so
            # only a further offset produces a frame distinct from every other.
            if abs(timestamp - colliding) < 0.01 or abs(timestamp - partner) < 0.01:
                return _jpeg(7)
            if abs(timestamp - (colliding + 0.5)) < 0.01:
                return _jpeg(7)
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert colliding not in completed.sample_times
    assert client.snapshot_calls > 6


async def test_review_records_uniqueness_replacements(
    hass: HomeAssistant, tmp_path
) -> None:
    """Uniqueness substitutions must be observable in the artifact metadata."""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if abs(timestamp - 103.2) < 0.01 or abs(timestamp - 105.0) < 0.01:
                return _jpeg(7)
            if abs(timestamp - 103.7) < 0.01:
                return _jpeg(7)
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.evidence_path is not None
    metadata = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    assert '"plan_version": 4' in metadata
    assert "uniqueness_replacements" in metadata


async def test_review_still_fails_when_no_unique_neighbour_exists(
    hass: HomeAssistant, tmp_path
) -> None:
    """A uniform scene keeps the existing failure rather than fabricating frames.

    Every frame renders identically here, so no retry can produce a distinct
    frame. The guard is deliberately *not* relaxed: emitting six copies of one
    image would tell the model it saw six moments when it saw one. The activity
    fails with the original code so the loss stays visible.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 20, "end_time": before + 20}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return _jpeg(128)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    with pytest.raises(MediaError, match="duplicate_evidence_frame"):
        await manager.async_build(record.activity_id)


async def test_review_tolerates_a_gap_under_a_discarded_change_candidate(
    hass: HomeAssistant, tmp_path
) -> None:
    """A recording hole under a change *candidate* must not fail the activity.

    Recording segments can leave sub-second holes. The nine proportional change
    candidates are only a search space -- three are kept -- so requiring every
    one of them to be covered aborts reviews that could have produced a
    complete sheet from the remaining candidates.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    # Without path_data the plan is the image_change branch:
    # first 103.2, candidates 104.56..115.44, last 116.8, postroll 119.8.
    # Punch a hole around the LAST candidate (115.44), which the selector may
    # legitimately discard in favour of the other eight.
    hole_start = 115.3
    hole_end = 115.6

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [
                {"start_time": after - 5, "end_time": hole_start},
                {"start_time": hole_end, "end_time": before + 5},
            ]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if hole_start < timestamp < hole_end:
                raise OSError("no recording at this instant")
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert all(not (hole_start < stamp < hole_end) for stamp in completed.sample_times)
    # The contractually required frames survive.
    assert 103.2 in completed.sample_times
    assert 116.8 in completed.sample_times
    assert 119.8 in completed.sample_times


async def test_review_nudges_a_motion_frame_out_of_a_recording_hole(
    hass: HomeAssistant, tmp_path
) -> None:
    """A motion frame inside a recording hole moves to a covered instant.

    The path_motion branch uses all three motion peaks, so unlike the change
    candidates there is nothing to discard: the frame itself has to move. A
    motion peak is a sampled instant rather than an exact moment, so shifting it
    within the same motion third keeps its meaning.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    # With path_data the plan is motion_times (105, 107, 109); punch a hole
    # around 107, the middle motion peak.
    hole_start = 106.8
    hole_end = 107.2
    nudge_window = 2.0

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [
                {"start_time": after - 5, "end_time": hole_start},
                {"start_time": hole_end, "end_time": before + 5},
            ]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if hole_start < timestamp < hole_end:
                raise OSError("no recording at this instant")
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    # Six distinct frames, none inside the hole.
    assert len(set(completed.sample_times)) == 6
    assert all(not (hole_start < stamp < hole_end) for stamp in completed.sample_times)
    # The nudged frame stays near the peak it replaces.
    assert any(abs(stamp - 107.0) <= nudge_window for stamp in completed.sample_times)
    # Order and the fixed frames are preserved.
    assert tuple(sorted(completed.sample_times)) == completed.sample_times
    assert 103.2 in completed.sample_times
    assert 116.8 in completed.sample_times
    assert 119.8 in completed.sample_times


async def test_review_records_motion_nudges_in_metadata(
    hass: HomeAssistant, tmp_path
) -> None:
    """A nudged motion frame must be observable in the artifact metadata."""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    hole_start = 106.8
    hole_end = 107.2

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [
                {"start_time": after - 5, "end_time": hole_start},
                {"start_time": hole_end, "end_time": before + 5},
            ]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if hole_start < timestamp < hole_end:
                raise OSError("no recording at this instant")
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.evidence_path is not None
    metadata = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    assert "recording_nudges" in metadata
    assert "107.0" in metadata


async def test_review_still_fails_when_a_required_frame_is_uncovered(
    hass: HomeAssistant, tmp_path
) -> None:
    """A hole under a frame that must be used still fails the activity.

    first/last/postroll are contractually required, so an uncovered one is a
    genuine gap rather than a discarded candidate.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            # Ends before the postroll frame (119.8), so a required frame is
            # uncovered no matter which candidates are discarded.
            return [{"start_time": after - 5, "end_time": 118.0}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    with pytest.raises(MediaError, match="recording_gap"):
        await manager.async_build(record.activity_id)


async def test_review_survives_a_failed_infrared_retry_fetch(
    hass: HomeAssistant, tmp_path
) -> None:
    """A retry fetch that 404s must keep the original frame, not fail the run.

    Retrying an IR frame is a best-effort improvement: the frame itself is
    already downloaded and usable. Frigate purges recordings in the background,
    so a neighbouring instant can 404 even though the original sample is still
    available. Letting that propagate would discard a whole activity over an
    optional upgrade -- the exact opposite of what the repair is for.

    Observed in production: replaying 09-17 08:20 raised
    FrigateApiError(http_404) from the +0.5s IR retry.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    from custom_components.frigate_vision.frigate import FrigateApiError

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # The planned frames succeed; the IR retry neighbour 404s.
            if abs(timestamp - 103.2) < 0.01:
                return _ir_jpeg()
            if abs(timestamp - 103.7) < 0.01:
                raise FrigateApiError("http_404")
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    # The 404 at +0.5s is skipped and a later offset succeeds, so the IR frame
    # is upgraded rather than either aborting or staying green. Which offset
    # wins depends on the ladder's granularity, so assert the intent: the frame
    # moved a little forward and is no longer the original green instant.
    assert 103.2 not in completed.sample_times
    assert 103.0 < completed.sample_times[0] < 105.0


async def test_review_keeps_the_frame_when_every_retry_fetch_fails(
    hass: HomeAssistant, tmp_path
) -> None:
    """When no retry can be fetched at all, the original frame is kept.

    A purged recording window can make every neighbouring instant unfetchable.
    The activity must still produce its six frames rather than failing over an
    optional improvement.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    from custom_components.frigate_vision.frigate import FrigateApiError

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            planned = (103.2, 105.0, 107.0, 109.0, 116.8, 119.8)
            if any(abs(timestamp - value) < 0.01 for value in planned):
                if abs(timestamp - 103.2) < 0.01:
                    return _ir_jpeg()
                return _changing_jpeg(timestamp)
            raise FrigateApiError("http_404")

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    # No replacement was available, so the original IR frame is kept.
    assert 103.2 in completed.sample_times


async def test_retry_helper_does_not_settle_two_frames_on_one_picture(
    hass: HomeAssistant, tmp_path
) -> None:
    """The shared repair helper must keep neighbours on different pictures.

    Frigate snaps a requested timestamp to its nearest stored frame, so two
    instants inside one frame interval return identical bytes. Reproduced on a
    real sheet: frames planned 0.21s apart were repaired to instants 5.7ms apart
    and came back byte-identical, and the uniqueness guard then rejected the
    sheet as `duplicate_evidence_frame` -- failing a whole activity over what was
    only an optional improvement.

    Each frame walks the ladder independently and only checks that a timestamp is
    unused, never that the picture differs, so two neighbours that both clear at
    the same boundary converge. This drives the helper directly, because the
    review selector picks frames by content and would otherwise choose a
    different set than the one that collides.
    """
    from custom_components.frigate_vision.media import (
        frame_is_infrared,
        frame_is_overexposed,
    )

    store = ActivityStore(hass, "entry_1")
    await store.async_load()

    # Two adjacent frames whose planned instants are 0.2s apart, as the real
    # case was: an anchor and the first change frame.
    planned = [(100.0, tmp_path / "planned_0.jpg"), (100.2, tmp_path / "planned_1.jpg")]
    for _, path in planned:
        path.write_bytes(_blown_jpeg())

    class Client:
        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # The camera's own frame grid: 0.5s. Blown before 101.5, clear after,
            # so both repairs first succeed inside the same cell.
            if timestamp < 101.5:
                return _blown_at(timestamp)
            return _jpeg(int(timestamp / 0.5) * 13 % 255)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    selected, _replacements = await manager._async_retry_frames(
        "front",
        tmp_path,
        planned,
        [{"start_time": 90.0, "end_time": 130.0}],
        offsets=(0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0),
        tag="exposure",
        needed=lambda _index, path, _result: frame_is_overexposed(path),
        acceptable=lambda path, _index, _result: (
            not frame_is_overexposed(path) and not frame_is_infrared(path)
        ),
    )

    assert len(selected) == 2
    # The heart of it: the two frames must be different pictures.
    validate_unique_frames([path for _, path in selected])


async def test_overexposure_repair_avoids_frames_that_collide(
    hass: HomeAssistant, tmp_path
) -> None:
    """A repaired sheet must satisfy the production uniqueness guard.

    End-to-end companion to the helper test: whatever the selector chooses, the
    finished sheet has to be shippable.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # Model the camera rather than the request: the picture depends only
            # on which stored frame the instant falls in, so two requests inside
            # one cell return identical bytes, as the real collision did.
            if 102.9 <= timestamp < 108.0:
                return _blown_at(timestamp)
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert tuple(sorted(set(completed.sample_times))) == completed.sample_times
    validate_unique_frames([Path(completed.evidence_path)])


async def test_overexposure_repair_never_accepts_a_night_vision_frame(
    hass: HomeAssistant, tmp_path
) -> None:
    """A blown frame must not be traded for an IR one.

    Both defects come from the camera changing state, so an IR window often sits
    immediately before the blown window. The ladder tries negative offsets first
    and would step straight into it, replacing an unusable frame with a
    different unusable frame -- and, worse, undoing the night-vision repair that
    had already run.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # Earlier than 103.2 is night vision; from 103.2 to 105.0 is blown;
            # after that it is colour, so a good candidate exists forward.
            if timestamp < 103.199:
                return _ir_jpeg()
            if timestamp <= 105.0:
                return _blown_at(timestamp)
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    metadata = json.loads(
        Path(completed.evidence_path).with_suffix(".json").read_text()
    )
    for entry in metadata.get("overexposure_replacements", []):
        assert entry["to"] > 105.0, (
            f"frame {entry['index']} was replaced with an instant that is still "
            f"unusable ({entry['to']})"
        )


def test_crop_person_maps_a_normalised_box_to_pixels() -> None:
    """box 是归一化的，必须按帧的真实尺寸换算。

    帧是 640x360（Frigate 的 detect 流分辨率）。box 的 [x,y,w,h] 直接乘即可。
    """
    from custom_components.frigate_vision.media import crop_person_box

    left, top, right, bottom = crop_person_box(
        (0.25, 0.44, 0.23, 0.52), frame_size=(640, 360), padding=0.0
    )
    assert (left, top) == (160, 158)
    assert right - left == 147
    assert bottom - top == 187


def test_crop_person_adds_padding_on_every_side() -> None:
    """padding 必须四边都加，否则头部或手里的包裹会被切掉。

    用户文档第 6 节明确要求不要紧贴 bbox。
    """
    from custom_components.frigate_vision.media import crop_person_box

    tight = crop_person_box(
        (0.40, 0.40, 0.20, 0.20), frame_size=(640, 360), padding=0.0
    )
    padded = crop_person_box(
        (0.40, 0.40, 0.20, 0.20), frame_size=(640, 360), padding=0.40
    )
    assert padded[0] < tight[0], "左边要更靠左"
    assert padded[1] < tight[1], "上边要更靠上"
    assert padded[2] > tight[2], "右边要更靠右"
    assert padded[3] > tight[3], "下边要更靠下"


def test_crop_person_clamps_to_the_frame() -> None:
    """裁剪区不能越出画面，否则会出现黑边或 Pillow 抛错。"""
    from custom_components.frigate_vision.media import crop_person_box

    left, top, right, bottom = crop_person_box(
        (0.0, 0.0, 0.10, 0.10), frame_size=(640, 360), padding=0.50
    )
    assert left >= 0 and top >= 0
    assert right <= 640 and bottom <= 360


def test_crop_person_rejects_a_degenerate_box() -> None:
    """零尺寸或非有限的 box 要拒绝，不能返回一个空裁剪区。

    Frigate 偶尔会在目标刚出现时给出极小的 box。
    """
    from custom_components.frigate_vision.media import crop_person_box

    for bad in (
        (0.5, 0.5, 0.0, 0.2),
        (0.5, 0.5, 0.2, 0.0),
        (0.5, 0.5, -0.1, 0.2),
        (float("nan"), 0.5, 0.2, 0.2),
    ):
        try:
            crop_person_box(bad, frame_size=(640, 360), padding=0.4)
        except ValueError:
            continue
        raise AssertionError(f"{bad} 应被拒绝")


def test_largest_person_box_picks_by_area_not_order() -> None:
    """取面积最大的那条，而不是第一条或最后一条。

    特写是为了看清衣着与携带物，人物占像素最多的那一帧才看得最清楚。
    Frigate 在目标刚出现时会给出极小的 box，所以「早但大」比「晚但小」更糟。
    """
    updates = (
        (100.0, (0.10, 0.10, 0.05, 0.10)),   # small, first
        (105.0, (0.20, 0.20, 0.30, 0.50)),   # largest, middle
        (110.0, (0.30, 0.30, 0.10, 0.20)),   # small, last
    )
    found = largest_person_box(updates)
    assert found is not None
    timestamp, box = found
    assert timestamp == 105.0
    assert box == (0.20, 0.20, 0.30, 0.50)


def test_largest_person_box_skips_unusable_entries() -> None:
    """畸形或零尺寸的 box 要跳过，不能让它赢下面积比较。"""
    updates = (
        (100.0, (0.1, 0.1, 0.0, 0.5)),        # zero width
        (101.0, (0.1, 0.1, float("nan"), 0.5)),
        (102.0, (0.1, 0.1, -0.2, 0.5)),       # negative
        (103.0, (0.2, 0.2, 0.20, 0.30)),      # the only usable one
    )
    found = largest_person_box(updates)
    assert found is not None
    assert found[0] == 103.0


def test_largest_person_box_returns_none_when_nothing_is_usable() -> None:
    """全不可用时返回 None —— 调用方据此降级，而不是抛错。"""
    assert largest_person_box(()) is None
    assert largest_person_box(((100.0, (0.1, 0.1, 0.0, 0.0)),)) is None


def test_the_sheet_and_the_budget_agree_on_the_column_width() -> None:
    """拼图用的栏宽与预算用的栏宽必须是同一个数字。

    这是实测出来的一个真实隐患：`PERSON_HIGHLIGHT_WIDTH`（const.py）与
    `build_contact_sheet` 的默认参数曾是同一个 448 的两份独立拷贝。若把常量下调
    （例如改成 256）而忘了改另一边，预算变成 767+256=1023 而拼图实际 1216 ——
    预算**小于**实际宽度，于是整张图被缩放，**特写被摧毁**，正是这功能要防的事。
    现在 media.py 直接引用该常量，这条测试把两者钉在一起。
    """
    import inspect

    from custom_components.frigate_vision.const import PERSON_HIGHLIGHT_WIDTH
    from custom_components.frigate_vision.media import build_contact_sheet

    default = inspect.signature(build_contact_sheet).parameters["highlight_width"]
    assert default.default == PERSON_HIGHLIGHT_WIDTH, (
        "拼图的默认栏宽与预算用的常量不同步，会让整张图被缩放"
    )


def test_crop_person_from_frame_uses_the_configured_padding(tmp_path) -> None:
    """裁剪必须真的用 `PERSON_CROP_PADDING`，且比紧贴 box 更大。

    这条测试补一个实测出来的覆盖缺口：把该常量改成 0.0 或 5.0，整套 452 个测试
    **全部通过**，因为其他测试都显式传 `padding=` 调 `crop_person_box`，没有任何
    测试读生产常量，也没有端到端断言裁剪尺寸。于是「人物紧贴边框、头和手里的
    包裹被切掉」——正是这个功能要解决的问题——可以静默上线。

    断言的是「带 padding 的结果严格大于紧贴 box」，而不是某个具体像素数，这样
    常量本身仍可调整，但调成 0 会立刻变红。
    """
    from custom_components.frigate_vision.media import (
        PERSON_CROP_PADDING,
        crop_person_box,
        crop_person_from_frame,
    )

    assert PERSON_CROP_PADDING > 0, "padding 为 0 会把头和手里的包裹切掉"
    # An upper bound too: a large enough padding clamps to the whole frame, and a
    # "close-up" that is the entire frame is just the frame -- the person is no
    # bigger than in any other view, so the feature silently does nothing. 5.0 was
    # measured to collapse a 160x180 box to the full 640x360.
    assert PERSON_CROP_PADDING < 1.0, "padding 太大会退化成整帧，特写就失去意义"

    box = (0.30, 0.25, 0.25, 0.50)
    frame = tmp_path / "original.jpg"
    Image.new("RGB", (640, 360), (180, 140, 110)).save(frame, "JPEG")
    with Image.open(frame) as source:
        frame_size = source.size

    tight = crop_person_box(box, frame_size=frame_size, padding=0.0)
    crop = crop_person_from_frame(frame, box)
    assert crop.size[0] > tight[2] - tight[0], "宽度没有把 padding 算进去"
    assert crop.size[1] > tight[3] - tight[1], "高度没有把 padding 算进去"


def test_crop_person_from_frame_reads_the_real_frame_size(tmp_path) -> None:
    """裁剪必须按帧的真实尺寸换算，而不是假定 640x360。

    box 是归一化的，所以帧尺寸错了，裁出来的位置就错了。当前帧恰好是 640x360，
    硬编码不会有症状；换一个尺寸就能看出来。
    """
    from custom_components.frigate_vision.media import crop_person_from_frame

    box = (0.25, 0.25, 0.50, 0.50)
    for size in ((640, 360), (1280, 720)):
        frame = tmp_path / f"frame_{size[0]}.jpg"
        Image.new("RGB", size, (180, 140, 110)).save(frame, "JPEG")
        crop = crop_person_from_frame(frame, box)
        # Half the frame plus padding on every side: comfortably more than half,
        # and far less than the whole frame. A hardcoded 640x360 would give the
        # same pixel size for both, so the two sizes must differ.
        assert crop.size[0] > size[0] * 0.5
        assert crop.size[0] < size[0]
        assert crop.size[1] > size[1] * 0.5
        assert crop.size[1] < size[1]


def test_a_face_wins_over_a_larger_box_without_one() -> None:
    """正面帧优先于「面积最大但背对镜头」的那一帧 —— 这正是本次要修的问题。

    真实数据：面积最大的 detection 取到的是背影，而四秒前的正面帧更小。
    """
    from custom_components.frigate_vision.media import _prefer_face

    sources = [
        ("back", (0.20, 0.30, 0.40, 0.60)),  # area 0.24, no face
        ("front", (0.30, 0.35, 0.20, 0.30)),  # area 0.06, has a face
    ]
    picked = _prefer_face(sources, {"front": True, "back": False})
    assert picked is not None
    assert picked[0] == "front", "有正脸的帧必须胜过面积更大的背影帧"


def test_within_faces_the_largest_box_wins() -> None:
    """规则是「正脸且最大面积」：正脸是门槛，面积在门槛内决定。"""
    from custom_components.frigate_vision.media import _prefer_face

    sources = [
        ("small", (0.10, 0.10, 0.20, 0.20)),  # face, area 0.04
        ("large", (0.10, 0.10, 0.40, 0.50)),  # face, area 0.20
    ]
    picked = _prefer_face(sources, {"small": True, "large": True})
    assert picked is not None
    assert picked[0] == "large"


def test_no_face_anywhere_falls_back_to_the_largest_box() -> None:
    """全都检不出人脸时，退回面积最大 —— 而不是什么都不选。"""
    from custom_components.frigate_vision.media import _prefer_face

    sources = [
        ("a", (0.10, 0.10, 0.20, 0.20)),
        ("b", (0.10, 0.10, 0.40, 0.50)),
    ]
    picked = _prefer_face(sources, {"a": False, "b": False})
    assert picked is not None
    assert picked[0] == "b"


def test_an_empty_answer_falls_back_to_the_largest_box() -> None:
    """服务不可达时 answers 为空 —— 必须等同于「没开这个功能」。"""
    from custom_components.frigate_vision.media import _prefer_face

    sources = [
        ("a", (0.10, 0.10, 0.20, 0.20)),
        ("b", (0.10, 0.10, 0.40, 0.50)),
    ]
    picked = _prefer_face(sources, {})
    assert picked is not None
    assert picked[0] == "b", "空答案必须退回面积最大，与未启用时逐字节一致"


def test_unasked_candidates_do_not_beat_a_confirmed_face() -> None:
    """只问了部分候选时，没问过的算「未知」，不能压过已确认的正脸。"""
    from custom_components.frigate_vision.media import _prefer_face

    sources = [
        ("unasked", (0.10, 0.10, 0.50, 0.60)),  # biggest, never asked about
        ("asked", (0.10, 0.10, 0.20, 0.20)),  # has a face
    ]
    picked = _prefer_face(sources, {"asked": True})
    assert picked is not None
    assert picked[0] == "asked"


def test_the_service_url_accepts_what_a_person_would_type() -> None:
    """配置项要接受手输的各种写法，否则是个配置陷阱。"""
    from custom_components.frigate_vision.faces import normalise_service_url

    assert normalise_service_url("") == ""
    assert normalise_service_url("   ") == ""
    for typed in (
        "http://192.168.166.50:8788",
        "192.168.166.50:8788",
        "http://192.168.166.50:8788/",
        "http://192.168.166.50:8788/face",
        "192.168.166.50:8788/face",
    ):
        assert normalise_service_url(typed) == "http://192.168.166.50:8788/face", typed


def test_the_face_crop_is_padded_like_the_close_up_itself() -> None:
    """送给人脸服务的裁剪必须和特写用同样的 padding。

    这是实测出的 bug：`_encode_crop` 一开始按 box 边缘直切（137x180），顶着发际线
    切掉了头顶，模型在 6 帧里一帧都没检出；同一批帧用 `PERSON_CROP_PADDING`
    （247x261）能检出 2 帧正脸。紧贴边缘的"人脸"是没有额头和发际线的人脸，而那
    正是检测器要看的东西。

    用同一张图分别走两条路，断言尺寸一致 —— 只要有人把 padding 去掉或改小，
    这里立刻会红。
    """
    import io as _io

    from custom_components.frigate_vision.faces import _encode_crop
    from custom_components.frigate_vision.media import (
        PERSON_CROP_PADDING,
        crop_person_box,
    )

    frame = _changing_jpeg(1.0)
    box = (0.30, 0.35, 0.20, 0.30)
    encoded = _encode_crop(frame, box)
    assert encoded is not None
    with Image.open(_io.BytesIO(encoded)) as sent:
        sent_size = sent.size
    with Image.open(_io.BytesIO(frame)) as source:
        expected = crop_person_box(
            box, frame_size=source.size, padding=PERSON_CROP_PADDING
        )
    assert sent_size == (expected[2] - expected[0], expected[3] - expected[1]), (
        "送给人脸服务的裁剪与特写用的 padding 不一致"
    )
    # And the padding must actually widen the crop past the box itself.
    with Image.open(_io.BytesIO(frame)) as source:
        bare = crop_person_box(box, frame_size=source.size, padding=0.0)
    assert sent_size[0] > bare[2] - bare[0], "裁剪没有留边，头顶会被切掉"


def test_candidates_spread_across_the_activity_not_just_its_start() -> None:
    """候选要覆盖整段活动，不能只取开头几帧。

    这是真实测出来的：用 detection 快照当候选时只有 1 个候选（检出人脸 0 个），
    无从挑选，特写还是背影。改用九宫格自己的 sample_times（6 帧）后有 2 帧检出
    正脸 —— 而正面帧出现在活动中段，取前 N 个就永远看不到。
    """
    from custom_components.frigate_vision.faces import candidate_frames

    frames = [(str(100 + i), f"p{i}") for i in range(9)]
    picked = candidate_frames(frames, 6)
    assert len(picked) == 6
    assert picked[0][0] == "100", "丢了首帧"
    assert picked[-1][0] == "108", "丢了末帧 —— 末帧和首帧一样是候选"
    # Spread, not truncated: the middle frames must be represented.
    assert picked[3][0] not in ("100", "101", "102", "103"), "只取了开头一段"
    # Under the cap everything is kept, in order.
    assert [t for t, _ in candidate_frames(frames[:3], 6)] == ["100", "101", "102"]


async def test_the_close_up_prefers_a_frontal_frame_from_the_sheet(
    hass: HomeAssistant, tmp_path
) -> None:
    """九宫格里有正脸时，特写改用它 —— 用真实回调，不是模拟。

    服务被替换成"只有中段那帧有脸"，以复现用户的场景：detection 快照是背影，
    而九宫格自己的帧里有正面。
    """
    import custom_components.frigate_vision.faces as faces_module

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _sealed_record_with_boxes("activity_1")
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id, "camera": camera, "label": "person",
                "start_time": 100, "end_time": 120,
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera, timestamp, height):
            return _changing_jpeg(timestamp)

        async def async_get_event_snapshot(self, event_id, camera, height):
            # Bright, not _changing_jpeg: the strip below the grid is filled
            # black and the tests locate the crop by luma, so the crop must be
            # clearly lighter than the fill. A dark fixture is invisible to them.
            return _solid_jpeg((240, 240))

    asked: list[str] = []

    async def fake_faces(_hass, _url, frames, **_kwargs):
        # The third frame is the frontal one; everything else is a back.
        for index, (frame_id, _data, _box) in enumerate(frames):
            asked.append(frame_id)
            if index == 2:
                return {frame_id: True}
        return {}

    original = faces_module.faces_for_frames
    faces_module.faces_for_frames = fake_faces
    try:
        manager = MediaManager(
            hass, store, Client(), tmp_path,
            ZoneRoles(near=frozenset({"near"}), transition=frozenset({"mid"}),
                      far=frozenset({"far"})),
            person_highlight=True,
            target_width=_TARGET_WIDTH,
            face_service_url="http://face.invalid/face",
        )
        completed = await manager.async_build(record.activity_id)
    finally:
        faces_module.faces_for_frames = original

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert asked, "候选帧没被送去问人脸 —— 说明候选源还是 detection 快照"
    assert len(asked) > 1, "只问了一个候选，无从挑选"


async def test_an_unreachable_face_service_still_produces_a_close_up(
    hass: HomeAssistant, tmp_path, socket_enabled
) -> None:
    """人脸服务连不上时，特写照常生成 —— 它是提升项，不是关键路径。

    这是整个功能的硬约束：服务没配、连不上、超时、答不上来，行为必须和
    「没这个功能」完全一样，绝不能因此丢掉整条活动的分析。

    `socket_enabled` 是必须的：这个测试**故意**连一个没人监听的端口
    （127.0.0.1:1）来模拟「服务连不上」。HA 的测试夹具默认禁止 socket，会把它
    判成「测试偷偷联网」——但这里联网正是被测行为本身。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _sealed_record_with_boxes("activity_1")
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id, "camera": camera, "label": "person",
                "start_time": 100, "end_time": 120,
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera, timestamp, height):
            return _changing_jpeg(timestamp)

        async def async_get_event_snapshot(self, event_id, camera, height):
            # Bright, not _changing_jpeg: the strip below the grid is filled
            # black and the tests locate the crop by luma, so the crop must be
            # clearly lighter than the fill. A dark fixture is invisible to them.
            return _solid_jpeg((240, 240))

    # A port nothing is listening on: every request fails immediately.
    manager = MediaManager(
        hass, store, Client(), tmp_path,
        ZoneRoles(near=frozenset({"near"}), transition=frozenset({"mid"}),
                  far=frozenset({"far"})),
        person_highlight=True,
        target_width=_TARGET_WIDTH,
        face_service_url="http://127.0.0.1:1/face",
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert completed.evidence_path is not None
    with Image.open(completed.evidence_path) as sheet:
        # 服务不可达时仍须拼出特写（退回面积最大），而不是降级成纯九宫格。
        _assert_has_close_up_strip(sheet, rows=2)


async def test_the_manager_adds_a_close_up_when_the_option_is_on(
    hass: HomeAssistant, tmp_path
) -> None:
    """开启后证据图确实变宽并含特写栏 —— 端到端，防接线错误。

    这是整条链路的验收点：box_updates -> 选最大 -> 配最近帧 -> 从原始帧裁剪
    -> 拼到右侧。任何一环断了，宽度都不会超过 target_width。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _sealed_record_with_boxes("activity_1")
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id, "camera": camera, "label": "person",
                "start_time": 100, "end_time": 120,
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera, timestamp, height):
            return _changing_jpeg(timestamp)

        async def async_get_event_snapshot(self, event_id, camera, height):
            # Bright, not _changing_jpeg: the strip below the grid is filled
            # black and the tests locate the crop by luma, so the crop must be
            # clearly lighter than the fill. A dark fixture is invisible to them.
            return _solid_jpeg((240, 240))

    manager = MediaManager(
        hass, store, Client(), tmp_path,
        ZoneRoles(near=frozenset({"near"}), transition=frozenset({"mid"}),
                  far=frozenset({"far"})),
        person_highlight=True,
        target_width=_TARGET_WIDTH,
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.evidence_path is not None
    with Image.open(completed.evidence_path) as sheet:
        _assert_has_close_up_strip(sheet, rows=2)


async def test_the_manager_omits_the_close_up_when_the_option_is_off(
    hass: HomeAssistant, tmp_path
) -> None:
    """关闭时走原路径 —— 拼图宽度不变，也不含特写栏（零回归）。"""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _sealed_record_with_boxes("activity_1")
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id, "camera": camera, "label": "person",
                "start_time": 100, "end_time": 120,
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera, timestamp, height):
            return _changing_jpeg(timestamp)

        async def async_get_event_snapshot(self, event_id, camera, height):
            # Bright, not _changing_jpeg: the strip below the grid is filled
            # black and the tests locate the crop by luma, so the crop must be
            # clearly lighter than the fill. A dark fixture is invisible to them.
            return _solid_jpeg((240, 240))

    manager = MediaManager(
        hass, store, Client(), tmp_path,
        ZoneRoles(near=frozenset({"near"}), transition=frozenset({"mid"}),
                  far=frozenset({"far"})),
        person_highlight=False,
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.evidence_path is not None
    with Image.open(completed.evidence_path) as sheet:
        assert sheet.size[0] == 1920, "关闭时拼图不该被提前缩放"


async def test_the_manager_still_builds_when_no_box_was_recorded(
    hass: HomeAssistant, tmp_path
) -> None:
    """没有 box 时降级为普通拼图，但分析照常完成（不抛错）。"""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _sealed_record_with_boxes("activity_1", boxes=())
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id, "camera": camera, "label": "person",
                "start_time": 100, "end_time": 120,
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera, timestamp, height):
            return _changing_jpeg(timestamp)

        async def async_get_event_snapshot(self, event_id, camera, height):
            # Bright, not _changing_jpeg: the strip below the grid is filled
            # black and the tests locate the crop by luma, so the crop must be
            # clearly lighter than the fill. A dark fixture is invisible to them.
            return _solid_jpeg((240, 240))

    manager = MediaManager(
        hass, store, Client(), tmp_path,
        ZoneRoles(near=frozenset({"near"}), transition=frozenset({"mid"}),
                  far=frozenset({"far"})),
        person_highlight=True,
        target_width=_TARGET_WIDTH,
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    with Image.open(completed.evidence_path) as sheet:
        assert sheet.size[0] == 1920, "无 box 时应退回普通拼图"


async def test_a_review_activity_gets_a_close_up_from_its_detections(
    hass: HomeAssistant, tmp_path
) -> None:
    """review 活动也必须拿到特写 —— 它的 box 只能从 detection 上取。

    实测发现的缺口：这个家里记录的**每一条**活动都来自 Frigate 的 review
    （72 条 standalone_review + 11 条 manual_review，没有一条来自原始 event），
    而 review 消息只列 `detections`、**本身不带 box** —— box 只存在于 detection
    上。原先只有 `parse_event_payload` 记 box，于是 `box_updates` 永远是空的，
    特写一次都没生成过，而开关看起来已经打开。

    这条测试走真实路径：review 记录 -> 按 detection id 向 Frigate 要 box ->
    裁剪 -> 拼到右侧。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _sealed_review_record("activity_1")
    assert record.box_updates == (), "review 记录本身不带 box，这正是被测的前提"
    await store.async_create(record)

    asked: list[str] = []

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            asked.append(event_id)
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 100,
                "end_time": 120,
                # The box lives here, not on the review message.
                "data": {"box": [0.30, 0.35, 0.32, 0.55]},
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera, timestamp, height):
            return _changing_jpeg(timestamp)

        async def async_get_event_snapshot(self, event_id, camera, height):
            # Bright, not _changing_jpeg: the strip below the grid is filled
            # black and the tests locate the crop by luma, so the crop must be
            # clearly lighter than the fill. A dark fixture is invisible to them.
            return _solid_jpeg((240, 240))

    manager = MediaManager(
        hass, store, Client(), tmp_path,
        ZoneRoles(near=frozenset({"near"}), transition=frozenset({"mid"}),
                  far=frozenset({"far"})),
        person_highlight=True,
        target_width=_TARGET_WIDTH,
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert asked, "没有按 detection id 去要 box —— review 路径上 box 无处可来"
    with Image.open(completed.evidence_path) as sheet:
        # This fixture plans six samples, so the grid is two rows here.
        _assert_has_close_up_strip(sheet, rows=2)


async def test_a_review_activity_without_a_box_degrades_instead_of_failing(
    hass: HomeAssistant, tmp_path
) -> None:
    """detection 查不到 box（或查询失败）时降级，分析照常完成。"""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _sealed_review_record("activity_1")
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id, "camera": camera, "label": "person",
                "start_time": 100, "end_time": 120,
                "data": {},  # no box
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera, timestamp, height):
            return _changing_jpeg(timestamp)

        async def async_get_event_snapshot(self, event_id, camera, height):
            # Bright, not _changing_jpeg: the strip below the grid is filled
            # black and the tests locate the crop by luma, so the crop must be
            # clearly lighter than the fill. A dark fixture is invisible to them.
            return _solid_jpeg((240, 240))

    manager = MediaManager(
        hass, store, Client(), tmp_path,
        ZoneRoles(near=frozenset({"near"}), transition=frozenset({"mid"}),
                  far=frozenset({"far"})),
        person_highlight=True,
        target_width=_TARGET_WIDTH,
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY, "没有 box 不该让分析失败"
    with Image.open(completed.evidence_path) as sheet:
        assert sheet.size[0] == 1920, "没有 box 时应退回普通拼图"


# --------------------------------------------------------------------------- #
# A replayed activity shares its root activity's evidence sheet.
#
# Measured on the deployment, 2026-09-28: `retry_failed` produced a record whose
# `evidence_path` pointed at the ROOT activity's JPEG while its own `activity_id`
# carried the `_attempt_1` suffix. The next `async_setup_entry` walked that record
# in `async_restore_registry`, found the filename did not equal its `activity_id`,
# raised `invalid_output_path`, and the whole config entry went to `setup_error`
# -- every entity unavailable until the store was repaired by hand.
#
# The sharing itself is deliberate, not a bug: a replay reuses the sheet that was
# already built, which is why `async_create_retry` keeps `EVIDENCE_READY` when the
# evidence has not expired (pinned by
# `test_a_lost_503_activity_with_evidence_is_replayed_with_its_sheet`). What was
# missing is that the two path validators did not know about it.
# --------------------------------------------------------------------------- #


def _write_sheet(path: Path, activity_id: str, samples: tuple[float, ...]) -> None:
    """One artifact pair exactly as `media.py` writes it, under a chosen id.

    The metadata carries the ROOT activity's id on purpose, mirroring what the
    deployment's own replayed record pointed at.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jpeg(128))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "activity_id": activity_id,
                "plan_version": 4,
                "mode": "review_six",
                "selection_source": "path_motion",
                "sample_times": list(samples),
            }
        ),
        "utf-8",
    )


async def test_a_replayed_activity_restores_from_its_root_sheet(
    hass: HomeAssistant, tmp_path
) -> None:
    """The restart path must accept a replay that shares its root's sheet.

    This is the production failure, reduced: rebuild the store the way
    `retry_failed` leaves it, then run exactly what `async_setup_entry` runs.
    """
    samples = (1.0, 2.0, 3.0)
    sheet = tmp_path / "entry_1" / "review_root.jpg"
    _write_sheet(sheet, "review_root", samples)

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="review_root",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.FAILED,
            created_at=1,
            updated_at=2,
            camera="front",
            error_code="provider_http_502",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(sheet),
            evidence_media_url="media-source://frigate_vision/entry_1/review_root",
            sample_times=samples,
        )
    )
    retry = await store.async_create_retry("review_root", now=3)
    assert retry.stage is ActivityStage.EVIDENCE_READY

    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )

    await manager.async_restore_registry()

    registry = hass.data[DATA_MEDIA_REGISTRY]
    assert registry["entry_1/review_root"] == sheet
    assert registry["entry_1/review_root_attempt_1"] == sheet, (
        "the replay must be registered against the sheet it actually reuses"
    )


async def test_a_replayed_activity_continues_analysis_from_its_root_sheet(
    hass: HomeAssistant, tmp_path
) -> None:
    """The same acceptance on the other path: `async_build` on EVIDENCE_READY.

    `async_build` repeats the identical check, so a fix applied only to
    `async_restore_registry` would leave the replay unable to be analysed.
    """
    samples = (1.0, 2.0, 3.0)
    sheet = tmp_path / "entry_1" / "review_root.jpg"
    _write_sheet(sheet, "review_root", samples)

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="review_root",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.FAILED,
            created_at=1,
            updated_at=2,
            camera="front",
            error_code="provider_http_502",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(sheet),
            evidence_media_url="media-source://frigate_vision/entry_1/review_root",
            sample_times=samples,
        )
    )
    retry = await store.async_create_retry("review_root", now=3)

    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )

    built = await manager.async_build(retry.activity_id)

    assert built.stage is ActivityStage.EVIDENCE_READY
    assert built.evidence_path == str(sheet), "replay keeps the sheet it reused"


async def test_a_replay_still_refuses_a_sheet_belonging_to_neither_id(
    hass: HomeAssistant, tmp_path
) -> None:
    """The acceptance must stay narrow: an unrelated file is still NOT registered.

    Without this, "accept the root's sheet" could be implemented as "accept any
    path", which is the sandbox escape the canonical check exists to prevent.

    拒绝的方式是**跳过并记日志**，不是抛异常。改成抛异常的那一版（历史行为）会让
    整个 config entry 进入 `setup_error`：`MediaError` 是 `RuntimeError`，而
    `async_setup_entry` 只转换 `ModelValidationError`/`FrigateApiError`/`OSError`，
    于是所有实体不可用、**连报修卡都不会有**，值还躺在 `.storage` 里够不着。这条
    路径的 docstring 自己就记着 2026-09-28 发生过一次。
    """
    samples = (1.0, 2.0, 3.0)
    stranger = tmp_path / "entry_1" / "someone_elses.jpg"
    _write_sheet(stranger, "someone_else", samples)

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="review_root",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            created_at=1,
            updated_at=2,
            camera="front",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(stranger),
            evidence_media_url="media-source://frigate_vision/entry_1/review_root",
            sample_times=samples,
        )
    )
    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )

    # Must not raise: one unverifiable record cannot take the whole entry offline.
    await manager.async_restore_registry()

    registry = hass.data.get(
        "frigate_vision_media_registry", {}
    )
    assert "entry_1/review_root" not in registry, (
        "指向别人的文件被登记了——那正是 canonical 检查要防的事"
    )


async def test_one_unverifiable_record_does_not_hide_the_others(
    hass: HomeAssistant, tmp_path
) -> None:
    """一条坏记录不能让**其他**好记录也恢复不了。

    历史行为是抛异常：不仅整个 entry 进入 `setup_error`（所有实体不可用），而且
    扫描在第一条坏记录处就中断——排在它后面的好记录**永远不会**回到注册表。于是
    媒体源对一个本可用的 artifact 报"找不到"。

    这里两个记录：坏的在前面（按 activity_id 排序），好的在后面。好的必须仍然被登记。
    """
    samples = (1.0, 2.0, 3.0)
    # "a_bad" sorts before "z_good", and its path belongs to neither of its ids.
    stranger = tmp_path / "entry_1" / "someone_elses.jpg"
    _write_sheet(stranger, "someone_else", samples)
    good = tmp_path / "entry_1" / "z_good.jpg"
    _write_sheet(good, "z_good", samples)

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    for activity_id, path in (("a_bad", stranger), ("z_good", good)):
        await store.async_create(
            ActivityRecord(
                activity_id=activity_id,
                entry_id="entry_1",
                source=ActivitySource.STANDALONE_REVIEW,
                stage=ActivityStage.EVIDENCE_READY,
                created_at=1,
                updated_at=2,
                camera="front",
                evidence_mode="review_six",
                evidence_revision=1,
                evidence_path=str(path),
                evidence_media_url=(
                    f"media-source://frigate_vision/entry_1/{activity_id}"
                ),
                sample_times=samples,
            )
        )
    manager = MediaManager(
        hass, store, object(), tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )

    await manager.async_restore_registry()

    registry = hass.data.get("frigate_vision_media_registry", {})
    assert "entry_1/z_good" in registry, (
        "排在一坏记录之后的好记录没有被恢复——扫描在坏记录处中断了"
    )
    assert "entry_1/a_bad" not in registry, "坏记录不该被登记"


async def test_metadata_that_is_not_a_json_object_is_ignored_not_fatal(
    tmp_path,
) -> None:
    """媒体目录里的 `.json` 是用户可改的字节；内容合法但非对象时不能打挂集成。

    实测：`null` / `[]` / `"text"` / `123` / `true` 全都让 `_read_existing` 抛
    `AttributeError`——因为它直接对解析结果调用 `.get()`，而 `AttributeError` 不在
    该函数的 `except` 列表里，于是它一路逃到 `async_setup_entry`，让整个 entry 进入
    `setup_error`。文件在用户的媒体目录里，手改或半截写入都会产生这种内容。
    """
    from custom_components.frigate_vision.media import _read_existing

    sheet = tmp_path / "a.jpg"
    Image.new("L", (64, 36), 100).save(sheet, "JPEG", quality=90)
    metadata = tmp_path / "a.json"

    for payload in ("null", "[]", '"text"', "123", "true", "3.5"):
        metadata.write_text(payload, encoding="utf-8")
        assert _read_existing(sheet, metadata, "a") is None, (
            f"metadata={payload!r} 应被当作不可读返回 None，而不是抛异常"
        )


async def test_a_dict_payload_with_hostile_values_is_ignored_not_fatal(
    tmp_path,
) -> None:
    """结构是对的对象、但字段内容是恶意的，也不能抛。

    实测过的载荷（全部返回 None，无一逃逸）：`mode` 是对象、`sample_times` 是字符串、
    `sample_times` 是 5000 位数字、`sample_times` 里嵌套列表、`sample_times` 里是布尔。

    最后两种尤其值得钉住：`float(True)` 是 1.0（布尔是 int 的子类），而超长数字串会
    撞上 CPython 的整数转换上限——两者都可能绕过"看着像检查"的代码。
    """
    from custom_components.frigate_vision.media import _read_existing

    sheet = tmp_path / "a.jpg"
    Image.new("L", (64, 36), 100).save(sheet, "JPEG", quality=90)
    metadata = tmp_path / "a.json"

    hostile = (
        # `mode` is an object: str() succeeds, so this one is accepted as a mode
        # string, which is harmless -- the metadata is advisory.
        '{"activity_id":"a","plan_version":4,"mode":{},"sample_times":[1,2,3]}',
        '{"activity_id":"a","plan_version":4,"mode":"x","sample_times":"abc"}',
        '{"activity_id":"a","plan_version":4,"mode":"x","sample_times":"'
        + "9" * 5000
        + '"}',
        '{"activity_id":"a","plan_version":4,"mode":"x","sample_times":[[1],2,3]}',
        '{"activity_id":"a","plan_version":4,"mode":"x","sample_times":[true,1,2]}',
        '{"activity_id":"a","plan_version":4,"mode":"x","sample_times":[1,2,3],'
        '"selection_source":{}}',
    )
    for payload in hostile:
        metadata.write_text(payload, encoding="utf-8")
        # Must not raise. Most return None; the first is allowed to return a tuple
        # because a dict `mode` stringifies without error and the field is advisory.
        _read_existing(sheet, metadata, "a")


async def test_metadata_that_is_a_broken_sheet_is_ignored_not_fatal(
    tmp_path,
) -> None:
    """同上的另一半：结构是对的对象但内容坏掉，也必须返回 None。"""
    from custom_components.frigate_vision.media import _read_existing

    sheet = tmp_path / "a.jpg"
    Image.new("L", (64, 36), 100).save(sheet, "JPEG", quality=90)
    metadata = tmp_path / "a.json"

    for payload in (
        "{}",
        '{"activity_id": "a"}',
        '{"activity_id": "a", "plan_version": 999}',
        '{"activity_id": "a", "plan_version": 4}',
        '{"activity_id": "a", "plan_version": 4, "mode": "x", "sample_times": null}',
        '{"activity_id": "a", "plan_version": 4, "mode": "x", "sample_times": [1]}',
    ):
        metadata.write_text(payload, encoding="utf-8")
        assert _read_existing(sheet, metadata, "a") is None, (
            f"metadata={payload!r} 应被当作不可读返回 None"
        )


async def test_cleanup_keeps_a_sheet_a_live_replay_still_points_at(
    hass: HomeAssistant, tmp_path
) -> None:
    """清理不能删掉**另一个记录仍在用**的拼图——replay 与它的 root 共享同一个文件。

    机制：`async_create_retry` 用 `replace(original, ...)` 造出 replay，于是它
    **继承** root 的 `evidence_path`；两个记录的 `evidence_path` 指向同一个文件。
    而清理是按**记录**逐个判断的，只看该记录自己的 `updated_at`：

        root 40 天前（过期）+ replay 刚建立（仍然有效）

    于是 root 那一轮把文件删了，而**刚刚建立、仍然有效**的 replay 还指着它。实测
    后果：下次启动 `async_restore_registry` 读不到文件；在 #2 修好之前，这会让整个
    config entry 进入 `setup_error`——所有实体不可用。

    判据是"还有别人在用吗"，不是"这条记录自己多大"。
    """
    now = time.time()
    store = ActivityStore(hass, "entry_1")
    await store.async_load()

    root = tmp_path / "media"
    (root / "entry_1").mkdir(parents=True)
    shared = root / "entry_1" / "review_root.jpg"
    _write_sheet(shared, "review_root", (1.0, 2.0, 3.0))

    old = now - 40 * 86400
    # A FAILED root is the realistic replay source: `async_create_retry` refuses a
    # COMPLETED one (`retry_not_safe`), and a failure is exactly what an operator
    # replays. It still carries an evidence path, which is what the replay inherits.
    await store.async_create(
        ActivityRecord(
            activity_id="review_root",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.FAILED,
            created_at=old,
            updated_at=old,
            camera="front",
            error_code="media_retry_exhausted",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(shared),
            evidence_media_url="media-source://frigate_vision/entry_1/review_root",
            sample_times=(1.0, 2.0, 3.0),
        )
    )
    retry = await store.async_create_retry("review_root", now=now)
    assert retry.evidence_path == str(shared), (
        "precondition: 实测 replay 继承 root 的 evidence_path"
    )

    manager = MediaManager(
        hass, store, object(), root,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    await manager.async_cleanup(retention_days=7, now=now)

    assert shared.is_file(), (
        "刚建立的 replay 还指着这张拼图，清理却把它删了——重启后该记录读不到证据"
    )


async def test_cleanup_still_removes_a_sheet_nothing_points_at(
    hass: HomeAssistant, tmp_path
) -> None:
    """加了「还有别人在用吗」之后，没人用的过期拼图**仍然**要删掉。

    否则保留期就形同虚设——这是清理功能存在的一半理由。
    """
    now = time.time()
    store = ActivityStore(hass, "entry_1")
    await store.async_load()

    root = tmp_path / "media"
    (root / "entry_1").mkdir(parents=True)
    lonely = root / "entry_1" / "review_old.jpg"
    _write_sheet(lonely, "review_old", (1.0, 2.0, 3.0))
    old = now - 40 * 86400
    await store.async_create(
        ActivityRecord(
            activity_id="review_old",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.COMPLETED,
            created_at=old,
            updated_at=old,
            camera="front",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(lonely),
            evidence_media_url="media-source://frigate_vision/entry_1/review_old",
            sample_times=(1.0, 2.0, 3.0),
        )
    )

    manager = MediaManager(
        hass, store, object(), root,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    removed = await manager.async_cleanup(retention_days=7, now=now)

    assert not lonely.is_file(), "没人引用的过期拼图必须被删掉"
    assert "review_old" in removed


async def test_one_unvalidatable_path_does_not_stop_the_whole_sweep(
    hass: HomeAssistant, tmp_path
) -> None:
    """一条路径校验失败的记录不能让**其余**记录的保留期静默失效。

    实测（两种排序）：

    | 坏记录的排序 | 结果 |
    |---|---|
    | 在前（`a_bad`） | 排在它**之后**的过期拼图永远不被清理 |
    | 在后（`z_bad`） | 排在它之前的正常清理 |

    而那条坏记录**永远不会被移除**（抛错发生在标记过期之前），所以它**永久阻塞**：
    定时任务每天重试、每天撞同一堵墙，`media_retention_days` 对排在它之后的记录
    静默失效，媒体目录无限增长。

    **仅靠配置就能触发**：媒体根目录来自 `hass.config.media_dirs`，用户改一下媒体
    目录，所有旧记录的路径就都落在新根之外。
    """
    now = time.time()
    store = ActivityStore(hass, "entry_1")
    await store.async_load()

    root = tmp_path / "media"
    (root / "entry_1").mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"\xff\xd8\xff\xd9")
    good = root / "entry_1" / "z_good.jpg"
    _write_sheet(good, "z_good", (1.0, 2.0, 3.0))

    old = now - 40 * 86400
    # "a_bad" sorts FIRST and its path is outside the media root; "z_good" sorts after.
    for activity_id, path in (("a_bad", outside), ("z_good", good)):
        await store.async_create(
            ActivityRecord(
                activity_id=activity_id,
                entry_id="entry_1",
                source=ActivitySource.STANDALONE_REVIEW,
                stage=ActivityStage.COMPLETED,
                created_at=old,
                updated_at=old,
                camera="front",
                evidence_mode="review_six",
                evidence_revision=1,
                evidence_path=str(path),
                evidence_media_url=(
                    f"media-source://frigate_vision/entry_1/{activity_id}"
                ),
                sample_times=(1.0, 2.0, 3.0),
            )
        )

    manager = MediaManager(
        hass, store, object(), root,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    removed = await manager.async_cleanup(retention_days=7, now=now)

    assert not good.is_file(), (
        "排在一坏记录之后的好记录没被清理——扫描在坏记录处中止了"
    )
    assert "z_good" in removed
    assert outside.is_file(), "根目录之外的文件永远不能被删除"
