"""Tests for the two silent-failure defects in the evidence planner.

Both defects share a shape: a guard that is *absent* rather than wrong. Nothing
raises, nothing logs, and the activity either loses frames or vanishes entirely.
That is why they need tests of their own rather than a line in an existing one.

Defect A -- `by_role["first"]` in the `image_change` branch.
    The planner mints the roles `first`/`last`/`postroll`, and the branch indexes
    `by_role` by those literal names. Any change that stops minting one of them
    turns the lookup into a `KeyError`, which `runtime.py` does not catch
    (`except (OSError, FrigateApiError, MediaError)`), so the activity stays at
    SEALED forever with no `error_code` and no Repair.

Defect B -- `recordings_cover(tuple(fixed.values()))`.
    `fixed` is filtered by the same three literal role names. When that filter
    yields nothing, `recordings_cover(())` is vacuously true because it is built
    on `all()`, and the recording-hole guard stops guarding.

The tests below pin the *behaviour*: a required frame that cannot be fetched
must fail loudly, and the required-frame set must never be empty.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from homeassistant.core import HomeAssistant
from PIL import Image

from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import (
    MediaError,
    MediaManager,
    recordings_cover,
    select_required_representatives,
)
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
)
from custom_components.frigate_vision.store import ActivityStore


def _jpeg(value: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 36), (value, value, value)).save(output, "JPEG")
    return output.getvalue()


def _changing_jpeg(timestamp: float) -> bytes:
    """A frame whose brightness varies smoothly and monotonically with time.

    Copied in spirit from `test_media._changing_jpeg`: fixtures must return a
    *different* picture for a different instant, or the uniqueness guard rejects
    a sheet the fixture wrongly made identical.
    """
    level = int((timestamp * 12) % 200) + 20
    return _jpeg(level)


def _record(activity_id: str = "review_entry_1_front_review_silent") -> ActivityRecord:
    return ActivityRecord(
        activity_id=activity_id,
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=120,
        camera="front",
        review_ids=("review_silent",),
        detection_ids=("event_1",),
        finalization_deadline=120,
    )


class TestRecordingsCoverCannotBeVacuous:
    """`all()` over an empty sequence is True, which silently disables a guard.

    The guard exists to fail an activity whose needed frame falls in a recording
    hole. If the caller ever passes an empty set of needed frames, the guard
    reports success instead -- and the sheet is built from whatever arrived.
    """

    def test_an_empty_required_set_is_not_treated_as_covered(self) -> None:
        """Pin the vacuous semantics explicitly, so callers cannot rely on it.

        `recordings_cover` answers "is every time in this list inside a
        recording", and an empty list trivially satisfies that. The defect is
        that a caller can reach it with an empty list after a filter that was
        supposed to select required frames. Asserting the semantics here makes
        the trap visible and keeps a future change to `all()` deliberate.
        """
        recordings = [{"start_time": 100.0, "end_time": 110.0}]
        assert recordings_cover((), recordings) is True

    def test_required_representatives_are_never_empty(self) -> None:
        """The structural fix for defect B.

        Rather than filtering by role name (which can yield nothing), the
        required set is derived from the frames themselves, so it cannot be empty
        whenever a sheet exists.
        """
        required = select_required_representatives(
            (103.2, 105.0, 107.0), person_start=103.0, person_end=117.0
        )
        assert required, "a non-empty selection must yield a non-empty requirement"
        assert all(timestamp in {103.2, 105.0, 107.0} for timestamp in required)

    def test_required_representatives_prefer_the_person_window(self) -> None:
        """A frame showing the person is not optional.

        The window is what gives the sheet its meaning, so a requirement built
        from the frames must pick frames inside it when any exist.
        """
        required = select_required_representatives(
            (90.0, 104.0, 130.0), person_start=103.0, person_end=117.0
        )
        assert 104.0 in required

    def test_required_representatives_always_cover_the_extremes(self) -> None:
        """The sheet's own boundaries are required whatever the window says.

        The latest frame is the emptied scene and the earliest is where the
        person appears, so both must be covered even when they sit outside the
        person window -- otherwise the guard would not notice a hole under them.
        """
        required = select_required_representatives(
            (90.0, 130.0), person_start=103.0, person_end=117.0
        )
        assert required == (90.0, 130.0)

    def test_an_empty_selection_is_refused_not_answered_with_empty(self) -> None:
        """The producer refuses an empty input rather than returning ().

        Returning an empty tuple is exactly the trap: `recordings_cover(())` is
        True, so an empty requirement reads as "covered". Refusing here makes
        that unreachable and, unlike a caller-side check, directly testable --
        the build path always plans at least three frames, so a caller guard
        could never be exercised.
        """
        with pytest.raises(MediaError):
            select_required_representatives(
                (), person_start=103.0, person_end=117.0
            )


class TestTheCoverageGuardIsWiredToTheSemanticRequirement:
    """The guard must be fed the semantic set, not the role-filtered one.

    Reverting only the call site (leaving the new helper in place, unused) is a
    mutation the behavioural tests cannot see: every plan the build path can
    produce happens to mint all three role names, so `fixed` is never empty in
    practice and both forms pass. That is precisely why the defect survived --
    it is invisible until the planner changes.

    So this is a structural assertion, in the same spirit as
    `test_the_sheet_has_no_dead_highlight_width_parameter`: it pins which helper
    the coverage guard consumes, so re-wiring it back has to be deliberate.
    """

    def test_the_build_feeds_the_guard_the_semantic_requirement(self) -> None:
        import inspect

        from custom_components.frigate_vision import media as media_module

        source = inspect.getsource(media_module.MediaManager._async_build_locked)

        assert "select_required_representatives(" in source, (
            "the coverage guard no longer derives its required set from the "
            "frames themselves"
        )
        assert "recordings_cover(required" in source, (
            "the guard must be given the semantic requirement, not `fixed`"
        )
        # The old shape must not come back: feeding `fixed` made the check
        # vacuous the moment the role names changed.
        assert "recordings_cover(tuple(fixed.values())" not in source, (
            "the role-filtered set is back in the coverage guard; it can be "
            "empty, and `all()` over nothing is true"
        )

    def test_the_semantic_requirement_is_stricter_than_the_role_filter(
        self,
    ) -> None:
        """The new set must actually cover more than the three role names did.

        A change that merely renamed the three names would still miss a hole
        under any other frame, so the requirement has to include the sheet's
        extremes -- which the role filter never did for the probe cells.
        """
        frames = (103.2, 105.0, 112.0, 125.0, 116.8, 119.8)
        required = select_required_representatives(
            frames, person_start=103.2, person_end=116.8
        )
        # The latest frame is the emptied scene; the role filter would have
        # required 119.8 instead and never noticed a hole under 125.0.
        assert max(frames) in required
        assert min(frames) in required
        assert 125.0 in required


class TestARecordingHoleFailsLoudly:
    """A frame that must be used and cannot be fetched fails the activity."""

    async def test_a_hole_under_a_required_frame_is_a_recording_gap(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Existing behaviour, kept as the regression net for the fix."""
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record()
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

            async def async_get_recordings(
                self, camera: str, after: float, before: float
            ):
                # Ends before the postroll frame, so a required frame is
                # uncovered no matter which candidates are discarded.
                return [{"start_time": after - 5, "end_time": 118.0}]

            async def async_get_snapshot(
                self, camera: str, timestamp: float, height: int
            ):
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


class TestTheImageChangeBranchDoesNotEscapeAsAKeyError:
    """A build must fail with a MediaError, never with a bare KeyError.

    `runtime.py` catches only `(OSError, FrigateApiError, MediaError)`. Anything
    else escapes the retry loop, leaving the activity at SEALED with no error
    code: invisible in the UI, absent from `ISSUES`, no `last_error`, and the
    identical failure after every restart.

    Defect A was a bare `by_role["first"]` lookup in the `image_change` branch.
    The lookup only fails when the planner stops minting that role -- which no
    end-to-end fixture produces, because the branch mints the names itself. So
    the defect is pinned where it actually lives: the anchor helper. A test that
    drove the whole build would pass both before and after the fix, which is the
    trap this file exists to avoid.
    """

    def test_a_missing_role_is_reported_as_a_media_error(self) -> None:
        """The exact condition the old code turned into a bare KeyError."""
        from custom_components.frigate_vision.media import _required_role

        with pytest.raises(MediaError) as caught:
            _required_role({}, "first")

        # The message must name the missing role: a future reader has to be able
        # to tell which anchor the sheet needed.
        assert "first" in str(caught.value)

    def test_every_anchor_role_is_guarded(self) -> None:
        """All three anchors, not just the one that happened to be reported."""
        from custom_components.frigate_vision.media import _required_role

        for role in ("first", "last", "postroll"):
            with pytest.raises(MediaError):
                _required_role({}, role)

    def test_a_present_role_is_returned_unchanged(self) -> None:
        """The guard must not alter the value it guards."""
        from custom_components.frigate_vision.media import _required_role

        frame = (103.2, "path")
        assert _required_role({"first": frame}, "first") == frame

    def test_the_guard_is_not_satisfied_by_a_different_role(self) -> None:
        """A dict holding only `last` must not answer for `first`."""
        from custom_components.frigate_vision.media import _required_role

        with pytest.raises(MediaError):
            _required_role({"last": (116.8, "path")}, "first")

    def test_an_incomplete_role_map_is_rejected_for_every_anchor(self) -> None:
        """A map missing any single anchor must fail, not just a fully empty one.

        The old code indexed three names in a row. Guarding only the first leaves
        the other two as bare lookups, which is the same defect one line later --
        and the end-to-end path cannot see it, because the branch mints all three
        names itself. So each anchor is checked against a map that holds exactly
        the other two.
        """
        from custom_components.frigate_vision.media import _required_role

        anchors = ("first", "last", "postroll")
        for missing in anchors:
            present = {
                role: (float(index), Path(f"{role}.jpg"))
                for index, role in enumerate(anchors)
                if role != missing
            }
            with pytest.raises(MediaError):
                _required_role(present, missing)

    def test_no_anchor_is_read_by_bare_indexing(self) -> None:
        """All three anchors go through the guard, not just the reported one.

        Guarding `first` alone leaves `by_role["last"]` and `by_role["postroll"]`
        as the original defect, one line further down -- and no behavioural test
        can see it, because the branch always mints those names. Pinning the
        source keeps a partial fix from passing as a complete one.
        """
        import inspect

        from custom_components.frigate_vision import media as media_module

        source = inspect.getsource(media_module.MediaManager._async_build_locked)
        for role in ("first", "last", "postroll"):
            assert f'by_role["{role}"]' not in source, (
                f'by_role["{role}"] is indexed directly again; a missing role '
                "raises KeyError, which runtime.py does not catch"
            )

    async def test_a_build_never_escapes_with_a_key_error(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Whole-build regression: any failure is a MediaError, never a KeyError.

        The image_change branch is reached when the event carries no usable
        `path_data`. This drives it end to end and asserts the *class* of any
        failure, so a future bare lookup cannot slip through as an unclassified
        exception.
        """
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record("review_entry_1_front_review_roles")
        await store.async_create(record)

        class Client:
            async def async_get_event(self, event_id: str, camera: str):
                # No path_data -> plan_evidence returns the image_change plan.
                return {
                    "id": event_id,
                    "camera": camera,
                    "label": "person",
                    "start_time": 103,
                    "end_time": 117,
                }

            async def async_get_recordings(
                self, camera: str, after: float, before: float
            ):
                return [{"start_time": after - 10, "end_time": before + 10}]

            async def async_get_snapshot(
                self, camera: str, timestamp: float, height: int
            ):
                return _changing_jpeg(timestamp)

        manager = MediaManager(
            hass,
            store,
            Client(),
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )

        try:
            await manager.async_build(record.activity_id)
        except MediaError:
            pass
        except Exception as exc:  # noqa: BLE001 - the defect is the class
            pytest.fail(
                "a non-MediaError escaped async_build; runtime.py catches only "
                f"(OSError, FrigateApiError, MediaError): {type(exc).__name__}: {exc}"
            )
