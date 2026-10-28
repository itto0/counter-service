import logging
import pytest
from src.detection.detector import Detection
from src.tracking.tracker import ChickenCounter, VirtualLineZone


def test_bytetrack_initialization():
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(line_zone=line, tracker_type="bytetrack")
    assert counter.tracker_type == "bytetrack"
    assert counter.total_count == 0


def test_bytetrack_centroid_fallback():
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(line_zone=line, tracker_type="centroid")
    assert counter.tracker_type == "centroid"


def test_bytetrack_fallback_when_dependencies_unavailable(monkeypatch, caplog):
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    with monkeypatch.context() as m:
        m.setattr("src.tracking.tracker.HAS_BYTETRACK", False)
        with caplog.at_level(logging.WARNING):
            counter = ChickenCounter(line_zone=line, tracker_type="bytetrack")
            assert counter.tracker_type == "centroid"
            assert counter._bytetrack is None
            assert any(
                "ByteTrack requested but dependencies unavailable; falling back to centroid tracking."
                in record.message
                for record in caplog.records
            )


def test_bytetrack_empty_frame_handling():
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(line_zone=line, tracker_type="bytetrack")

    # Initial empty frame update
    events = counter.update([])
    assert events == []
    assert counter.total_count == 0

    # Feed a detection to create track
    det = [Detection(bbox=(180, 95, 220, 190), confidence=0.9, class_name="chicken")]
    counter.update(det, timestamp=1000.0)
    assert len(counter.tracks) == 1

    # Subsequent empty frame update does not crash and returns empty list
    events_empty = counter.update([], timestamp=1000.033)
    assert events_empty == []


def test_bytetrack_crossing_increments_count():
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(
        line_zone=line,
        tracker_type="bytetrack",
        track_thresh=0.25,
        match_thresh=0.7,
    )

    # Frame 1: Detection above line (anchor bottom center at y=190 < 200)
    det1 = [Detection(bbox=(180, 95, 220, 190), confidence=0.9, class_name="chicken")]
    events1 = counter.update(det1, timestamp=1000.0)
    assert events1 == []
    assert counter.total_count == 0

    # Frame 2: Continuous conveyor movement across line (anchor y=202 >= 200, IoU ~ 0.78 >= match_thresh 0.7)
    det2 = [Detection(bbox=(180, 107, 220, 202), confidence=0.9, class_name="chicken")]
    events2 = counter.update(det2, timestamp=1000.033)
    assert len(events2) == 1
    assert events2[0].line_id == "line-1"
    assert events2[0].class_name == "chicken"
    assert events2[0].confidence == pytest.approx(0.9)
    assert counter.total_count == 1

    # Frame 3: Object continues downward (anchor y=214, IoU ~ 0.78 >= match_thresh 0.7, already counted)
    det3 = [Detection(bbox=(180, 119, 220, 214), confidence=0.9, class_name="chicken")]
    events3 = counter.update(det3, timestamp=1000.066)
    assert events3 == []
    assert counter.total_count == 1

    total_events = events1 + events2 + events3
    assert len(total_events) == 1


def test_bytetrack_multi_object_concurrent_tracking():
    """Verify ByteTrack tracks and counts multiple concurrent objects crossing the line."""
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(
        line_zone=line,
        tracker_type="bytetrack",
        track_thresh=0.25,
        match_thresh=0.7,
    )

    # Frame 1: Two objects above line at different X coordinates
    # Object A at x=100, Object B at x=280; both anchors at y=190 < 200
    frame1_dets = [
        Detection(bbox=(80, 95, 120, 190), confidence=0.9, class_name="chicken"),
        Detection(bbox=(260, 95, 300, 190), confidence=0.9, class_name="chicken"),
    ]
    events1 = counter.update(frame1_dets, timestamp=1000.0)
    assert events1 == []
    assert counter.total_count == 0

    # Frame 2: Both objects cross the line simultaneously (y=202 >= 200)
    frame2_dets = [
        Detection(bbox=(80, 107, 120, 202), confidence=0.9, class_name="chicken"),
        Detection(bbox=(260, 107, 300, 202), confidence=0.9, class_name="chicken"),
    ]
    events2 = counter.update(frame2_dets, timestamp=1000.033)
    assert len(events2) == 2
    track_ids = {ev.track_id for ev in events2}
    assert len(track_ids) == 2
    assert counter.total_count == 2

    # Frame 3: Both objects continue downward, no duplicate counts
    frame3_dets = [
        Detection(bbox=(80, 119, 120, 214), confidence=0.9, class_name="chicken"),
        Detection(bbox=(260, 119, 300, 214), confidence=0.9, class_name="chicken"),
    ]
    events3 = counter.update(frame3_dets, timestamp=1000.066)
    assert events3 == []
    assert counter.total_count == 2


def test_bytetrack_reverse_direction_not_counted():
    """Verify objects moving in the reverse direction across the line are not counted."""
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(
        line_zone=line,
        tracker_type="bytetrack",
        track_thresh=0.25,
        match_thresh=0.7,
    )

    # Frame 1: Object below line (anchor y=210 > 200)
    det1 = [Detection(bbox=(180, 115, 220, 210), confidence=0.9, class_name="chicken")]
    events1 = counter.update(det1, timestamp=1000.0)
    assert events1 == []
    assert counter.total_count == 0

    # Frame 2: Object moves UP across line (anchor y=198 < 200)
    det2 = [Detection(bbox=(180, 103, 220, 198), confidence=0.9, class_name="chicken")]
    events2 = counter.update(det2, timestamp=1000.033)
    assert events2 == []
    assert counter.total_count == 0

    # Frame 3: Object continues upward (anchor y=186 < 200)
    det3 = [Detection(bbox=(180, 91, 220, 186), confidence=0.9, class_name="chicken")]
    events3 = counter.update(det3, timestamp=1000.066)
    assert events3 == []
    assert counter.total_count == 0


def test_bytetrack_completion_accuracy():
    """Verify ByteTrack properly computes completion accuracy when objects cross and expire."""
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(
        line_zone=line,
        tracker_type="bytetrack",
        max_age=2,
        track_thresh=0.25,
        match_thresh=0.7,
    )

    # Frame 1: Detection above line
    det1 = [Detection(bbox=(180, 95, 220, 190), confidence=0.9, class_name="chicken")]
    counter.update(det1, timestamp=1000.0)

    # Frame 2: Crossing
    det2 = [Detection(bbox=(180, 107, 220, 202), confidence=0.9, class_name="chicken")]
    counter.update(det2, timestamp=1000.033)

    assert counter.total_count == 1
    assert counter.get_tracking_accuracy_pct(reset_window=False) == 100.0

