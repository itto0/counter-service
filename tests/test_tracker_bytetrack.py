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


def test_bytetrack_crossing_increments_count():
    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
    counter = ChickenCounter(
        line_zone=line,
        tracker_type="bytetrack",
        track_thresh=0.25,
        match_thresh=0.7,
    )

    # Frame 1: Detection above line (y: 100 to 180)
    det1 = [Detection(bbox=(180, 100, 220, 180), confidence=0.9, class_name="chicken")]
    events1 = counter.update(det1, timestamp=1000.0)

    # Frame 2: Same object moving across line (y: 170 to 220)
    det2 = [Detection(bbox=(180, 170, 220, 220), confidence=0.9, class_name="chicken")]
    events2 = counter.update(det2, timestamp=1000.033)

    # Frame 3: Object continues downward (y: 210 to 260)
    det3 = [Detection(bbox=(180, 210, 220, 260), confidence=0.9, class_name="chicken")]
    events3 = counter.update(det3, timestamp=1000.066)

    total_events = events1 + events2 + events3
    assert len(total_events) == 1
    assert total_events[0].line_id == "line-1"
    assert total_events[0].class_name == "chicken"
    assert counter.total_count == 1
