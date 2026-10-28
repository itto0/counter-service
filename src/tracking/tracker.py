"""Object tracking and virtual line crossing counter implementation."""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

try:
    import numpy as np
    import supervision as sv
    try:
        from trackers import ByteTrackTracker
        HAS_BYTETRACK = True
    except ImportError:
        if hasattr(sv, "ByteTrack"):
            ByteTrackTracker = sv.ByteTrack
            HAS_BYTETRACK = True
        else:
            ByteTrackTracker = None
            HAS_BYTETRACK = False
except ImportError:
    HAS_BYTETRACK = False
    ByteTrackTracker = None
    sv = None
    np = None

try:
    from src.detection.detector import Detection
except ImportError:
    from src.detection.detector import Detection

try:
    from src.contracts import CountEventDTO
except ImportError:
    try:
        from src.contracts import CountEventDTO
    except ImportError:
        from shared.contracts import CountEventDTO

logger = logging.getLogger("counter.tracker")

__all__ = ["TrackedObject", "VirtualLineZone", "ChickenCounter"]


@dataclass
class TrackedObject:
    """Represents an active or historical object track."""

    track_id: int
    last_anchor: tuple[float, float]
    current_anchor: tuple[float, float]
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    class_name: str = "chicken"
    confidence: float = 0.0
    missed_frames: int = 0
    age: int = 0
    counted: bool = False


class VirtualLineZone:
    """Virtual line zone defining a boundary and crossing direction."""

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        direction: str = "down",
    ) -> None:
        """Initialize virtual line zone.

        Args:
            start: Starting coordinates (x, y) of the line segment.
            end: Ending coordinates (x, y) of the line segment.
            direction: Crossing direction: "down", "up", "left", "right", "in", or "out".
        """
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.direction = direction.lower()

    def crosses(self, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        """Check whether movement from p1 to p2 crosses the line segment in configured direction.

        Args:
            p1: Previous anchor point (x, y).
            p2: Current anchor point (x, y).

        Returns:
            True if segment (p1 -> p2) intersects (start -> end) in the active direction.
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        dx = x2 - x1
        dy = y2 - y1

        # Check direction of movement
        if self.direction in ("down", "in"):
            if dy <= 0:
                return False
        elif self.direction in ("up", "out"):
            if dy >= 0:
                return False
        elif self.direction == "right":
            if dx <= 0:
                return False
        elif self.direction == "left":
            if dx >= 0:
                return False

        ax, ay = self.start
        bx, by = self.end

        # Line segment vector AB = (bx - ax, by - ay)
        # Movement segment vector CD = (x2 - x1, y2 - y1)
        det = (bx - ax) * dy - (by - ay) * dx
        if abs(det) < 1e-9:
            return False

        t = ((x1 - ax) * dy - (y1 - ay) * dx) / det
        u = ((x1 - ax) * (by - ay) - (y1 - ay) * (bx - ax)) / det

        eps = 1e-6
        # t must lie along segment AB: [0, 1]
        # u must lie along movement segment CD: [0, 1) to count exactly on first cross
        return (-eps <= t <= 1.0 + eps) and (-eps <= u < 1.0)


class _ByteTrackAdapter:
    """Adapter for ByteTrackTracker (trackers) or ByteTrack (supervision)."""

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.4,
        match_thresh: float = 0.7,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        self.minimum_consecutive_frames = minimum_consecutive_frames
        try:
            self._tracker = ByteTrackTracker(
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate,
                track_activation_threshold=track_activation_threshold,
                minimum_iou_threshold=match_thresh,
                minimum_consecutive_frames=minimum_consecutive_frames,
            )
        except TypeError:
            self._tracker = ByteTrackTracker(
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate,
                track_activation_threshold=track_activation_threshold,
                minimum_matching_threshold=match_thresh,
                minimum_consecutive_frames=minimum_consecutive_frames,
            )

    def update(self, detections: sv.Detections, timestamp: float | None = None) -> sv.Detections:
        if hasattr(self._tracker, "update"):
            res = self._tracker.update(detections, timestamp=timestamp)
        elif hasattr(self._tracker, "update_with_detections"):
            res = self._tracker.update_with_detections(detections)
        else:
            res = self._tracker(detections)

        # In trackers >= 2.6.0 with minimum_consecutive_frames == 1,
        # newly spawned tracklets start with tracker_id == -1.
        # Allocate tracker_ids immediately so initial frame anchor is captured.
        if self.minimum_consecutive_frames == 1 and hasattr(self._tracker, "tracks"):
            for track in self._tracker.tracks:
                if getattr(track, "tracker_id", -1) == -1 and hasattr(self._tracker, "_allocate_tracker_id"):
                    track.tracker_id = self._tracker._allocate_tracker_id()
            if hasattr(res, "tracker_id") and res.tracker_id is not None and len(res.tracker_id) > 0:
                tracker_ids = list(res.tracker_id)
                for i in range(len(tracker_ids)):
                    if tracker_ids[i] == -1 and i < len(detections):
                        box = detections.xyxy[i]
                        for track in self._tracker.tracks:
                            if hasattr(track, "get_state_bbox") and np.allclose(track.get_state_bbox(), box, atol=1e-3):
                                tracker_ids[i] = track.tracker_id
                                break
                res.tracker_id = np.array(tracker_ids, dtype=int)
        return res


class ChickenCounter:
    """Associates detections into tracks, detects line crossings, and emits count events."""

    def __init__(
        self,
        line_zone: VirtualLineZone,
        line_id: str = "line-1",
        max_age: int = 30,
        max_distance: float = 60.0,
        tracker_type: str = "centroid",
        track_thresh: float = 0.4,
        match_thresh: float = 0.7,
    ) -> None:
        """Initialize counter with a virtual line zone and tracking thresholds.

        Args:
            line_zone: Virtual line zone for crossing detection.
            line_id: Identifier of the monitored line.
            max_age: Maximum consecutive missed frames before track eviction.
            max_distance: Maximum Euclidean distance in pixels to match detection to track.
            tracker_type: Tracking algorithm: "bytetrack" or "centroid".
            track_thresh: Detection confidence threshold for track activation.
            match_thresh: Matching threshold for track association.
        """
        self.line_zone = line_zone
        self.line_id = line_id
        self.max_age = max_age
        self.max_distance = max_distance
        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.total_count = 0
        self.tracks: dict[int, TrackedObject] = {}
        self.counted_track_ids: set[int] = set()
        self._next_track_id = 1
        self._window_completed_tracks: int = 0
        self._window_lost_tracks: int = 0
        self._total_completed_tracks: int = 0
        self._total_lost_tracks: int = 0
        self._min_track_hits: int = 2

        if tracker_type.lower() == "bytetrack":
            if HAS_BYTETRACK and ByteTrackTracker is not None:
                self.tracker_type = "bytetrack"
                self._bytetrack = _ByteTrackAdapter(
                    lost_track_buffer=max_age,
                    frame_rate=30.0,
                    track_activation_threshold=track_thresh,
                    match_thresh=match_thresh,
                    minimum_consecutive_frames=1,
                )
            else:
                logger.warning(
                    "ByteTrack requested but dependencies unavailable; falling back to centroid tracking."
                )
                self.tracker_type = "centroid"
                self._bytetrack = None
        else:
            self.tracker_type = "centroid"
            self._bytetrack = None

    def update(
        self,
        detections: list[Detection],
        timestamp: float | None = None,
    ) -> list[CountEventDTO]:
        """Update tracker with frame detections, check crossings, and return count events.

        Args:
            detections: List of Detection objects in current frame.
            timestamp: Epoch or frame timestamp float in seconds, or None for current UTC time.

        Returns:
            List of CountEventDTO emitted during this frame update.
        """
        if timestamp is not None:
            try:
                event_ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                event_ts = datetime.now(timezone.utc)
        else:
            event_ts = datetime.now(timezone.utc)

        if self.tracker_type == "bytetrack" and self._bytetrack is not None and sv is not None and np is not None:
            return self._update_bytetrack(detections, timestamp=timestamp, event_ts=event_ts)
        else:
            return self._update_centroid(detections, event_ts=event_ts)

    def _update_bytetrack(
        self,
        detections: list[Detection],
        timestamp: float | None,
        event_ts: datetime,
    ) -> list[CountEventDTO]:
        events: list[CountEventDTO] = []
        if len(detections) == 0:
            sv_dets = sv.Detections.empty()
        else:
            boxes = []
            confidences = []
            class_names = []
            for d in detections:
                if hasattr(d, "bbox") and d.bbox is not None:
                    boxes.append(list(d.bbox))
                else:
                    boxes.append([float(d.x1), float(d.y1), float(d.x2), float(d.y2)])
                confidences.append(float(d.confidence))
                class_names.append(getattr(d, "class_name", "chicken"))
            xyxy = np.array(boxes, dtype=np.float32)
            conf = np.array(confidences, dtype=np.float32)
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                data={"class_name": np.array(class_names)},
            )

        res = self._bytetrack.update(sv_dets, timestamp=timestamp)

        active_track_ids: set[int] = set()
        if hasattr(res, "tracker_id") and res.tracker_id is not None and len(res.tracker_id) > 0:
            for i, tid in enumerate(res.tracker_id):
                if tid is None or tid < 0:
                    continue
                # Use 1-based track_id (+1) for consistent positive IDs matching TrackedObject contracts
                track_id = int(tid) + 1
                active_track_ids.add(track_id)
                x1, y1, x2, y2 = res.xyxy[i]
                anchor = ((float(x1) + float(x2)) / 2.0, float(y2))
                confidence = (
                    float(res.confidence[i])
                    if (res.confidence is not None and len(res.confidence) > i)
                    else 1.0
                )
                class_name = "chicken"
                if (
                    hasattr(res, "data")
                    and isinstance(res.data, dict)
                    and "class_name" in res.data
                    and res.data["class_name"] is not None
                    and len(res.data["class_name"]) > i
                ):
                    class_name = str(res.data["class_name"][i])
                elif i < len(detections):
                    class_name = getattr(detections[i], "class_name", "chicken")

                if track_id in self.tracks:
                    track = self.tracks[track_id]
                    track.last_anchor = track.current_anchor
                    track.current_anchor = anchor
                    track.bbox = (float(x1), float(y1), float(x2), float(y2))
                    track.confidence = confidence
                    track.missed_frames = 0
                    track.age += 1
                else:
                    track = TrackedObject(
                        track_id=track_id,
                        last_anchor=anchor,
                        current_anchor=anchor,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        class_name=class_name,
                        confidence=confidence,
                        missed_frames=0,
                        age=1,
                        counted=False,
                    )
                    self.tracks[track_id] = track

                if track_id not in self.counted_track_ids and not track.counted:
                    if self.line_zone.crosses(track.last_anchor, track.current_anchor):
                        track.counted = True
                        self.counted_track_ids.add(track_id)
                        self.total_count += 1
                        self._window_completed_tracks += 1
                        self._total_completed_tracks += 1
                        events.append(
                            CountEventDTO(
                                track_id=track_id,
                                ts_utc=event_ts,
                                confidence=confidence,
                                line_id=self.line_id,
                                class_name=track.class_name,
                            )
                        )

        # Increment missed frames for unmatched tracks and prune stale ones
        stale_ids: list[int] = []
        for track_id, track in self.tracks.items():
            if track_id not in active_track_ids:
                track.missed_frames += 1
                max_allowed_missed = min(self.max_age, 3) if track.counted else self.max_age
                if track.missed_frames > max_allowed_missed:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            track = self.tracks[track_id]
            if not track.counted and track.age >= self._min_track_hits:
                self._window_lost_tracks += 1
                self._total_lost_tracks += 1
            del self.tracks[track_id]

        return events

    def _update_centroid(
        self,
        detections: list[Detection],
        event_ts: datetime,
    ) -> list[CountEventDTO]:
        events: list[CountEventDTO] = []

        # Find closest track-detection pairs within max_distance
        candidates: list[tuple[int, float, int, int]] = []
        for track_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                anchor = det.anchor_bottom_center
                dist = (
                    (track.current_anchor[0] - anchor[0]) ** 2
                    + (track.current_anchor[1] - anchor[1]) ** 2
                ) ** 0.5
                if dist <= self.max_distance:
                    # Prioritize active tracks (missed_frames == 0) over stale tracks (missed_frames > 0)
                    candidates.append((track.missed_frames, dist, track_id, det_idx))

        candidates.sort(key=lambda item: (item[0], item[1]))

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        matches: list[tuple[int, int]] = []

        for _, dist, track_id, det_idx in candidates:
            if track_id not in matched_track_ids and det_idx not in matched_det_indices:
                matched_track_ids.add(track_id)
                matched_det_indices.add(det_idx)
                matches.append((track_id, det_idx))

        # Update matched tracks and check crossing
        for track_id, det_idx in matches:
            track = self.tracks[track_id]
            det = detections[det_idx]
            anchor = det.anchor_bottom_center

            track.last_anchor = track.current_anchor
            track.current_anchor = anchor
            track.bbox = (float(det.x1), float(det.y1), float(det.x2), float(det.y2))
            track.class_name = det.class_name
            track.confidence = det.confidence
            track.missed_frames = 0
            track.age += 1

            if track_id not in self.counted_track_ids and not track.counted:
                if self.line_zone.crosses(track.last_anchor, track.current_anchor):
                    track.counted = True
                    self.counted_track_ids.add(track_id)
                    self.total_count += 1
                    self._window_completed_tracks += 1
                    self._total_completed_tracks += 1
                    events.append(
                        CountEventDTO(
                            track_id=track_id,
                            ts_utc=event_ts,
                            confidence=det.confidence,
                            line_id=self.line_id,
                            class_name=det.class_name,
                        )
                    )

        # Register unmatched detections as new tracks
        newly_created_ids: set[int] = set()
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                new_id = self._next_track_id
                self._next_track_id += 1
                anchor = det.anchor_bottom_center
                self.tracks[new_id] = TrackedObject(
                    track_id=new_id,
                    last_anchor=anchor,
                    current_anchor=anchor,
                    bbox=(float(det.x1), float(det.y1), float(det.x2), float(det.y2)),
                    class_name=det.class_name,
                    confidence=det.confidence,
                    missed_frames=0,
                    age=1,
                    counted=False,
                )
                newly_created_ids.add(new_id)

        # Increment missed frames for unmatched tracks and prune stale ones
        stale_ids: list[int] = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids and track_id not in newly_created_ids:
                track.missed_frames += 1
                max_allowed_missed = min(self.max_age, 3) if track.counted else self.max_age
                if track.missed_frames > max_allowed_missed:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            track = self.tracks[track_id]
            if not track.counted and track.age >= self._min_track_hits:
                self._window_lost_tracks += 1
                self._total_lost_tracks += 1
            del self.tracks[track_id]

        return events

    def get_tracking_accuracy_pct(self, reset_window: bool = True) -> float | None:
        """Calculate tracking completion accuracy percentage for the current window.

        Returns:
            Accuracy percentage in range [0.0, 100.0] or None if no tracks finished in window.
        """
        total = self._window_completed_tracks + self._window_lost_tracks
        if total == 0:
            return None
        acc = (self._window_completed_tracks / total) * 100.0
        if reset_window:
            self._window_completed_tracks = 0
            self._window_lost_tracks = 0
        return round(min(100.0, max(0.0, acc)), 2)
