# Review Package: Task 1

**Base:** 0874c2bdceb2ececc4fbcd17561f7a1f463eede6
**Head:** 65cf9c9d1c2955137d48b4eaa7bd4562019380a3

## Commits
65cf9c9 feat(tracker): implement tracking completion accuracy calculation

## Stat Summary
 src/tracking/tracker.py         | 34 +++++++++++++++++++++++++++++++---
 tests/test_tracker_bytetrack.py | 24 ++++++++++++++++++++++++
 tests/test_tracking.py          | 75 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 3 files changed, 130 insertions(+), 3 deletions(-)

## Full Diff
```diff
diff --git a/src/tracking/tracker.py b/src/tracking/tracker.py
index 272da0c..6630e2c 100644
--- a/src/tracking/tracker.py
+++ b/src/tracking/tracker.py
@@ -215,6 +215,11 @@ class ChickenCounter:
         self.tracks: dict[int, TrackedObject] = {}
         self.counted_track_ids: set[int] = set()
         self._next_track_id = 1
+        self._window_completed_tracks: int = 0
+        self._window_lost_tracks: int = 0
+        self._total_completed_tracks: int = 0
+        self._total_lost_tracks: int = 0
+        self._min_track_hits: int = 2
 
         if tracker_type.lower() == "bytetrack":
             if HAS_BYTETRACK and ByteTrackTracker is not None:
@@ -337,7 +342,7 @@ class ChickenCounter:
                         class_name=class_name,
                         confidence=confidence,
                         missed_frames=0,
-                        age=0,
+                        age=1,
                         counted=False,
                     )
                     self.tracks[track_id] = track
@@ -347,6 +352,8 @@ class ChickenCounter:
                         track.counted = True
                         self.counted_track_ids.add(track_id)
                         self.total_count += 1
+                        self._window_completed_tracks += 1
+                        self._total_completed_tracks += 1
                         events.append(
                             CountEventDTO(
                                 track_id=track_id,
@@ -362,12 +369,15 @@ class ChickenCounter:
         for track_id, track in self.tracks.items():
             if track_id not in active_track_ids:
                 track.missed_frames += 1
-                track.age += 1
                 max_allowed_missed = min(self.max_age, 3) if track.counted else self.max_age
                 if track.missed_frames > max_allowed_missed:
                     stale_ids.append(track_id)
 
         for track_id in stale_ids:
+            track = self.tracks[track_id]
+            if not track.counted and track.age >= self._min_track_hits:
+                self._window_lost_tracks += 1
+                self._total_lost_tracks += 1
             del self.tracks[track_id]
 
         return events
@@ -423,6 +433,8 @@ class ChickenCounter:
                     track.counted = True
                     self.counted_track_ids.add(track_id)
                     self.total_count += 1
+                    self._window_completed_tracks += 1
+                    self._total_completed_tracks += 1
                     events.append(
                         CountEventDTO(
                             track_id=track_id,
@@ -448,7 +460,7 @@ class ChickenCounter:
                     class_name=det.class_name,
                     confidence=det.confidence,
                     missed_frames=0,
-                    age=0,
+                    age=1,
                     counted=False,
                 )
                 newly_created_ids.add(new_id)
@@ -458,12 +470,30 @@ class ChickenCounter:
         for track_id, track in self.tracks.items():
             if track_id not in matched_track_ids and track_id not in newly_created_ids:
                 track.missed_frames += 1
-                track.age += 1
                 max_allowed_missed = min(self.max_age, 3) if track.counted else self.max_age
                 if track.missed_frames > max_allowed_missed:
                     stale_ids.append(track_id)
 
         for track_id in stale_ids:
+            track = self.tracks[track_id]
+            if not track.counted and track.age >= self._min_track_hits:
+                self._window_lost_tracks += 1
+                self._total_lost_tracks += 1
             del self.tracks[track_id]
 
         return events
+
+    def get_tracking_accuracy_pct(self, reset_window: bool = True) -> float | None:
+        """Calculate tracking completion accuracy percentage for the current window.
+
+        Returns:
+            Accuracy percentage in range [0.0, 100.0] or None if no tracks finished in window.
+        """
+        total = self._window_completed_tracks + self._window_lost_tracks
+        if total == 0:
+            return None
+        acc = (self._window_completed_tracks / total) * 100.0
+        if reset_window:
+            self._window_completed_tracks = 0
+            self._window_lost_tracks = 0
+        return round(min(100.0, max(0.0, acc)), 2)
diff --git a/tests/test_tracker_bytetrack.py b/tests/test_tracker_bytetrack.py
index 7191d6b..5ed229b 100644
--- a/tests/test_tracker_bytetrack.py
+++ b/tests/test_tracker_bytetrack.py
@@ -153,3 +153,27 @@ def test_bytetrack_reverse_direction_not_counted():
     events3 = counter.update(det3, timestamp=1000.066)
     assert events3 == []
     assert counter.total_count == 0
+
+
+def test_bytetrack_completion_accuracy():
+    """Verify ByteTrack properly computes completion accuracy when objects cross and expire."""
+    line = VirtualLineZone(start=(0, 200), end=(400, 200), direction="down")
+    counter = ChickenCounter(
+        line_zone=line,
+        tracker_type="bytetrack",
+        max_age=2,
+        track_thresh=0.25,
+        match_thresh=0.7,
+    )
+
+    # Frame 1: Detection above line
+    det1 = [Detection(bbox=(180, 95, 220, 190), confidence=0.9, class_name="chicken")]
+    counter.update(det1, timestamp=1000.0)
+
+    # Frame 2: Crossing
+    det2 = [Detection(bbox=(180, 107, 220, 202), confidence=0.9, class_name="chicken")]
+    counter.update(det2, timestamp=1000.033)
+
+    assert counter.total_count == 1
+    assert counter.get_tracking_accuracy_pct(reset_window=False) == 100.0
+
diff --git a/tests/test_tracking.py b/tests/test_tracking.py
index 5a3a34d..cbe8ef0 100644
--- a/tests/test_tracking.py
+++ b/tests/test_tracking.py
@@ -646,3 +646,78 @@ def test_active_track_priority_prevents_stale_track_theft():
     assert counter.tracks[2].counted is True
 
 
+def test_tracker_completion_accuracy_healthy():
+    """Verify 100% accuracy when all valid tracks cross the line."""
+    line = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
+    counter = ChickenCounter(line_zone=line)
+
+    # Track 1 crosses
+    det1_a = Detection(x1=100, y1=150, x2=140, y2=190, confidence=0.9, class_id=0, class_name="chicken")
+    counter.update([det1_a])
+    det1_b = Detection(x1=100, y1=170, x2=140, y2=215, confidence=0.9, class_id=0, class_name="chicken")
+    counter.update([det1_b])
+
+    assert counter.total_count == 1
+    acc = counter.get_tracking_accuracy_pct(reset_window=False)
+    assert acc == 100.0
+
+
+def test_tracker_completion_accuracy_with_lost_tracks():
+    """Verify accuracy drops when tracks die before crossing."""
+    line = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
+    counter = ChickenCounter(line_zone=line, max_age=2)
+
+    # Track 1 crosses
+    det1_a = Detection(x1=100, y1=150, x2=140, y2=190, confidence=0.9, class_id=0, class_name="chicken")
+    counter.update([det1_a])
+    det1_b = Detection(x1=100, y1=170, x2=140, y2=215, confidence=0.9, class_id=0, class_name="chicken")
+    counter.update([det1_b])
+
+    # Track 2 appears for 2 frames then disappears without crossing
+    det2_a = Detection(x1=300, y1=50, x2=340, y2=90, confidence=0.9, class_id=0, class_name="chicken")
+    counter.update([det2_a])
+    det2_b = Detection(x1=300, y1=60, x2=340, y2=100, confidence=0.9, class_id=0, class_name="chicken")
+    counter.update([det2_b])
+
+    # Advance 4 empty frames to prune Track 2 (max_age=2)
+    for _ in range(4):
+        counter.update([])
+
+    # 1 completed, 1 lost -> 50.0%
+    acc = counter.get_tracking_accuracy_pct(reset_window=True)
+    assert acc == 50.0
+
+
+def test_tracker_completion_accuracy_ignores_single_frame_noise():
+    """Verify single-frame transient detections do not count as lost carcass tracks."""
+    line = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
+    counter = ChickenCounter(line_zone=line, max_age=2)
+
+    # Track 1 crosses
+    det1_a = Detection(x1=100, y1=150, x2=140, y2=190, confidence=0.9, class_id=0, class_name="chicken")
+    counter.update([det1_a])
+    det1_b = Detection(x1=100, y1=170, x2=140, y2=215, confidence=0.9, class_id=0, class_name="chicken")
+    counter.update([det1_b])
+
+    # Spurious 1-frame noise detection
+    noise = Detection(x1=500, y1=50, x2=510, y2=60, confidence=0.5, class_id=0, class_name="chicken")
+    counter.update([noise])
+
+    # Advance 4 empty frames to prune noise
+    for _ in range(4):
+        counter.update([])
+
+    # 1 completed, noise ignored -> 100.0%
+    acc = counter.get_tracking_accuracy_pct(reset_window=True)
+    assert acc == 100.0
+
+
+def test_tracker_completion_accuracy_idle_returns_none():
+    """Verify get_tracking_accuracy_pct returns None when no tracks terminate in the window."""
+    line = VirtualLineZone(start=(0, 200), end=(640, 200), direction="down")
+    counter = ChickenCounter(line_zone=line)
+
+    assert counter.get_tracking_accuracy_pct() is None
+
+
```
