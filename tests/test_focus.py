"""Unit tests for the tracking and attention logic (no camera or model needed).

Run with:  python -m pytest tests
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from focus.attention import ABSENT, FOCUSED, LOOKING_AWAY, AttentionMonitor, head_yaw_ratio  # noqa: E402
from focus.report import write_report  # noqa: E402
from focus.tracker import SeatTracker  # noqa: E402


# ---------------------------------------------------------------- head yaw
def test_yaw_is_zero_when_facing_camera():
    assert head_yaw_ratio((40, 50), (60, 50), (50, 60)) == 0


def test_yaw_grows_when_head_turns():
    small = head_yaw_ratio((40, 50), (60, 50), (52, 60))
    big = head_yaw_ratio((40, 50), (60, 50), (62, 60))
    assert abs(small) < 0.35 < abs(big)
    assert head_yaw_ratio((40, 50), (60, 50), (38, 60)) < -0.35  # other direction


def test_yaw_handles_overlapping_eyes():
    assert head_yaw_ratio((50, 50), (50, 50), (60, 60)) == 0.0


# ----------------------------------------------------------------- tracker
def test_seats_keep_ids_when_faces_move_slightly():
    t = SeatTracker(max_distance=0.1)
    assert t.update([(0.2, 0.5), (0.7, 0.5)]) == [1, 2]
    assert t.update([(0.72, 0.51), (0.21, 0.49)]) == [2, 1]  # order swapped


def test_student_returning_to_seat_keeps_id():
    t = SeatTracker(max_distance=0.1)
    t.update([(0.2, 0.5), (0.7, 0.5)])
    t.update([(0.2, 0.5)])                       # seat 2 empty
    assert t.update([(0.2, 0.5), (0.69, 0.5)]) == [1, 2]


def test_far_face_becomes_new_seat():
    t = SeatTracker(max_distance=0.1)
    t.update([(0.2, 0.5)])
    assert t.update([(0.2, 0.5), (0.9, 0.9)]) == [1, 2]


def test_two_faces_cannot_take_the_same_seat():
    t = SeatTracker(max_distance=0.2)
    t.update([(0.5, 0.5)])
    ids = t.update([(0.52, 0.5), (0.55, 0.5)])
    assert ids[0] == 1 and ids[1] != 1


# --------------------------------------------------------------- attention
def run(monitor, sid, yaws, fps=10):
    """Feed a sequence of yaw values (None = absent) at the given fps."""
    events = []
    for i, y in enumerate(yaws):
        events += monitor.update(i / fps, {sid: y})
    return events


def test_classify():
    m = AttentionMonitor(yaw_thresh=0.35)
    assert m.classify(0.1) == FOCUSED
    assert m.classify(-0.5) == LOOKING_AWAY
    assert m.classify(None) == ABSENT


def test_short_glance_is_not_flagged():
    m = AttentionMonitor(away_seconds=3)
    events = run(m, 1, [0.0] * 10 + [0.6] * 15 + [0.0] * 10)  # 1.5 s glance
    assert events == []
    assert m.students[1].away_time > 0


def test_long_look_away_is_flagged_once_with_final_duration():
    m = AttentionMonitor(away_seconds=3)
    events = run(m, 1, [0.0] * 10 + [0.6] * 50 + [0.0] * 10)  # 5 s away
    assert [e.kind for e in events] == ["looking_away"]
    assert len(m.students[1].events) == 1
    assert abs(m.students[1].events[0].duration - 5.0) < 0.2  # closed when they looked back


def test_leaving_seat_is_flagged():
    m = AttentionMonitor(absent_seconds=10)
    events = run(m, 1, [0.0] * 10 + [None] * 120)  # 12 s missing
    assert [e.kind for e in events] == ["left_seat"]


def test_focus_percent():
    m = AttentionMonitor()
    run(m, 1, [0.0] * 31 + [0.6] * 10)  # 3 s focused, ~1 s away
    assert 70 < m.students[1].focus_percent < 80


def test_report_files(tmp_path):
    m = AttentionMonitor(away_seconds=1)
    run(m, 1, [0.0] * 5 + [0.8] * 20)
    m.finish(2.5)
    students_csv, events_csv = write_report(m, str(tmp_path))
    rows = list(csv.DictReader(open(students_csv, encoding="utf-8")))
    assert rows[0]["student"] == "S1" and rows[0]["looking_away_events"] == "1"
    events = list(csv.DictReader(open(events_csv, encoding="utf-8")))
    assert events[0]["event"] == "looking_away"


# ------------------------------------------------------------ tiling / NMS
from focus.boxes import iou, non_max_suppression, tile_grid  # noqa: E402


def test_tiles_cover_whole_frame_with_overlap():
    tiles = tile_grid(600, 300, rows=2, cols=3, overlap=0.2)
    assert len(tiles) == 6
    assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 600
    assert min(t[1] for t in tiles) == 0 and max(t[3] for t in tiles) == 300
    assert tiles[0][2] > tiles[1][0]  # neighbours overlap


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - 50 / 150) < 1e-9


def test_nms_merges_duplicates_from_overlapping_tiles():
    faces = [{"box": (100, 100, 20, 20), "score": 0.7},
             {"box": (102, 101, 20, 20), "score": 0.9},   # same face, other tile
             {"box": (300, 100, 20, 20), "score": 0.6}]   # different face
    kept = non_max_suppression(faces)
    assert len(kept) == 2
    assert kept[0]["score"] == 0.9
