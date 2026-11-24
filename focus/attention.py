"""Head direction and per-student attention bookkeeping.

Everything here is plain Python so it can be unit-tested without a camera.
"""
from dataclasses import dataclass, field

# Attention states
FOCUSED = "focused"
LOOKING_AWAY = "looking_away"
ABSENT = "absent"


def head_yaw_ratio(right_eye, left_eye, nose):
    """How far the head is turned sideways, from three face keypoints.

    Returns (nose_x - eye_midpoint_x) / eye_distance:
      about 0      -> facing the camera
      large +/-    -> head turned to one side
    Points are (x, y) in any consistent unit. Returns 0.0 if the eyes overlap.
    """
    eye_dist = abs(left_eye[0] - right_eye[0])
    if eye_dist < 1e-6:
        return 0.0
    mid_x = (left_eye[0] + right_eye[0]) / 2.0
    return (nose[0] - mid_x) / eye_dist


@dataclass
class Event:
    student: int
    kind: str        # "looking_away" or "left_seat"
    start: float     # seconds since start of session
    duration: float  # seconds


@dataclass
class StudentRecord:
    """Attention history of one seat/student."""

    student: int
    first_seen: float
    state: str = FOCUSED
    state_since: float = 0.0
    focused_time: float = 0.0
    away_time: float = 0.0
    absent_time: float = 0.0
    flagged: bool = False          # current away/absent streak already reported
    events: list = field(default_factory=list)

    @property
    def tracked_time(self):
        return self.focused_time + self.away_time + self.absent_time

    @property
    def focus_percent(self):
        t = self.tracked_time
        return 100.0 * self.focused_time / t if t > 0 else 100.0


class AttentionMonitor:
    """Turns per-frame observations into focus statistics and events.

    Call update() once per frame with the time and, for every known student,
    their head-yaw ratio (or None if their face was not found this frame).

    - |yaw| > yaw_thresh for at least away_seconds   -> "looking_away" event
    - face missing for at least absent_seconds       -> "left_seat" event
    Looking down (e.g. writing) is NOT counted as looking away.
    """

    def __init__(self, yaw_thresh=0.35, away_seconds=3.0, absent_seconds=10.0):
        self.yaw_thresh = yaw_thresh
        self.away_seconds = away_seconds
        self.absent_seconds = absent_seconds
        self.students = {}
        self.last_t = None

    def classify(self, yaw):
        if yaw is None:
            return ABSENT
        return LOOKING_AWAY if abs(yaw) > self.yaw_thresh else FOCUSED

    def update(self, t, observations):
        """observations: {student_id: yaw_ratio or None}. Returns new events."""
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        new_events = []

        for sid, yaw in observations.items():
            rec = self.students.get(sid)
            if rec is None:
                rec = self.students[sid] = StudentRecord(sid, first_seen=t, state_since=t)
                dt_s = 0.0
            else:
                dt_s = dt

            # time since last frame belongs to the state we were in
            if rec.state == FOCUSED:
                rec.focused_time += dt_s
            elif rec.state == LOOKING_AWAY:
                rec.away_time += dt_s
            else:
                rec.absent_time += dt_s

            new_state = self.classify(yaw)
            if new_state != rec.state:
                self._close_streak(rec, t)
                rec.state = new_state
                rec.state_since = t
                rec.flagged = False

            limit = {LOOKING_AWAY: self.away_seconds, ABSENT: self.absent_seconds}.get(rec.state)
            if limit is not None and not rec.flagged and t - rec.state_since >= limit:
                rec.flagged = True
                kind = "looking_away" if rec.state == LOOKING_AWAY else "left_seat"
                ev = Event(sid, kind, rec.state_since, t - rec.state_since)
                rec.events.append(ev)
                new_events.append(ev)
        return new_events

    def _close_streak(self, rec, t):
        """Fix the final duration of a flagged streak once it ends."""
        if rec.flagged and rec.events:
            rec.events[-1].duration = t - rec.state_since

    def finish(self, t):
        """Close any open streaks at the end of the session."""
        for rec in self.students.values():
            self._close_streak(rec, t)
