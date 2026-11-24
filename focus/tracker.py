"""Seat-based tracker: gives each student a stable ID across frames.

In an exam hall students stay in their seats, so a student is identified by
where their face is. A face is matched to the nearest known seat; a face far
from every known seat becomes a new seat. Seats are never deleted, so a student
who leaves and comes back keeps the same ID.
"""
import math


class SeatTracker:
    def __init__(self, max_distance=0.12):
        """max_distance: how far (as a fraction of frame width) a face may
        move and still count as the same seat."""
        self.max_distance = max_distance
        self.seats = {}  # id -> (x, y) last known centre, normalised 0..1
        self._next_id = 1

    def update(self, centres):
        """centres: list of (x, y) face centres, normalised to 0..1.

        Returns a list of seat IDs, one per centre, in the same order.
        """
        pairs = []
        for i, (x, y) in enumerate(centres):
            for sid, (sx, sy) in self.seats.items():
                d = math.hypot(x - sx, y - sy)
                if d <= self.max_distance:
                    pairs.append((d, i, sid))
        pairs.sort()

        assigned = [None] * len(centres)
        used = set()
        for d, i, sid in pairs:  # greedy: closest pairs first
            if assigned[i] is None and sid not in used:
                assigned[i] = sid
                used.add(sid)

        for i, c in enumerate(centres):
            if assigned[i] is None:
                assigned[i] = self._next_id
                self._next_id += 1
            self.seats[assigned[i]] = c
        return assigned
