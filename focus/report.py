"""Write the end-of-session report as CSV files and a printed summary."""
import csv
import os


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def write_report(monitor, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    students_path = os.path.join(out_dir, "students.csv")
    events_path = os.path.join(out_dir, "events.csv")

    with open(students_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student", "focus_percent", "focused_s", "looking_away_s",
                    "absent_s", "looking_away_events", "left_seat_events"])
        for sid in sorted(monitor.students):
            r = monitor.students[sid]
            w.writerow([f"S{sid}", f"{r.focus_percent:.1f}", f"{r.focused_time:.1f}",
                        f"{r.away_time:.1f}", f"{r.absent_time:.1f}",
                        sum(e.kind == "looking_away" for e in r.events),
                        sum(e.kind == "left_seat" for e in r.events)])

    with open(events_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student", "event", "start", "duration_s"])
        events = sorted((e for r in monitor.students.values() for e in r.events),
                        key=lambda e: e.start)
        for e in events:
            w.writerow([f"S{e.student}", e.kind, fmt_time(e.start), f"{e.duration:.1f}"])

    return students_path, events_path


def summary_text(monitor):
    lines = ["", "Session summary", "-" * 52,
             f"{'Student':<9}{'Focus':>7}{'Away':>9}{'Absent':>9}{'Flags':>8}"]
    for sid in sorted(monitor.students):
        r = monitor.students[sid]
        lines.append(f"{'S' + str(sid):<9}{r.focus_percent:>6.0f}%{fmt_time(r.away_time):>9}"
                     f"{fmt_time(r.absent_time):>9}{len(r.events):>8}")
    if not monitor.students:
        lines.append("No students were detected.")
    return "\n".join(lines)
