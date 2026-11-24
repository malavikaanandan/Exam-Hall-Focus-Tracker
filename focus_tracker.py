"""Exam Hall Focus Tracker

Watches students through a camera (or a recorded video), gives each seat a
stable ID, and measures how much of the time each student faces their own
paper/screen. It flags long look-aways and students leaving their seat, so an
invigilator can review those moments. It never stores face images.

    python focus_tracker.py                      # webcam
    python focus_tracker.py --source exam.mp4    # recorded video
    python focus_tracker.py --help               # all options
"""
import argparse
import os
import sys
import time
import urllib.request
from datetime import datetime

import cv2

from focus.boxes import non_max_suppression, tile_grid
from focus.attention import ABSENT, FOCUSED, LOOKING_AWAY, AttentionMonitor, head_yaw_ratio
from focus.report import fmt_time, summary_text, write_report
from focus.tracker import SeatTracker

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
DEFAULT_MODEL = os.path.join(HERE, "models", "blaze_face_short_range.tflite")

COLOURS = {FOCUSED: (0, 200, 0), LOOKING_AWAY: (0, 165, 255), ABSENT: (128, 128, 128)}
RED = (0, 0, 255)


# ----------------------------------------------------------------- detector
class FaceDetector:
    """Wrapper around MediaPipe's Tasks FaceDetector, with optional tiling.

    With tiles=(rows, cols), the frame is cut into overlapping tiles that are
    enlarged by `upscale` before detection, so small, distant faces are found
    too (e.g. a ceiling CCTV camera). Duplicates from overlapping tiles are
    merged with non-maximum suppression.
    """

    def __init__(self, model_path, min_confidence, tiles=(1, 1), upscale=1.0):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.tiles = tiles
        self.upscale = upscale
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,  # stateless: allows several calls per frame
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def _detect_region(self, region_bgr, offset_x, offset_y, scale):
        """Detect faces in one region; map results back to full-frame pixels."""
        if scale != 1.0:
            region_bgr = cv2.resize(region_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = region_bgr.shape[:2]
        rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))

        faces = []
        for d in result.detections:
            bb = d.bounding_box
            box = (int(offset_x + bb.origin_x / scale), int(offset_y + bb.origin_y / scale),
                   int(bb.width / scale), int(bb.height / scale))
            kps = [(offset_x + k.x * w / scale, offset_y + k.y * h / scale) for k in (d.keypoints or [])]
            score = d.categories[0].score if d.categories else 0.0
            faces.append({"box": box, "keypoints": kps, "score": score})
        return faces

    def detect(self, frame_bgr):
        """Returns faces as dicts: box=(x, y, w, h) px, keypoints=[(x, y) px], score."""
        rows, cols = self.tiles
        if (rows, cols) == (1, 1):
            return self._detect_region(frame_bgr, 0, 0, self.upscale)
        h, w = frame_bgr.shape[:2]
        faces = []
        for x0, y0, x1, y1 in tile_grid(w, h, rows, cols):
            faces += self._detect_region(frame_bgr[y0:y1, x0:x1], x0, y0, self.upscale)
        return non_max_suppression(faces)

    def close(self):
        self._detector.close()


def ensure_model(path):
    if os.path.exists(path):
        return
    if path != DEFAULT_MODEL:
        sys.exit(f"Model not found: {path}")
    print(f"Downloading face detection model (~230 KB) to {path} ...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:
        sys.exit(f"Could not download the model ({e}).\n"
                 f"Download it manually from {MODEL_URL}\nand save it as {path}")


# -------------------------------------------------------------------- draw
def blur_box(frame, box):
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        k = max(15, (min(x1 - x0, y1 - y0) // 3) | 1)
        frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (k, k), 0)


def draw(frame, seen, monitor, t, blur):
    """seen: {student_id: face dict} for faces found in this frame."""
    for sid, face in seen.items():
        rec = monitor.students[sid]
        if blur:
            blur_box(frame, face["box"])
        colour = RED if rec.flagged else COLOURS[rec.state]
        x, y, w, h = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        label = f"S{sid} {rec.focus_percent:.0f}%"
        if rec.state == LOOKING_AWAY:
            label += " away"
        cv2.putText(frame, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    absent = [f"S{s}" for s, r in monitor.students.items() if r.state == ABSENT]
    flagged = sum(r.flagged for r in monitor.students.values())
    header = f"{fmt_time(t)}  Students: {len(monitor.students)}  Present: {len(seen)}  Flagged now: {flagged}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(frame, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if absent:
        cv2.putText(frame, "Away from seat: " + ", ".join(absent), (8, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)


# -------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Track student focus in an exam hall from a camera or video.")
    p.add_argument("--source", default="0", help="webcam index (e.g. 0) or path to a video file")
    p.add_argument("--model", default=DEFAULT_MODEL, help="MediaPipe face detector .tflite model")
    p.add_argument("--min-confidence", type=float, default=0.5, help="minimum face detection confidence")
    p.add_argument("--tiles", default="1x1",
                   help="split each frame into ROWSxCOLS overlapping tiles to find small, distant "
                        "faces, e.g. 2x3 for a ceiling CCTV camera (slower)")
    p.add_argument("--upscale", type=float, default=1.0,
                   help="enlarge each tile (or the frame) by this factor before detection, e.g. 3")
    p.add_argument("--yaw-thresh", type=float, default=0.35,
                   help="head-turn ratio above which a student counts as looking away")
    p.add_argument("--away-seconds", type=float, default=3.0,
                   help="seconds of looking away before it is flagged")
    p.add_argument("--absent-seconds", type=float, default=10.0,
                   help="seconds out of seat before it is flagged")
    p.add_argument("--seat-distance", type=float, default=0.12,
                   help="max face movement (fraction of frame width) to stay the same seat")
    p.add_argument("--report-dir", default=None,
                   help="folder for students.csv and events.csv (default: reports/session_<time>)")
    p.add_argument("--output", default=None, help="save the annotated video to this .mp4 file")
    p.add_argument("--blur", action="store_true", help="blur faces in the display and saved video")
    p.add_argument("--no-display", action="store_true", help="don't open a window (for processing videos)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_model(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {args.source}")
    is_file = not isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    try:
        rows, cols = (int(n) for n in args.tiles.lower().split("x"))
    except ValueError:
        sys.exit(f"--tiles must look like 2x3, got {args.tiles!r}")
    detector = FaceDetector(args.model, args.min_confidence, (rows, cols), args.upscale)
    tracker = SeatTracker(args.seat_distance)
    monitor = AttentionMonitor(args.yaw_thresh, args.away_seconds, args.absent_seconds)
    writer = None
    report_dir = args.report_dir or os.path.join(
        "reports", "session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    print("Focus tracking started." + ("" if args.no_display else " Press 'q' to quit."))
    start, frame_no, t = time.time(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # video files use their own timeline; webcams use the real clock
            t = frame_no / fps if is_file else time.time() - start
            frame_no += 1
            h, w = frame.shape[:2]

            faces = [f for f in detector.detect(frame) if len(f["keypoints"]) >= 3]
            centres = [((f["box"][0] + f["box"][2] / 2) / w, (f["box"][1] + f["box"][3] / 2) / h)
                       for f in faces]
            ids = tracker.update(centres)
            seen = dict(zip(ids, faces))

            obs = {sid: None for sid in tracker.seats}  # everyone not seen is absent
            for sid, f in seen.items():
                kp = f["keypoints"]
                obs[sid] = head_yaw_ratio(kp[0], kp[1], kp[2])
            for ev in monitor.update(t, obs):
                what = "looking away" if ev.kind == "looking_away" else "left their seat"
                print(f"[{fmt_time(t)}] S{ev.student} {what} (since {fmt_time(ev.start)})")

            draw(frame, seen, monitor, t, args.blur)

            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Exam Hall Focus Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        monitor.finish(t)
        cap.release()
        detector.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(summary_text(monitor))
    students_csv, events_csv = write_report(monitor, report_dir)
    print(f"\nReport saved to {students_csv} and {events_csv}")


if __name__ == "__main__":
    main()
