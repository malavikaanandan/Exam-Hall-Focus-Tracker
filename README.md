# Exam Hall Focus Tracker

A computer vision tool that helps invigilators monitor an exam hall. It watches students through a camera (or a recorded video), gives every seat a stable ID, measures how much of the time each student faces their own work, and **flags long look-aways and students leaving their seat** for review. At the end it writes a per-student report.

Built with **MediaPipe** face detection and **OpenCV**.

> **Privacy by design:** no face images are ever saved. The output is only numbers and timestamps. Faces can also be blurred in the live view and saved video with `--blur`.

## Features
- **Multi-student tracking:** each seat gets an ID (S1, S2, ...) that stays the same even if the student leaves and comes back
- **Head-turn detection:** works out whether a student is facing forward or turned sideways, using face keypoints (eyes and nose)
- **Writing is not penalised:** looking down at the paper counts as focused; only sideways head turns count as looking away
- **Smart flagging:**
  - Looking away for **3+ seconds**: flagged
  - Out of seat for **10+ seconds**: flagged
  - Short glances are ignored
- **Live view:** colour-coded boxes and a live focus % per student

  | Colour | Meaning |
  |---|---|
  | 🟩 Green | Focused |
  | 🟧 Orange | Looking away (not flagged yet) |
  | 🟥 Red | Flagged |
- **Session report:** `students.csv` (focus %, time away and absent per student) and `events.csv` (every flagged moment with a timestamp)
- **Works on recordings:** process a recorded exam video without opening a window (`--no-display`) and save an annotated copy (`--output`)
- **Face blurring** for privacy (`--blur`)
- **Unit tests** for all the tracking and attention logic

## How it works

```
Camera / video
     │
     ▼
MediaPipe face detector ──► face box + 6 keypoints (eyes, nose, mouth, ears)
     │
     ├──► SeatTracker ──────► stable student ID per seat
     │
     ├──► Head yaw ratio ───► facing forward or turned sideways
     │
     ▼
AttentionMonitor ──► focus %, flagged events ──► live view + CSV report
```

### 1. Head direction
MediaPipe gives the positions of both eyes and the nose tip. When a student faces forward, the nose sits midway between the eyes. When they turn their head, it shifts to one side:

```
yaw ratio = (nose_x − eye_midpoint_x) / eye_distance
```

- `|yaw| ≤ 0.35`: **focused**
- `|yaw| > 0.35`: **looking away**

Only the horizontal position is used, so looking down to write doesn't count as looking away.

### 2. Seat tracking
Students stay in their seats, so each face is matched to the nearest known seat (closest pairs first, one face per seat). A face far from every known seat becomes a new seat. Seats are never deleted, so when a student comes back they keep the same ID, and the time they were away is recorded.

### 3. Attention timeline
For every frame, each student is **focused**, **looking away** or **absent**. The time is added to that state, and a streak that lasts longer than the limit becomes a flagged event. Its final duration is recorded when the streak ends.

## Project structure
```
focus_tracker.py      # main app: camera/video, detection, drawing, CLI
focus/
  attention.py        # head yaw + AttentionMonitor (focus %, events)
  tracker.py          # SeatTracker (stable student IDs)
  report.py           # CSV report + printed summary
  boxes.py            # tiling + non-max suppression for small faces
tests/test_focus.py   # unit tests (no camera needed)
```

## Getting started

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run
```bash
python focus_tracker.py                          # live webcam
python focus_tracker.py --source exam.mp4        # recorded video
python focus_tracker.py --source exam.mp4 --no-display --output annotated.mp4 --blur
python focus_tracker.py --source cctv.mp4 --tiles 2x3 --upscale 3   # small, distant faces
```
Press **q** to stop. On the first run, the MediaPipe face model (`blaze_face_short_range.tflite`, about 230 KB) is downloaded automatically into `models/`.

When you stop, a summary like this is printed (example numbers):
```
Session summary
----------------------------------------------------
Student    Focus     Away   Absent   Flags
S1           94%    00:21    00:00       1
S2           81%    01:05    00:12       3
```
The CSV files are saved in `reports/session_<date>_<time>/`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--source` | `0` | Webcam index or video file path |
| `--yaw-thresh` | `0.35` | Head-turn ratio above which a student counts as looking away |
| `--away-seconds` | `3` | Seconds of looking away before flagging |
| `--absent-seconds` | `10` | Seconds out of seat before flagging |
| `--seat-distance` | `0.12` | How far a face can move (fraction of frame width) and still be the same seat |
| `--tiles` | `1x1` | Split each frame into ROWSxCOLS overlapping tiles to find small, distant faces (slower) |
| `--upscale` | `1` | Enlarge each tile before detection, e.g. `3` |
| `--min-confidence` | `0.5` | Minimum face detection confidence |
| `--report-dir` | `reports/session_<time>` | Where to save the CSV report |
| `--output` | off | Save the annotated video |
| `--blur` | off | Blur faces |
| `--no-display` | off | Don't open a window |
| `--model` | `models/blaze_face_short_range.tflite` | Face detector model |

## Tests
```bash
pip install pytest
python -m pytest tests
```
16 tests cover head-yaw maths, seat tracking (movement, students returning, conflicts), flagging rules, focus %, the CSV report, and tiling with duplicate removal.

## Tested on

| Footage | Result |
|---|---|
| Webcam recording of one person at a laptop | ✅ Face found in every frame, head-yaw values stayed between -0.14 and 0.05 while facing the camera, and the "left seat" flag fired correctly after 10 s |
| [**Altima IP Camera – Class Room**](https://youtu.be/EPf9O7UXRoM) (CCTV of a real exam hall, 640×360, ceiling camera, about 25 candidates) | ⚠️ Without tiling: **0 faces**. With `--tiles 2x3 --upscale 3`: 3–4 detections per frame, but several are false positives on the backs of heads, and most students in the front rows are missed because they're looking down at their papers. The focus numbers from this video are **not reliable** |

**Takeaway:** the system works with a **front-facing camera close to the students** (a webcam per desk or row, or online exams). A high-angle CCTV camera over a whole hall needs a face detector trained on surveillance footage. See *Future work*.

*Test video: "Altima IP CAMERA Class Room" by [Altima Security Technologies](https://www.youtube.com/@altimaGlobal), used only for local testing. No frames from it are included in this repository.*

## Limitations
- MediaPipe's short-range face model works best within **about 2 m of the camera**, so one camera covers one row or a small group of desks. A large hall needs several cameras. Tiling (`--tiles`) helps a little with distant faces, but it can't find faces that are pointing down.
- Head direction is estimated from 2D keypoints, so it's an approximation. Tune `--yaw-thresh` for your camera angle.
- If two students swap seats, their IDs swap too, because IDs belong to seats, not people.

## Future work
- Swap in a detector trained for CCTV and small faces (for example RetinaFace or YOLO-face) to handle ceiling cameras
- Use head pose in 3D (MediaPipe Face Landmarker) instead of the 2D yaw ratio
- Combine several cameras covering different rows into one report

## Responsible use
This tool **flags moments for a human to review**. It does not decide that anyone cheated. Looking away can be completely innocent: thinking, stretching, or looking at the clock. Anyone using it should:
- tell students that the exam is being monitored,
- follow their institution's rules and local privacy law (for example GDPR),
- always have an invigilator review flags before taking any action.

## Author
**Malavika M A** ([@malavikaanandan](https://github.com/malavikaanandan)), B.Tech in Artificial Intelligence and Data Science

My contributions:
- **Idea and design:** an attention tracker built for exam halls, with seat-based IDs, head-turn detection that doesn't penalise writing, and flagging rules that ignore short glances
- **Implementation:** the seat tracker, the head-yaw ratio, the attention timeline (`AttentionMonitor`), the CSV reports, the live colour-coded view and face blurring
- **Small-face detection:** tiling with enlarged tiles and non-max suppression, added after testing on real CCTV footage
- **Testing and evaluation:** 16 unit tests, plus testing on webcam and real exam hall CCTV footage, with the results and limitations reported honestly
- **Responsible-use guidelines** for deploying it fairly

Built with the help of AI-assisted coding tools.

## Credits
- Face detection: [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) by Google (Apache 2.0), using the BlazeFace short-range model
- Video processing: [OpenCV](https://opencv.org/)
