# Face Recognition Attendance System — Synthetic Dataset

A fully synthetic dataset designed for prototyping and benchmarking **face-recognition-based
attendance systems**. It pairs a labeled face-image gallery with a realistic daily attendance
log, so it can be used for both the *recognition* model and the *downstream attendance analytics*.

> ⚠️ **All faces in this dataset are procedurally generated cartoon avatars — no real
> people, no scraped photos.** This makes the dataset safe to redistribute publicly and
> free of any privacy/consent concerns, while still preserving the folder structure,
> labels, and intra-class variation (pose/lighting jitter) you'd expect from a real
> face-attendance dataset. Swap in your own real photos using the same folder structure
> if you need a production-grade recognition model.

## Contents

```
face_attendance_dataset/
├── images/
│   ├── person_001/
│   │   ├── person_001_01.jpg   # 128x128 px
│   │   ├── person_001_02.jpg
│   │   └── ... (15 images per person)
│   ├── person_002/
│   └── ... (30 identities total, 450 images)
├── metadata/
│   └── persons.csv             # identity roster
├── attendance/
│   ├── attendance_log.csv      # per-day, per-person check-in/out records
│   └── daily_summary.csv       # aggregated daily attendance stats
├── dataset-metadata.json       # Kaggle dataset metadata
├── LICENSE
└── README.md
```

## Files

### `images/person_XXX/*.jpg`
- 30 identities × 15 images = **450 images**, 128×128 RGB JPEG.
- Each identity has fixed traits (skin tone, hair color/style, eye color, glasses).
- Each image applies random jitter (rotation, brightness, contrast, slight blur, small
  positional shift) to simulate the pose/lighting variation you'd see across multiple
  real photos of the same person — useful for training an embedding/recognition model
  with intra-class variation.

### `metadata/persons.csv`
| column | description |
|---|---|
| person_id | unique identity ID, matches the image folder name |
| name | synthetic full name |
| department | department the person belongs to |
| enrollment_date | date the person was enrolled in the system |
| num_images | number of images available for that identity |

### `attendance/attendance_log.csv`
Daily attendance records from **2025-07-01 to 2025-09-30** (weekdays only), generated
per enrolled person from their enrollment date onward.

| column | description |
|---|---|
| date | attendance date (YYYY-MM-DD) |
| person_id | identity ID |
| name | person's name |
| department | department |
| status | `Present`, `Late`, or `Absent` |
| check_in_time | HH:MM:SS, blank if absent |
| check_out_time | HH:MM:SS, blank if absent |
| recognition_confidence | simulated face-match confidence score (0.87–0.999), blank if absent |
| camera_id | which entrance camera logged the event, blank if absent |

### `attendance/daily_summary.csv`
Per-day aggregate counts (present/late/absent/total enrolled) and attendance rate %.

## Suggested uses
- Train/benchmark a face embedding model (e.g. FaceNet/ArcFace-style) on the identity
  folders, then evaluate open-set/closed-set recognition accuracy.
- Build an end-to-end attendance pipeline: face → identity → log entry → dashboard.
- Practice time-series/analytics work on `attendance_log.csv` (absenteeism trends,
  punctuality by department, etc.) independent of the vision task.
- Use the folder structure as a template and drop in your own real photos per person
  if you need a dataset for a production recognition model.

## License
Released under **CC0 1.0 (Public Domain)** — see `LICENSE`. Free to use, modify, and
redistribute for any purpose, including commercial use.
