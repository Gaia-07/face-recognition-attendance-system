# Dataset Card: Face Attendance Dataset (FRAS Benchmark)

## 1. Overview & Provenance
- **Dataset Title**: Synthetic Avatar Attendance Dataset (`face_attendance_dataset`)
- **License**: Creative Commons Zero (CC0) Public Domain Dedication
- **Academic Origin**: Derived for the project *AI-Based Face Recognition Attendance System*, Government Women's Polytechnic, Ranchi (Authored by Bhoomika under guidance of MD Rashid Sir).
- **Domain**: Biometrics, Computer Vision, Metric Learning, Automated Attendance Systems.

---

## 2. Dataset Structure & Schema

The dataset comprises three primary directories:
```
data/raw/face_attendance_dataset/
├── images/                   # 30 identity folders (person_001 to person_030)
│   └── person_001..030/      # 15 cropped images per identity (450 images, 128x128 RGB)
├── metadata/
│   └── persons.csv           # 30 enrolled profiles with department and enrollment dates
└── attendance/
    ├── attendance_log.csv    # 1,980 historical attendance logs across 66 weekdays
    └── daily_summary.csv     # Per-day aggregated attendance rates
```

### Table Schemas & Referential Constraints
| Table | Primary / Foreign Keys | Schema Fields | Verified Rows |
| :--- | :--- | :--- | :--- |
| **`persons.csv`** | PK: `person_id` | `person_id` (str), `name` (str), `department` (str), `enrollment_date` (date), `num_images` (int) | 30 |
| **`attendance_log.csv`** | FK: `person_id` | `date` (date), `person_id` (str), `name` (str), `department` (str), `status` (str), `check_in_time` (time), `check_out_time` (time), `recognition_confidence` (float), `camera_id` (str) | 1,980 |
| **`daily_summary.csv`** | PK: `date` | `date` (date), `present_count` (int), `late_count` (int), `absent_count` (int), `total_enrolled` (int), `attendance_rate_pct` (float) | 66 |

All 1,980 log rows strictly join against `persons.csv` with zero dangling foreign keys.

---

## 3. Verified Exploratory Statistics

All statistics below were verified programmatically by `scripts/validate_data.py`:

### Identity & Demographic Breakdown
The 30 synthetic identities span 8 organizational departments:
- **Sales**: 7 identities (23.3%)
- **Marketing**: 5 identities (16.7%)
- **Engineering**: 4 identities (13.3%)
- **HR**: 3 identities (10.0%)
- **Operations**: 3 identities (10.0%)
- **Design**: 3 identities (10.0%)
- **Finance**: 3 identities (10.0%)
- **Support**: 2 identities (6.7%)

### Attendance Records & Status Distribution
Historical tracking spans July 1, 2025 to September 30, 2025 (66 consecutive weekdays):
- **Present**: 1,587 records (80.15%) — check-in before or at `09:15:00`
- **Late**: 250 records (12.63%) — check-in at or after `09:16:00`
- **Absent**: 143 records (7.22%) — non-appearance

### Hardware Camera Captures
- `LOBBY_CAM_01`: 629 records (31.77%)
- `ENTRANCE_CAM_02`: 605 records (30.56%)
- `ENTRANCE_CAM_01`: 603 records (30.45%)

### Image Properties
- **Total Count**: 450 images (30 identities × 15 images)
- **Spatial Resolution**: $128 \times 128 \times 3$ (RGB)
- **File Format**: Standard JPEG
- **Data Corruption**: 0 corrupt files (100% readable)
- **Mean Luminance (Rec. 601)**: $184.79 \pm 25.24$

---

## 4. Leakage-Free Splitting Protocols

To prevent optimistic bias, deterministic splits are seeded (`seed=42`) and written to `data/splits/`:

### Closed-Set Protocol (`data/splits/closed_set.json`)
Evaluates classification and gallery retrieval when all test identities were visible during training:
- **Train**: 10 images / person (300 images total, 66.7%)
- **Val**: 2 images / person (60 images total, 13.3%)
- **Test**: 3 images / person (90 images total, 20.0%)
- **Disjointness**: 0 overlapping images across sets.

### Open-Set Protocol (`data/splits/open_set.json`)
Evaluates true open-universe biometric verification and unseen identity rejection:
- **Train Partition**: 20 identities (300 images) — used to train the embedding backbone with ArcFace.
- **Validation Partition**: 5 disjoint identities — 3 gallery images (15 total) and 12 probe images (60 total). Used strictly to calibrate decision thresholds ($\tau^*$) at $\text{FAR} \le 1.0\%$.
- **Test Partition**: 5 disjoint, completely unseen identities — 3 gallery images (15 total) and 12 probe images (60 total). Evaluated strictly with the frozen threshold.
- **Identity Disjointness**: $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$ (Zero leakage).

---

## 5. Critical Engineering Disclaimers

> [!WARNING]
> **Synthetic Nature & Domain Shift**: The images in this benchmark are procedurally generated 2D cartoon avatars with synthetically injected perturbations (rotation, illumination shifts, blur). Because avatars exhibit sharp geometric contours and flat color regions, metric learning models readily achieve near-ceiling closed-set accuracy. These results **do not** transfer directly to human facial recognition in unconstrained real-world environments.

> [!IMPORTANT]
> **Simulated Confidence Disclosure**: The `recognition_confidence` column in `attendance_log.csv` contains pre-simulated values ranging between 0.8700 and 0.9990. It is **never** utilized as a training supervision signal or cited as proof of model accuracy. All actual model confidences are recomputed from normalized cosine similarities of the deployed embedders.
