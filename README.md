# 🏋️ Squat Posture Analyzer

> A real-time computer vision system that evaluates squat form using **MediaPipe Pose**, delivers instant corrective feedback, and counts repetitions — via a clean **Gradio** web interface supporting both video upload and live webcam.

---

## 📸 Demo

| Video Upload Mode | Webcam Mode |
|:-:|:-:|
| Upload a pre-recorded squat video | Live real-time analysis via your webcam |
| Get back an annotated video with feedback overlay | See skeleton, angles & feedback in real time |

---

## ✨ Features

- 🦴 **Pose Estimation** — MediaPipe's pre-trained BlazePose model extracts 33 body landmarks per frame
- 📐 **4 Biomechanical Rules** — Back angle, knee-over-toe, squat depth, heel lift (see [Evaluation Rules](#-evaluation-rules))
- 🔄 **Rep Counter** — Phase-state machine automatically counts completed squat repetitions
- 🎨 **Rich Annotations** — Skeleton overlay, joint labels, angle values, colour-coded feedback panel
- 📹 **Video Upload** — Process any `.mp4` file; download the annotated result
- 📷 **Live Webcam** — Stream from your webcam with sub-second latency
- 🌐 **Gradio Web UI** — Runs in your browser; optional public sharing via Gradio tunnels

---

## 🔬 Technical Approach

### 1. Pose Estimation (MediaPipe Pose)

MediaPipe Pose is used to detect **33 body landmarks** in each frame. Landmarks are returned as *normalised* coordinates `(x, y) ∈ [0, 1]` relative to the frame dimensions.

For squat analysis the following left-side landmarks are used:

| Landmark | MediaPipe Index |
|----------|----------------|
| Left Shoulder | 11 |
| Left Hip | 23 |
| Left Knee | 25 |
| Left Ankle | 27 |
| Left Foot Index (big toe) | 31 |

> **Why left-side?** The system is optimised for a **left-side view** of the subject, which provides the clearest sagittal-plane profile of the squat.

### 2. Angle Calculation

The interior angle at a joint **B** formed by landmarks **A → B → C** is computed as:

```
cos θ = (BA · BC) / (|BA| × |BC|)
θ     = arccos(clamp(cos θ, -1, 1))
```

This is implemented in `SquatLogic.calculate_angle(a, b, c)`.

### 3. Phase / Rep State Machine

A simple 2-state machine tracks the squat cycle:

```
STANDING ──(hip descends to knee level)──► SQUATTING
           ◄─(hip rises above knee level)── SQUATTING  → rep_count += 1
```

---

## 📏 Evaluation Rules

All thresholds are defined as constants at the top of `squat_logic.py` and can be tuned.

### Rule 1 — Back / Torso Angle

| | |
|---|---|
| **Method** | Interior angle at the hip, computed from `shoulder → hip → knee` |
| **Threshold** | `< 50°` signals excessive forward lean |
| **Feedback** | ❌ *Straighten your back (torso leaning too far forward)* |

```
BACK_ANGLE_MIN = 50.0  # degrees
```

### Rule 2 — Knee Over Toe

| | |
|---|---|
| **Method** | Relative x-position: `knee_x - ankle_x > KNEE_TOE_TOLERANCE` |
| **Threshold** | `> 0.04` (normalised units, ~4 % of frame width) |
| **Feedback** | ❌ *Keep your knees behind your toes* |

```
KNEE_TOE_TOLERANCE = 0.04  # normalised coords
```

### Rule 3 — Squat Depth

| | |
|---|---|
| **Method** | Vertical distance: `hip_y < knee_y - DEPTH_MARGIN` (y increases downward) |
| **Threshold** | `DEPTH_MARGIN = 0.02` |
| **Feedback** | ❌ *Lower your hips further (increase squat depth)* |
| **Note** | Only checked when `phase == "SQUATTING"` to avoid false positives while standing |

```
DEPTH_MARGIN = 0.02  # normalised coords
```

### Rule 4 — Heel Lift

| | |
|---|---|
| **Method** | `ankle_y < foot_y - HEEL_LIFT_MARGIN` — ankle rising above toe level |
| **Threshold** | `HEEL_LIFT_MARGIN = 0.03` |
| **Feedback** | ❌ *Keep your heels on the ground* |

```
HEEL_LIFT_MARGIN = 0.03  # normalised coords
```

---

## ⚠️ Assumptions

1. **Side-view capture** — The system is calibrated for a clear left-side profile. A frontal view will cause the x-axis rules (knee-over-toe) to behave incorrectly.
2. **Single person** — Only one subject should be in the frame. MediaPipe's standard Pose model (non-holistic) tracks the most prominent person.
3. **Adequate lighting** — Dark or heavily backlit scenes may reduce landmark confidence.
4. **Consistent frame rate** — The rep counter is based on positional transitions, not time, so an inconsistent frame rate does not cause counting errors.
5. **Normalised coordinates** — All thresholds are expressed in normalised `[0, 1]` space, making them resolution-agnostic.
6. **Left-side landmarks only** — If the subject faces right (right side visible), landmarks 12/24/26/28/32 (right side) should be substituted in `_extract_joints()`.

---

## 🚧 Limitations

| Limitation | Detail |
|---|---|
| **Side-view dependency** | Frontal or rear views reduce accuracy significantly |
| **No temporal smoothing** | Noisy landmark jitter can trigger false feedback on single frames |
| **Fixed-threshold rules** | Thresholds are not personalised to body proportions or flexibility |
| **Single-person only** | Multi-person scenes are unsupported |
| **Occlusion sensitivity** | Covered joints (e.g., loose clothing) lower MediaPipe confidence |
| **Webcam latency** | Processing latency depends on hardware; a GPU is not required but helps |
| **No model training** | No custom model was trained; relies entirely on MediaPipe's pre-trained weights |

---

---

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- A webcam (required for Live Webcam mode)
- Python 3.10 (recommended — this project was tested on `squat_env` with Python 3.10)

### Step 1 — Create the Conda Environment

```bash
conda create -n squat_env python=3.10 -y
```

### Step 2 — Activate the Environment

```bash
conda activate squat_env
```

> You should see `(squat_env)` appear at the start of your terminal prompt.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages into the `squat_env` environment.

### `requirements.txt` (for reference)

```
mediapipe==0.10.9
opencv-python==4.9.0.80
numpy==1.26.4
gradio>=4.44.1
```

> **Note:** `gradio>=4.44.1` is required for compatibility with the pydantic/fastapi versions in modern conda environments.

---

## Usage

### Step 4 — Launch the Web App

Make sure `squat_env` is activated, then run:

```bash
(squat_env) python app.py
```

The Gradio interface will open automatically at **http://localhost:7860**.
A public shareable URL is also printed if `share=True` is set.

### Video Upload Mode

1. Click the **"Video Upload"** tab.
2. Drop or browse to your `.mp4` squat video.
3. Click **"Analyse Video"**.
4. The annotated video (skeleton, feedback panel, rep counter) downloads to the right panel.

> **Tip:** Sample videos are included in the `videos/` folder:
> - `videos/correct_posture_squat.mp4`
> - `videos/incorrect_posture_squats.mp4`
> - `videos/Exercise Tutorial - Squat.mp4`

### Live Webcam Mode

1. Click the **"Live Webcam"** tab.
2. Grant browser camera permissions when prompted.
3. Stand in **left-side view** (left side of body facing the camera) within ~2–3 m.
4. Perform squats — feedback status (PASS / WARNING / ALERT), phase, and rep count update in real time.
5. Click **"Reset Session"** to clear the rep counter and restart.

---

## Project Structure

```
cv_mini_project/
├── app.py                          # Gradio application (video upload + webcam tabs)
├── squat_logic.py                  # Core posture evaluation logic & rep counter
├── requirements.txt                # Python dependencies
├── videos/                         # Sample squat videos
│   ├── correct_posture_squat.mp4   # Reference: correct squat form
│   ├── incorrect_posture_squats.mp4# Reference: common form errors
│   └── Exercise Tutorial - Squat.mp4
└── README.md                       # This file
```

---

## 📚 References

- [The Real Science of the Squat – Squat University](https://squatuniversity.com/2016/04/20/the-real-science-of-the-squat/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📄 License

This project is submitted as part of a Computer Vision Engineering Mini Project (CVP). Free to use and adapt for educational purposes.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Demonstration Video

Mini project demonstration video link: "https://drive.google.com/file/d/1HXG7dwR44fWLphQBYrqEHhFuI1fb0x2K/view?usp=sharing"