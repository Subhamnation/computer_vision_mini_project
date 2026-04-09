"""
app.py
------
Squat Posture Analyzer – Gradio Web Application

Modes:
  Tab 1 – Video Upload : Process a pre-recorded video; download annotated result.
  Tab 2 – Webcam       : Live webcam stream with real-time posture feedback.

Usage:
  python app.py
"""

import cv2
import mediapipe as mp
import numpy as np
import tempfile
import gradio as gr

from squat_logic import SquatLogic, STATUS_PASS, STATUS_WARNING, STATUS_ALERT

# ─── MediaPipe setup ──────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ─── Colour palette (BGR) ────────────────────────────────────────────────────
CLR_PASS    = (40,  200, 40)    # green
CLR_WARNING = (30,  180, 240)   # amber
CLR_ALERT   = (40,   50, 220)   # red
CLR_INFO    = (210, 210, 210)   # light grey
CLR_PANEL   = (18,   18,  28)   # near-black navy
CLR_ACCENT  = (200, 140,  40)   # gold

STATUS_COLOR = {
    STATUS_PASS:    CLR_PASS,
    STATUS_WARNING: CLR_WARNING,
    STATUS_ALERT:   CLR_ALERT,
}

PANEL_W = 420   # px – fixed-width left sidebar


def _draw_overlay(frame, result: dict):
    """Render the professional annotation overlay onto *frame* (in-place, BGR)."""
    h, w = frame.shape[:2]

    # ── Left sidebar panel ─────────────────────────────────────────────────────
    sidebar = np.full((h, PANEL_W, 3), CLR_PANEL, dtype=np.uint8)

    # Header bar
    cv2.rectangle(sidebar, (0, 0), (PANEL_W, 56), CLR_ACCENT, -1)
    cv2.putText(sidebar, "SQUAT POSTURE ANALYZER",
                (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, (20, 20, 20), 2)

    # Overall status badge
    overall   = result.get("overall", STATUS_PASS)
    ov_color  = STATUS_COLOR.get(overall, CLR_INFO)
    badge_lbl = f"OVERALL: {overall}"
    cv2.rectangle(sidebar, (12, 66), (PANEL_W - 12, 104), ov_color, -1)
    cv2.putText(sidebar, badge_lbl,
                (20, 94), cv2.FONT_HERSHEY_DUPLEX, 0.70, (10, 10, 10), 2)

    # Phase & rep counter
    phase     = result.get("phase", "STANDING")
    rep_count = result.get("rep_count", 0)
    ph_color  = CLR_PASS if phase == "SQUATTING" else CLR_WARNING

    cv2.putText(sidebar, "PHASE",
                (16, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_INFO, 1)
    cv2.putText(sidebar, phase,
                (16, 162), cv2.FONT_HERSHEY_DUPLEX, 0.82, ph_color, 2)

    cv2.putText(sidebar, "REPS",
                (220, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_INFO, 1)
    cv2.putText(sidebar, str(rep_count),
                (220, 162), cv2.FONT_HERSHEY_DUPLEX, 1.10, CLR_ACCENT, 2)

    # Back angle
    angles    = result.get("angles", {})
    back_ang  = angles.get("back_angle", None)
    if back_ang is not None:
        cv2.putText(sidebar, "BACK ANGLE",
                    (16, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_INFO, 1)
        cv2.putText(sidebar, f"{back_ang:.1f} deg",
                    (16, 224), cv2.FONT_HERSHEY_DUPLEX, 0.78, (180, 210, 255), 2)

    # Divider
    cv2.line(sidebar, (12, 238), (PANEL_W - 12, 238), (60, 60, 80), 1)

    # Rule-by-rule feedback items
    items = result.get("items", [])
    y = 262
    for item in items:
        if y + 52 > h:
            break
        color = STATUS_COLOR.get(item.status, CLR_INFO)

        # Status tag pill
        cv2.rectangle(sidebar, (12, y - 18), (130, y + 4), color, -1)
        cv2.putText(sidebar, item.status,
                    (18, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1)

        # Rule name
        cv2.putText(sidebar, item.rule,
                    (138, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_INFO, 1)

        # Message (word-wrapped at ~48 chars)
        msg   = item.message
        lines = [msg[i:i+46] for i in range(0, len(msg), 46)]
        for j, line in enumerate(lines[:2]):
            cv2.putText(sidebar, line,
                        (18, y + 20 + j * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        y += 70
        cv2.line(sidebar, (12, y - 8), (PANEL_W - 12, y - 8), (40, 40, 55), 1)

    # Footer
    cv2.putText(sidebar, "MediaPipe Pose  |  Antigravity CV",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 90), 1)

    # ── Compose: sidebar | video frame ─────────────────────────────────────────
    if w + PANEL_W <= frame.shape[1] + PANEL_W:     # always true – just compositing
        canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
        canvas[:, :PANEL_W] = sidebar
        canvas[:, PANEL_W:] = frame
        frame[:] = canvas[:, PANEL_W: PANEL_W + w]  # write back annotated video
        return canvas

    return frame


def _extract_joints(lm):
    """Return named left-side landmark coordinates (normalised)."""
    return {
        "shoulder": [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                     lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
        "hip":      [lm[mp_pose.PoseLandmark.LEFT_HIP].x,
                     lm[mp_pose.PoseLandmark.LEFT_HIP].y],
        "knee":     [lm[mp_pose.PoseLandmark.LEFT_KNEE].x,
                     lm[mp_pose.PoseLandmark.LEFT_KNEE].y],
        "ankle":    [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                     lm[mp_pose.PoseLandmark.LEFT_ANKLE].y],
        "foot":     [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                     lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y],
    }


def _annotate_frame(frame, pose_results, logic: SquatLogic):
    """Apply pose skeleton and info overlay; return composite canvas."""
    h, w = frame.shape[:2]

    if pose_results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 230, 180), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(220, 200, 60), thickness=2),
        )

        lm     = pose_results.pose_landmarks.landmark
        joints = _extract_joints(lm)
        result = logic.evaluate_posture(joints)

        # Label key joints
        for name, idx in [("Hip",   mp_pose.PoseLandmark.LEFT_HIP),
                           ("Knee",  mp_pose.PoseLandmark.LEFT_KNEE),
                           ("Ankle", mp_pose.PoseLandmark.LEFT_ANKLE)]:
            px = int(lm[idx].x * w) + 8
            py = int(lm[idx].y * h) - 8
            cv2.putText(frame, name, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        return _draw_overlay(frame, result), result

    else:
        no_result = {
            "items":     [],
            "overall":   "NO POSE",
            "rep_count": logic.rep_count,
            "phase":     logic.phase,
            "angles":    {},
        }
        cv2.putText(frame, "NO POSE DETECTED — adjust camera position",
                    (12, 40), cv2.FONT_HERSHEY_DUPLEX, 0.70, CLR_WARNING, 2)
        return frame, no_result


# ─── Video Upload handler ─────────────────────────────────────────────────────

def process_uploaded_video(video):
    """Process full video; return path to annotated output file."""
    if video is None:
        return None

    video_path = video["name"] if isinstance(video, dict) else video

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open the uploaded video file.")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w  = orig_w + PANEL_W

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out = cv2.VideoWriter(tmp.name,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (out_w, orig_h))
    logic = SquatLogic()

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            canvas, _ = _annotate_frame(frame, results, logic)
            out.write(canvas)

    cap.release()
    out.release()
    return tmp.name


# ─── Webcam / Live frame handler ──────────────────────────────────────────────
_webcam_logic: SquatLogic = SquatLogic()
_webcam_pose              = None


def process_webcam_frame(frame):
    """Process a single webcam frame (RGB numpy array); return annotated RGB."""
    global _webcam_pose, _webcam_logic

    if frame is None:
        return None

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if _webcam_pose is None:
        _webcam_pose = mp_pose.Pose(min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = _webcam_pose.process(rgb)
    canvas, _ = _annotate_frame(bgr, results, _webcam_logic)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def reset_webcam():
    """Reset rep counter and pose session."""
    global _webcam_logic, _webcam_pose
    _webcam_logic = SquatLogic()
    if _webcam_pose is not None:
        _webcam_pose.close()
        _webcam_pose = None
    return None, "Session reset — rep counter cleared."


# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body, .gradio-container {
    background: #0d0d1a !important;
    font-family: 'Inter', sans-serif !important;
    color: #e0e0f0 !important;
}

/* Header */
#app-header {
    background: linear-gradient(135deg, #1a1a3e 0%, #0d0d1a 100%);
    border-bottom: 2px solid #c88a28;
    padding: 28px 32px 20px 32px;
    border-radius: 12px 12px 0 0;
    margin-bottom: 8px;
}
#app-header h1 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #f0c040, #e08030);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
#app-header p {
    font-size: 1.05rem !important;
    color: #9090b8 !important;
    margin: 0;
}

/* Tabs */
.tab-nav {
    background: #13132a !important;
    border-radius: 8px !important;
    padding: 4px !important;
}
.tab-nav button {
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    color: #7070a0 !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    transition: all 0.2s;
}
.tab-nav button.selected {
    background: #c88a28 !important;
    color: #0d0d1a !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #c88a28, #e0a830) !important;
    color: #0d0d1a !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    border: none !important;
    transition: opacity 0.2s;
}
button.primary:hover { opacity: 0.88; }

button.secondary {
    background: #1e1e3a !important;
    color: #c88a28 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    border: 1px solid #c88a28 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
}

/* Labels */
label span {
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    color: #8888b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Info/tip row */
#tips-row {
    background: #13132a;
    border: 1px solid #2a2a50;
    border-radius: 10px;
    padding: 16px 24px;
    margin-top: 12px;
    font-size: 0.92rem !important;
    color: #7878a8 !important;
}
"""

with gr.Blocks(title="Squat Posture Analyzer") as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="app-header">
      <h1>Squat Posture Analyzer</h1>
      <p>
        Real-time biomechanical analysis powered by <strong>MediaPipe Pose</strong>
        &nbsp;|&nbsp; Evaluates: Back Angle &bull; Knee Tracking &bull; Squat Depth &bull; Heel Contact
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Video Upload ────────────────────────────────────────────────
        with gr.TabItem("  Video Upload  "):
            gr.Markdown(
                "**Upload a squat recording** (MP4, side-view preferred). "
                "The system will annotate every frame and return the processed video."
            )
            with gr.Row(equal_height=True):
                vid_in  = gr.Video(label="Input Video", format="mp4", scale=1)
                vid_out = gr.Video(label="Annotated Output", format="mp4", scale=1)

            with gr.Row():
                btn_process = gr.Button("Analyse Video", variant="primary", scale=0)

            btn_process.click(
                fn=process_uploaded_video,
                inputs=vid_in,
                outputs=vid_out,
            )

        # ── Tab 2: Live Webcam ─────────────────────────────────────────────────
        with gr.TabItem("  Live Webcam  "):
            gr.Markdown(
                "**Enable your webcam** for real-time squat analysis. "
                "Position yourself in **left-side view** for optimal landmark detection."
            )
            with gr.Row(equal_height=True):
                cam_in  = gr.Image(
                    label="Webcam Input",
                    sources=["webcam"],
                    streaming=True,
                    scale=1,
                )
                cam_out = gr.Image(label="Annotated Output", scale=1)

            cam_in.stream(
                fn=process_webcam_frame,
                inputs=cam_in,
                outputs=cam_out,
            )

            with gr.Row():
                btn_reset  = gr.Button("Reset Session", variant="secondary", scale=0)
                reset_info = gr.Textbox(
                    label="System Status",
                    interactive=False,
                    placeholder="Session active",
                    scale=1,
                )

            btn_reset.click(
                fn=reset_webcam,
                inputs=None,
                outputs=[cam_out, reset_info],
            )

    # ── Tips footer ─────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="tips-row">
      <strong style="color:#c88a28;">Best Practices</strong> &nbsp;|&nbsp;
      Shoot from the <strong>left side</strong> &nbsp;&bull;&nbsp;
      Ensure adequate, even lighting &nbsp;&bull;&nbsp;
      Wear fitted clothing for clearer body contours &nbsp;&bull;&nbsp;
      Keep the full body in frame throughout the squat
    </div>
    """)


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", css=CSS)