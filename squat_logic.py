"""
squat_logic.py
--------------
Core posture evaluation module for the Squat Posture Analyzer.

Feedback uses professional severity-tagged messages:
  [PASS]    – rule satisfied
  [WARNING] – minor deviation detected
  [ALERT]   – significant form error requiring correction

Evaluation Rules (per CVP.docx):
  1. Back/Torso Angle  – angle at hip between shoulder->hip and knee->hip vectors.
  2. Knee-Over-Toe     – knee x-position must not significantly exceed ankle x-position.
  3. Squat Depth       – hip must descend at/below knee level during squatting phase.
  4. Heel Lift         – foot-to-ankle vertical gap must not exceed normal range.

Rep counting uses a simple phase-state machine:
  STANDING -> SQUATTING (hip drops to/below knee) -> STANDING = +1 rep
"""

import numpy as np


# ─── Severity constants ───────────────────────────────────────────────────────
STATUS_PASS    = "PASS"
STATUS_WARNING = "WARNING"
STATUS_ALERT   = "ALERT"

# ─── Biomechanical thresholds ─────────────────────────────────────────────────
BACK_ANGLE_MIN      = 50.0   # degrees; angle at hip below this = excessive forward lean
BACK_ANGLE_WARNING  = 70.0   # degrees; between this and MIN = mild lean (WARNING)
KNEE_TOE_TOLERANCE  = 0.04   # normalised units; knee ahead of ankle by this margin = ALERT
KNEE_TOE_WARNING    = 0.02   # normalised units; mild knee drift = WARNING
DEPTH_MARGIN        = 0.02   # normalised units; hip must be at/below knee_y - this
HEEL_LIFT_MARGIN    = 0.12   # normalised units; foot_y - ankle_y > this = heel off ground


class FeedbackItem:
    """
    A single piece of posture feedback with a status code and message.

    Attributes
    ----------
    status  : str   – STATUS_PASS | STATUS_WARNING | STATUS_ALERT
    rule    : str   – short rule name (e.g. 'Back Angle')
    message : str   – human-readable corrective instruction
    """
    def __init__(self, status: str, rule: str, message: str):
        self.status  = status
        self.rule    = rule
        self.message = message

    def label(self) -> str:
        """Return a formatted single-line string for overlay rendering."""
        return f"[{self.status}] {self.rule}: {self.message}"

    def __repr__(self):
        return f"FeedbackItem(status={self.status!r}, rule={self.rule!r})"


class SquatLogic:
    """
    Stateful squat posture evaluator with professional severity-based feedback.
    Call evaluate_posture() once per frame; state is maintained between calls.
    """

    def __init__(self):
        self.rep_count: int  = 0
        self.phase:     str  = "STANDING"   # "STANDING" | "SQUATTING"
        self._prev_phase: str = "STANDING"

    # ── Geometry utilities ────────────────────────────────────────────────────

    @staticmethod
    def calculate_angle(a, b, c) -> float:
        """
        Interior angle at vertex *b* formed by rays b->a and b->c, in degrees.
        Uses the dot-product formula: theta = arccos( (ba . bc) / |ba||bc| )
        """
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)
        ba = a - b
        bc = c - b
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0:
            return 180.0
        cos_val = np.dot(ba, bc) / norm
        return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

    # ── Evaluation rules ─────────────────────────────────────────────────────

    def _rule_back_angle(self, shoulder, hip, knee) -> tuple[FeedbackItem, float]:
        """
        Rule 1 – Back / Torso Angle
        Computes the interior angle at the hip using landmarks:
            shoulder -> hip -> knee
        A value near 180 deg means an upright torso; lower values indicate forward lean.
        """
        angle = self.calculate_angle(shoulder, hip, knee)

        if angle < BACK_ANGLE_MIN:
            item = FeedbackItem(
                STATUS_ALERT, "Back Angle",
                f"Straighten your torso — excessive forward lean detected ({angle:.0f} deg)"
            )
        elif angle < BACK_ANGLE_WARNING:
            item = FeedbackItem(
                STATUS_WARNING, "Back Angle",
                f"Maintain a more upright torso ({angle:.0f} deg)"
            )
        else:
            item = FeedbackItem(
                STATUS_PASS, "Back Angle",
                f"Torso alignment is good ({angle:.0f} deg)"
            )
        return item, angle

    def _rule_knee_over_toe(self, knee, ankle) -> FeedbackItem:
        """
        Rule 2 – Knee-Over-Toe
        Relative positional analysis on normalised x-coordinates.
        In a left-side view, excessive knee_x > ankle_x means the knee has drifted
        past the toes — a recognised form fault.
        """
        diff = knee[0] - ankle[0]

        if diff > KNEE_TOE_TOLERANCE:
            return FeedbackItem(
                STATUS_ALERT, "Knee Tracking",
                f"Drive knees back — they are extending past the toes ({diff:.3f} offset)"
            )
        elif diff > KNEE_TOE_WARNING:
            return FeedbackItem(
                STATUS_WARNING, "Knee Tracking",
                f"Slight forward knee drift detected ({diff:.3f} offset)"
            )
        return FeedbackItem(
            STATUS_PASS, "Knee Tracking",
            "Knee position is within acceptable range"
        )

    def _rule_squat_depth(self, hip, knee) -> FeedbackItem:
        """
        Rule 3 – Squat Depth
        Distance metric on normalised y-coordinates (y increases downward in image space).
        The hip must descend to at least knee level for a full squat.
        """
        hip_y, knee_y = hip[1], knee[1]
        if hip_y < (knee_y - DEPTH_MARGIN):
            deficit = knee_y - hip_y
            return FeedbackItem(
                STATUS_ALERT, "Squat Depth",
                f"Lower your hips to reach parallel depth (deficit: {deficit:.3f})"
            )
        return FeedbackItem(
            STATUS_PASS, "Squat Depth",
            "Adequate squat depth achieved"
        )

    def _rule_heel_lift(self, ankle, foot) -> FeedbackItem:
        """
        Rule 4 – Heel Lift
        In normal stance the foot-index (big toe) sits below the ankle:
            foot_y - ankle_y ~ 0.06 to 0.10 (normalised)
        When the heel lifts, the ankle rises (lower y-value), widening the gap
        beyond HEEL_LIFT_MARGIN.
        """
        gap = foot[1] - ankle[1]
        if gap > HEEL_LIFT_MARGIN:
            return FeedbackItem(
                STATUS_ALERT, "Heel Contact",
                f"Keep heels firmly on the ground (lift gap: {gap:.3f})"
            )
        return FeedbackItem(
            STATUS_PASS, "Heel Contact",
            "Heel contact is stable"
        )

    # ── Phase / rep state machine ─────────────────────────────────────────────

    def _update_phase(self, hip, knee):
        """
        STANDING  -> SQUATTING  when hip_y >= knee_y - DEPTH_MARGIN
        SQUATTING -> STANDING   when hip rises back up → completed rep
        """
        in_squat = hip[1] >= (knee[1] - DEPTH_MARGIN)

        if in_squat:
            self.phase = "SQUATTING"
        else:
            if self._prev_phase == "SQUATTING":
                self.rep_count += 1
            self.phase = "STANDING"

        self._prev_phase = self.phase

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_posture(self, joints: dict) -> dict:
        """
        Evaluate squat posture for a single video frame.

        Parameters
        ----------
        joints : dict
            Keys: 'shoulder', 'hip', 'knee', 'ankle', 'foot'
            Values: [x, y] normalised MediaPipe landmark coordinates.

        Returns
        -------
        dict with keys:
            items     : list[FeedbackItem]  – one item per rule
            overall   : str                 – 'PASS' | 'WARNING' | 'ALERT'
            rep_count : int
            phase     : str                 – 'STANDING' | 'SQUATTING'
            angles    : dict                – {'back_angle': float}
        """
        shoulder = joints["shoulder"]
        hip      = joints["hip"]
        knee     = joints["knee"]
        ankle    = joints["ankle"]
        foot     = joints["foot"]

        self._update_phase(hip, knee)

        items: list[FeedbackItem] = []
        angles: dict              = {}

        # Rule 1 – Back angle
        back_item, back_angle = self._rule_back_angle(shoulder, hip, knee)
        angles["back_angle"] = round(back_angle, 1)
        items.append(back_item)

        # Rule 2 – Knee tracking
        items.append(self._rule_knee_over_toe(knee, ankle))

        # Rule 3 – Squat depth (only evaluated when actively squatting)
        if self.phase == "SQUATTING":
            items.append(self._rule_squat_depth(hip, knee))

        # Rule 4 – Heel lift
        items.append(self._rule_heel_lift(ankle, foot))

        # Overall severity: ALERT > WARNING > PASS
        statuses = [i.status for i in items]
        if STATUS_ALERT in statuses:
            overall = STATUS_ALERT
        elif STATUS_WARNING in statuses:
            overall = STATUS_WARNING
        else:
            overall = STATUS_PASS

        return {
            "items":     items,
            "overall":   overall,
            "rep_count": self.rep_count,
            "phase":     self.phase,
            "angles":    angles,
        }

    def reset(self):
        """Reset rep counter and phase (call before processing a new video)."""
        self.rep_count    = 0
        self.phase        = "STANDING"
        self._prev_phase  = "STANDING"