import math
import cv2


def calculate_angle(a, b, c):
    """
    Calculate the 2D planar angle ABC in degrees with vertex at point b.

    Args:
        a: First point (x, y)
        b: Middle point / vertex (x, y)
        c: Third point (x, y)

    Returns:
        float: Angle in degrees in range [0, 180].
    """
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle

    return angle


def calculate_torso_angle(left_shoulder, right_shoulder, left_hip, right_hip):
    """
    Calculate forward torso inclination angle in degrees relative to vertical (0 deg = upright).

    Args:
        left_shoulder: (x, y) in pixel coordinates
        right_shoulder: (x, y) in pixel coordinates
        left_hip: (x, y) in pixel coordinates
        right_hip: (x, y) in pixel coordinates

    Returns:
        float: Torso inclination angle from vertical in degrees.
    """
    # Shoulder midpoint
    sx = (left_shoulder[0] + right_shoulder[0]) / 2.0
    sy = (left_shoulder[1] + right_shoulder[1]) / 2.0

    # Hip midpoint
    hx = (left_hip[0] + right_hip[0]) / 2.0
    hy = (left_hip[1] + right_hip[1]) / 2.0

    dx = abs(sx - hx)
    dy = max(hy - sy, 1e-5)  # Avoid division by zero, positive downwards in screen space

    return math.degrees(math.atan2(dx, dy))


def evaluate_squat_depth(knee_angle):
    """
    Evaluate squat depth based on knee flexion angle.

    Returns:
        tuple: (status_text, score 0-100, bgr_color)
    """
    if knee_angle < 110:
        return "Good Depth", 100, (0, 255, 0)       # Green
    elif knee_angle <= 130:
        return "Almost There", 70, (0, 255, 255)    # Yellow
    else:
        return "Go Lower", 40, (0, 140, 255)        # Orange


def evaluate_knee_alignment(left_knee, right_knee, left_ankle, right_ankle):
    """
    Detect knee valgus (knees collapsing inward relative to ankles).

    Returns:
        tuple: (status_text, score 0-100, bgr_color)
    """
    knee_dist = abs(left_knee[0] - right_knee[0])
    ankle_dist = abs(left_ankle[0] - right_ankle[0])

    # If ankles are too close together or user is in side view, cannot reliably detect knee valgus
    if ankle_dist < 35:
        return "Alignment OK", 95, (0, 255, 0)

    ratio = knee_dist / ankle_dist

    if ratio >= 0.82:
        return "Good Alignment", 100, (0, 255, 0)     # Green
    elif ratio >= 0.68:
        return "Watch Knees", 65, (0, 255, 255)       # Yellow
    else:
        return "Check Knees", 35, (0, 0, 255)         # Red (valgus collapse)


def evaluate_torso_posture(torso_angle):
    """
    Evaluate spine/torso inclination angle during squat.

    Returns:
        tuple: (status_text, score 0-100, bgr_color)
    """
    if torso_angle < 35:
        return "Good Posture", 100, (0, 255, 0)     # Green
    elif torso_angle < 50:
        return "Slight Lean", 70, (0, 255, 255)     # Yellow
    else:
        return "Too Much Lean", 35, (0, 0, 255)     # Red (excessive forward lean)


def calculate_form_score(depth_score, alignment_score, posture_score):
    """
    Compute a composite exercise form score (0-100%) weighted across components.
    Weights: Depth (40%), Knee Alignment (30%), Torso Posture (30%).
    """
    score = (0.40 * depth_score) + (0.30 * alignment_score) + (0.30 * posture_score)
    return int(max(0, min(100, round(score))))


def draw_skeleton(frame, landmarks, w, h, color=(0, 255, 0)):
    """
    Draw connected skeletal lines between key anatomical joints for visual tracking.
    """
    # Key landmark indices:
    # 11: left_shoulder, 12: right_shoulder
    # 23: left_hip, 24: right_hip
    # 25: left_knee, 26: right_knee
    # 27: left_ankle, 28: right_ankle
    connections = [
        (11, 12),  # Shoulders
        (11, 23),  # Left torso
        (12, 24),  # Right torso
        (23, 24),  # Hips
        (23, 25),  # Left thigh
        (25, 27),  # Left shin
        (24, 26),  # Right thigh
        (26, 28),  # Right shin
    ]

    for start_idx, end_idx in connections:
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)

    # Draw joint points
    for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
        pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        cv2.circle(frame, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 7, color, 2, cv2.LINE_AA)


def get_webcam(preferred_indices=(1, 0), fallback_range=4):
    """
    Initialize and return a functional OpenCV VideoCapture device with automatic fallback.

    Args:
        preferred_indices: Tuple of camera indices to try in order (e.g. (1, 0) for phone then laptop).
        fallback_range: Total number of indices to probe if preferred indices fail.

    Returns:
        tuple: (cv2.VideoCapture instance, active_camera_index)
        If no camera is available, returns (None, -1).
    """
    candidates = list(preferred_indices)
    for idx in range(fallback_range):
        if idx not in candidates:
            candidates.append(idx)

    for idx in candidates:
        for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("DEFAULT", cv2.CAP_ANY)]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        h, w = test_frame.shape[:2]
                        print(f"[FitVision AI] Connected to Camera {idx} ({w}x{h}) using {backend_name} backend.")
                        return cap, idx
                    cap.release()
            except Exception:
                continue

    print("[FitVision AI] ERROR: Could not initialize any camera.")
    print("Troubleshooting steps:")
    print(" 1. Check if the camera is currently in use by another program (Zoom, Teams, Browser, etc.).")
    print(" 2. If using a phone camera (e.g., DroidCam/Iriun), ensure the phone app and desktop client are running.")
    print(" 3. Verify Windows Camera permissions in Settings > Privacy & security > Camera.")
    print(" 4. Run 'python camera_test.py' to diagnose all connected video devices.")

    return None, -1
