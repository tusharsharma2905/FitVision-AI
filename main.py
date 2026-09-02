import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# SETTINGS
# =========================

MODEL_PATH = "pose_landmarker_lite.task"


# =========================
# MEDIAPIPE POSE SETUP
# =========================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.PoseLandmarker.create_from_options(options)


# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# =========================
# VARIABLES
# =========================

counter = 0
stage = "up"

down_frames = 0
up_frames = 0

frame_count = 0
start_time = time.time()


# =========================
# ANGLE FUNCTION
# =========================

def calculate_angle(a, b, c):
    import math

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# =========================
# MAIN LOOP
# =========================

timestamp = 0

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Increase timestamp
    timestamp += 33

    # Detect pose
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # =========================
    # POSE DETECTED
    # =========================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # ---------------------------------
        # LEFT LEG
        # ---------------------------------

        left_hip = landmarks[23]
        left_knee = landmarks[25]
        left_ankle = landmarks[27]

        # ---------------------------------
        # RIGHT LEG
        # ---------------------------------

        right_hip = landmarks[24]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]

        # ---------------------------------
        # CONVERT LANDMARKS TO PIXELS
        # ---------------------------------

        h, w, _ = frame.shape

        left_hip_point = (
            left_hip.x * w,
            left_hip.y * h
        )

        left_knee_point = (
            left_knee.x * w,
            left_knee.y * h
        )

        left_ankle_point = (
            left_ankle.x * w,
            left_ankle.y * h
        )

        right_hip_point = (
            right_hip.x * w,
            right_hip.y * h
        )

        right_knee_point = (
            right_knee.x * w,
            right_knee.y * h
        )

        right_ankle_point = (
            right_ankle.x * w,
            right_ankle.y * h
        )

        # ---------------------------------
        # CALCULATE KNEE ANGLES
        # ---------------------------------

        left_angle = calculate_angle(
            left_hip_point,
            left_knee_point,
            left_ankle_point
        )

        right_angle = calculate_angle(
            right_hip_point,
            right_knee_point,
            right_ankle_point
        )

        # Average both knees
        knee_angle = (left_angle + right_angle) / 2

        # =========================
        # SQUAT COUNTER
        # =========================


        DOWN_THRESHOLD = 120
        UP_THRESHOLD = 155

        REQUIRED_DOWN_FRAMES = 5
        REQUIRED_UP_FRAMES = 5


        if knee_angle < DOWN_THRESHOLD:

            down_frames += 1
            up_frames = 0

            # Only enter DOWN after
            # several consecutive frames
            if down_frames >= REQUIRED_DOWN_FRAMES:
                stage = "down"


        elif knee_angle > UP_THRESHOLD:

            up_frames += 1
            down_frames = 0

            # Only count when we were actually DOWN
            # and remain UP for several frames
            if up_frames >= REQUIRED_UP_FRAMES:

                if stage == "down":
                    counter += 1
                    print("SQUAT COUNTED:", counter)

                stage = "up"


        else:

            # Angle is between thresholds
            # Don't immediately change stage
            down_frames = 0
            up_frames = 0

        # =========================
        # DRAW LANDMARKS
        # =========================

        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # =========================
        # DISPLAY INFORMATION
        # =========================

        cv2.putText(
            frame,
            f"Squats: {counter}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Stage: {stage}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Left Knee: {int(left_angle)}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Right Knee: {int(right_angle)}",
            (10, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Average: {int(knee_angle)}",
            (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # =========================
    # FPS
    # =========================

    frame_count += 1

    elapsed_time = time.time() - start_time

    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =========================
    # SHOW FRAME
    # =========================

    cv2.imshow(
        "FitVision AI - Squat Counter",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()
landmarker.close()