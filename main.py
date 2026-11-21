import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import (
    calculate_angle,
    calculate_torso_angle,
    evaluate_squat_depth,
    evaluate_knee_alignment,
    evaluate_torso_posture,
    calculate_form_score,
    draw_skeleton,
    get_webcam,
)


# ==========================================
# SETTINGS & CONFIGURATION
# ==========================================

MODEL_PATH = "pose_landmarker_lite.task"

# Repetition counting thresholds (knee angle in degrees)
DOWN_THRESHOLD = 120
UP_THRESHOLD = 155

# Frame-based debounce (prevents jitter / false counts)
REQUIRED_DOWN_FRAMES = 5
REQUIRED_UP_FRAMES = 5


def main():
    # ==========================================
    # MEDIAPIPE POSE LANDMARKER INITIALIZATION
    # ==========================================

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    # ==========================================
    # CAMERA INITIALIZATION WITH FALLBACK
    # ==========================================

    # Allow CLI argument: e.g. "python main.py 0" or default to probe (1, 0)
    cli_idx = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    preferred = (cli_idx, 1, 0) if cli_idx is not None else (1, 0)

    cap, active_cam_idx = get_webcam(preferred_indices=preferred)

    if cap is None:
        print("[FitVision AI] CRITICAL: Could not open any camera. Exiting.")
        landmarker.close()
        return

    # ==========================================
    # WORKOUT STATE VARIABLES
    # ==========================================

    counter = 0
    stage = "up"

    down_frames = 0
    up_frames = 0

    frame_count = 0
    start_time = time.time()
    timestamp = 0

    # Form analysis session tracking
    last_rep_score = None
    rep_min_knee_angle = 180.0
    rep_scores = []

    print(f"\n[FitVision AI] Active Camera: {active_cam_idx}")
    print("[FitVision AI] Exercise Form Coach started. Press 'q' to quit.\n")

    try:
        while cap.isOpened():

            success, frame = cap.read()
            if not success:
                print("[FitVision AI] Warning: Camera frame dropped or unavailable.")
                break

            # Mirror horizontally for intuitive user interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR -> RGB for MediaPipe Tasks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp += 33
            result = landmarker.detect_for_video(mp_image, timestamp)

            # Default metrics
            live_form_score = 100
            depth_status, depth_color = "Standby", (200, 200, 200)
            align_status, align_color = "Standby", (200, 200, 200)
            posture_status, posture_color = "Standby", (200, 200, 200)
            knee_angle = 180
            torso_angle = 0

            # ==========================================
            # POSE DETECTION & FORM ANALYSIS
            # ==========================================

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                # Key joints in pixel coordinates
                # Shoulders
                ls = (landmarks[11].x * w, landmarks[11].y * h)
                rs = (landmarks[12].x * w, landmarks[12].y * h)
                # Hips
                lh = (landmarks[23].x * w, landmarks[23].y * h)
                rh = (landmarks[24].x * w, landmarks[24].y * h)
                # Knees
                lk = (landmarks[25].x * w, landmarks[25].y * h)
                rk = (landmarks[26].x * w, landmarks[26].y * h)
                # Ankles
                la = (landmarks[27].x * w, landmarks[27].y * h)
                ra = (landmarks[28].x * w, landmarks[28].y * h)

                # --------------------------------------
                # 1. Joint Angle Calculations
                # --------------------------------------
                left_knee_angle = calculate_angle(lh, lk, la)
                right_knee_angle = calculate_angle(rh, rk, ra)
                knee_angle = (left_knee_angle + right_knee_angle) / 2.0

                torso_angle = calculate_torso_angle(ls, rs, lh, rh)

                # --------------------------------------
                # 2. Form Analysis Components
                # --------------------------------------
                depth_status, depth_score, depth_color = evaluate_squat_depth(knee_angle)
                align_status, align_score, align_color = evaluate_knee_alignment(lk, rk, la, ra)
                posture_status, posture_score, posture_color = evaluate_torso_posture(torso_angle)

                # Composite Form Score (0 - 100%)
                live_form_score = calculate_form_score(depth_score, align_score, posture_score)

                # Dynamic skeleton color based on form quality
                if live_form_score >= 80:
                    skeleton_color = (0, 255, 0)      # Green
                elif live_form_score >= 60:
                    skeleton_color = (0, 255, 255)    # Yellow
                else:
                    skeleton_color = (0, 70, 255)     # Red/Orange

                # Draw connected anatomical skeleton
                draw_skeleton(frame, landmarks, w, h, color=skeleton_color)

                # --------------------------------------
                # 3. Squat Repetition & Form Tracking
                # --------------------------------------
                if knee_angle < DOWN_THRESHOLD:
                    down_frames += 1
                    up_frames = 0

                    if down_frames >= REQUIRED_DOWN_FRAMES:
                        stage = "down"

                    # Accumulate form scores and track deepest point of the rep
                    rep_scores.append(live_form_score)
                    if knee_angle < rep_min_knee_angle:
                        rep_min_knee_angle = knee_angle

                elif knee_angle > UP_THRESHOLD:
                    up_frames += 1
                    down_frames = 0

                    if up_frames >= REQUIRED_UP_FRAMES:
                        if stage == "down":
                            counter += 1
                            # Compute representative score for the completed rep
                            if rep_scores:
                                last_rep_score = int(sum(rep_scores) / len(rep_scores))
                            else:
                                last_rep_score = live_form_score

                            print(
                                f"[FitVision AI] SQUAT #{counter} COMPLETED | "
                                f"Min Knee Angle: {int(rep_min_knee_angle)} deg | "
                                f"Form Score: {last_rep_score}%"
                            )

                            # Reset per-rep accumulators
                            rep_scores = []
                            rep_min_knee_angle = 180.0

                        stage = "up"

                else:
                    # Transition zone between thresholds
                    down_frames = 0
                    up_frames = 0

            # ==========================================
            # SEMI-TRANSPARENT HUD OVERLAY
            # ==========================================

            overlay = frame.copy()
            panel_w, panel_h = 320, 250
            cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Border for HUD panel
            cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)

            # Header
            cv2.putText(
                frame,
                "FITVISION AI COACH",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 200),
                2,
                cv2.LINE_AA,
            )

            # Reps & Stage
            cv2.putText(
                frame,
                f"Squats: {counter}   [{stage.upper()}]",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0) if stage == "down" else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Form Evaluation Metrics
            cv2.putText(
                frame,
                f"Depth:   {depth_status}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                depth_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Knees:   {align_status}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                align_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Posture: {posture_status}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                posture_color,
                2,
                cv2.LINE_AA,
            )

            # Live and Last Rep Form Scores
            score_color = (0, 255, 0) if live_form_score >= 80 else ((0, 255, 255) if live_form_score >= 60 else (0, 70, 255))
            cv2.putText(
                frame,
                f"Live Form Score: {live_form_score}%",
                (20, 172),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                score_color,
                2,
                cv2.LINE_AA,
            )

            last_score_str = f"{last_rep_score}%" if last_rep_score is not None else "--"
            cv2.putText(
                frame,
                f"Last Rep Score:  {last_score_str}",
                (20, 197),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Angles & Device Metrics
            cv2.putText(
                frame,
                f"Knee: {int(knee_angle)} deg | Lean: {int(torso_angle)} deg",
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )

            # FPS & System info (Bottom Left of HUD)
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            cv2.putText(
                frame,
                f"FPS: {int(fps)} | Cam: {active_cam_idx}",
                (20, 248),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

            # ==========================================
            # DISPLAY FRAME
            # ==========================================

            cv2.imshow("FitVision AI - Exercise Form Coach", frame)

            # Exit cleanly when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # ==========================================
        # CLEANUP
        # ==========================================
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\n[FitVision AI] Camera released and session closed cleanly.")


if __name__ == "__main__":
    main()
