import unittest
from utils import (
    calculate_angle,
    calculate_torso_angle,
    evaluate_squat_depth,
    evaluate_knee_alignment,
    evaluate_torso_posture,
    calculate_form_score
)

class TestFitVisionLogic(unittest.TestCase):
    def test_calculate_angle_straight(self):
        angle = calculate_angle((0, 0), (0, 1), (0, 2))
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_calculate_angle_right_angle(self):
        angle = calculate_angle((0, 1), (0, 0), (1, 0))
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_calculate_angle_acute(self):
        angle = calculate_angle((1, 1), (0, 0), (1, 0))
        self.assertAlmostEqual(angle, 45.0, places=1)

    def test_torso_angle_vertical(self):
        # Shoulders vertically aligned above hips
        ls, rs = (90, 50), (110, 50)
        lh, rh = (90, 150), (110, 150)
        angle = calculate_torso_angle(ls, rs, lh, rh)
        self.assertAlmostEqual(angle, 0.0, places=1)

    def test_torso_angle_inclined(self):
        # 45 degree forward lean
        ls, rs = (190, 50), (210, 50)
        lh, rh = (90, 150), (110, 150)
        angle = calculate_torso_angle(ls, rs, lh, rh)
        self.assertAlmostEqual(angle, 45.0, places=1)

    def test_squat_depth_evaluation(self):
        status, score, _ = evaluate_squat_depth(100)
        self.assertEqual(status, "Good Depth")
        self.assertEqual(score, 100)

        status, score, _ = evaluate_squat_depth(120)
        self.assertEqual(status, "Almost There")
        self.assertEqual(score, 70)

        status, score, _ = evaluate_squat_depth(140)
        self.assertEqual(status, "Go Lower")
        self.assertEqual(score, 40)

    def test_knee_alignment_evaluation(self):
        # Good alignment: knees 90px apart, ankles 100px apart (ratio 0.9)
        status, score, _ = evaluate_knee_alignment((155, 300), (245, 300), (150, 450), (250, 450))
        self.assertEqual(status, "Good Alignment")
        self.assertEqual(score, 100)

        # Valgus collapse: knees 50px apart, ankles 100px apart (ratio 0.5)
        status, score, _ = evaluate_knee_alignment((175, 300), (225, 300), (150, 450), (250, 450))
        self.assertEqual(status, "Check Knees")
        self.assertEqual(score, 35)

    def test_torso_posture_evaluation(self):
        status, score, _ = evaluate_torso_posture(20)
        self.assertEqual(status, "Good Posture")
        self.assertEqual(score, 100)

        status, score, _ = evaluate_torso_posture(42)
        self.assertEqual(status, "Slight Lean")
        self.assertEqual(score, 70)

        status, score, _ = evaluate_torso_posture(58)
        self.assertEqual(status, "Too Much Lean")
        self.assertEqual(score, 35)

    def test_composite_form_score(self):
        perfect_score = calculate_form_score(100, 100, 100)
        self.assertEqual(perfect_score, 100)

        poor_score = calculate_form_score(40, 35, 35)
        self.assertEqual(poor_score, 37)

    def test_squat_state_transition(self):
        counter = 0
        stage = "up"
        down_frames = 0
        up_frames = 0
        DOWN_THRESHOLD = 120
        UP_THRESHOLD = 155
        REQUIRED_DOWN_FRAMES = 5
        REQUIRED_UP_FRAMES = 5

        for _ in range(REQUIRED_DOWN_FRAMES):
            knee_angle = 100
            if knee_angle < DOWN_THRESHOLD:
                down_frames += 1
                up_frames = 0
                if down_frames >= REQUIRED_DOWN_FRAMES:
                    stage = "down"

        self.assertEqual(stage, "down")
        self.assertEqual(counter, 0)

        for _ in range(REQUIRED_UP_FRAMES):
            knee_angle = 165
            if knee_angle > UP_THRESHOLD:
                up_frames += 1
                down_frames = 0
                if up_frames >= REQUIRED_UP_FRAMES:
                    if stage == "down":
                        counter += 1
                    stage = "up"

        self.assertEqual(stage, "up")
        self.assertEqual(counter, 1)

if __name__ == "__main__":
    unittest.main()
