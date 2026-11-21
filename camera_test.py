import cv2
import sys

def scan_cameras(max_tested=4):
    """Scan and list all available camera devices."""
    print("Scanning available camera devices...")
    available = []
    for idx in range(max_tested):
        # Prefer DirectShow on Windows for speed and reliability
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
            
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                available.append((idx, w, h))
                print(f" [OK] Camera Index {idx}: Active ({w}x{h})")
            else:
                print(f" [WARN] Camera Index {idx}: Opened but could not read frame")
            cap.release()
        else:
            print(f" [NONE] Camera Index {idx}: Not available")
    return available

def test_camera(index=None):
    devices = scan_cameras()
    
    if not devices:
        print("\nERROR: No functional camera found on any index (0-3).")
        print("Troubleshooting tips:")
        print(" 1. Check Windows Camera Privacy Settings (Settings > Privacy & Security > Camera).")
        print(" 2. If using a phone camera (e.g. DroidCam/Iriun), make sure the app and PC client are running.")
        print(" 3. Ensure no other application (Zoom, Teams, Browser) is using the webcam.")
        return

    selected_index = index if index is not None else devices[0][0]
    print(f"\nOpening Camera Index {selected_index}...")

    cap = cv2.VideoCapture(selected_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(selected_index)

    if not cap.isOpened():
        print(f"ERROR: Could not open Camera {selected_index}")
        return

    print(f"Camera {selected_index} stream started. Press 'q' in the video window to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read frame from camera.")
            break

        cv2.putText(
            frame,
            f"Camera Index: {selected_index} (Press 'q' to quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(f"FitVision AI - Camera {selected_index} Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test closed successfully.")

if __name__ == "__main__":
    idx_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test_camera(idx_arg)
