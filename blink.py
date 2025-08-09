import cv2
import mediapipe as mp
import numpy as np
import time

# ---------- Mediapipe Setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR calculation
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array(landmarks[eye_indices[1]])
    p2 = np.array(landmarks[eye_indices[5]])
    p3 = np.array(landmarks[eye_indices[2]])
    p4 = np.array(landmarks[eye_indices[4]])
    p5 = np.array(landmarks[eye_indices[0]])
    p6 = np.array(landmarks[eye_indices[3]])
    A = np.linalg.norm(p2 - p1)
    B = np.linalg.norm(p4 - p3)
    C = np.linalg.norm(p6 - p5)
    return (A + B) / (2.0 * C)

def main():
    # ---------- Force External Webcam ----------
    camera_index = 1  # Change to 0 if needed
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    # Set safe resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}")
        return

    # ---------- Blink Variables ----------
    blink_count = 0
    blink_state = "open"
    frames_closed = 0
    calibration_frames = 30
    ear_values = []
    last_center = None
    movement_tolerance = 5

    prev_time = 0
    print("[INFO] Starting blink detection on external webcam. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

                left_ear = eye_aspect_ratio(points, LEFT_EYE)
                right_ear = eye_aspect_ratio(points, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                # Movement detection
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                face_center = (np.mean(x_coords), np.mean(y_coords))
                movement = np.linalg.norm(np.array(face_center) - np.array(last_center)) if last_center else 0
                last_center = face_center

                # Draw eyes
                for idx in LEFT_EYE + RIGHT_EYE:
                    cx, cy = int(points[idx][0]), int(points[idx][1])
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 255), -1)

                # Calibration
                if calibration_frames > 0:
                    ear_values.append(ear)
                    calibration_frames -= 1
                    cv2.putText(frame, "Calibrating...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue
                else:
                    open_eye_ear = np.mean(ear_values)
                    blink_close_thresh = open_eye_ear * 0.75
                    blink_open_thresh = open_eye_ear * 0.90

                # Blink detection
                if movement < movement_tolerance:
                    if blink_state == "open" and ear < blink_close_thresh:
                        frames_closed += 1
                        if frames_closed >= 2:
                            blink_state = "closed"
                            blink_count += 1
                            print(f"[INFO] Blink #{blink_count}")
                    elif blink_state == "closed" and ear > blink_open_thresh:
                        blink_state = "open"
                        frames_closed = 0
                else:
                    frames_closed = 0  # reset if head moves

                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # FPS display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Blink Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program finished.")

if __name__ == "__main__":
    main()