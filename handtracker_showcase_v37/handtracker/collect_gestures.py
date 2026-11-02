"""
collect_gestures.py
-------------------
Simple tool to collect labeled hand-landmark data for training a custom
ML gesture classifier later (Milestone 5).

Usage:
    python collect_gestures.py

Controls while running:
    1–7   Label the current hand pose with that gesture
    s     Save the current sample
    q     Quit and write the CSV

This produces a CSV you can later feed into a scikit-learn classifier
(kNN / Random Forest / SVM) to replace or complement the rule-based +
MediaPipe AI recognizer.
"""

import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

import config

LABELS = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "pointing",
    ord("4"): "two_finger",
    ord("5"): "thumb_up",
    ord("6"): "thumb_down",
    ord("7"): "pinch",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, f"gestures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


def landmarks_to_feature_vector(hand_landmarks):
    """Flatten 21 landmarks (x,y,z) into a 63-dim vector, then normalize
    relative to wrist so the features are roughly translation-invariant."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = pts[0]
    pts = pts - wrist  # relative to wrist
    # Scale by hand size (distance wrist → middle MCP)
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale
    return pts.flatten().tolist()


def main():
    print("Gesture Data Collector")
    print("Keys: 1=open_palm 2=fist 3=pointing 4=two_finger")
    print("      5=thumb_up 6=thumb_down 7=pinch   S=save sample   Q=quit")
    print(f"Samples will be written to: {OUT_FILE}\n")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    current_label = None
    samples = []
    last_save_msg = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            feature = None
            if results.multi_hand_landmarks:
                hl = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                feature = landmarks_to_feature_vector(hl)

            # HUD
            cv2.putText(frame, f"Label: {current_label or '(none – press 1-7)'}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
            cv2.putText(frame, f"Samples collected: {len(samples)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if last_save_msg:
                cv2.putText(frame, last_save_msg, (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "S = save sample   Q = quit & write CSV",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (180, 180, 180), 1)

            cv2.imshow("Collect Gestures", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in LABELS:
                current_label = LABELS[key]
                last_save_msg = f"Ready to save '{current_label}'"
            elif key == ord("s") and current_label and feature is not None:
                samples.append([current_label] + feature)
                last_save_msg = f"Saved #{len(samples)}  ({current_label})"
                print(last_save_msg)
            elif key == ord("s"):
                last_save_msg = "Need a hand in view AND a label selected"
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if samples:
        header = ["label"] + [f"f{i}" for i in range(63)]
        with open(OUT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(samples)
        print(f"\nWrote {len(samples)} samples → {OUT_FILE}")
    else:
        print("\nNo samples collected.")


if __name__ == "__main__":
    main()
