"""Face Touch Guard - detects face touching via webcam and plays an alert sound."""

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Config ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FALLBACK_SOUND = "/System/Library/Sound/Sosumi.aiff"
ALERT_SOUNDS = []
REWARD_SOUNDS = []
COOLDOWN_SECONDS = 1.0
REWARD_DELAY_SECONDS = 8.0
CONSECUTIVE_FRAMES_NEEDED = 2
FACE_BOX_MARGIN = 80  # pixels of padding around face bounding box
MIN_HAND_SIZE_PX = 80
MIN_HAND_DETECTION_CONFIDENCE = 0.80
MIN_HAND_TRACKING_CONFIDENCE = 0.80


# Fingertip landmark indices in MediaPipe Hands
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

# Model URLs and local paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
FACE_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Snapshot directory: user's Pictures/FaceTouch
SNAPSHOT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "FaceTouch")


def _populate_sounds(target_list, path, sound_type):
    target_list.extend(Path(path).glob("*.wav"))

    if not target_list:
        print(f"No {sound_type} sounds found")


def populate_reward_sounds(reward_path="./audio/good"):
    _populate_sounds(REWARD_SOUNDS, reward_path, "reward")


def populate_alert_sounds(alert_path="./audio/alert"):
    _populate_sounds(ALERT_SOUNDS, alert_path, "alert")


def get_random_sound(sound_list: list):
    if not sound_list:
        return None
    return str(np.random.choice(sound_list))


def download_models():
    """Download MediaPipe model files if not present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for path, url, name in [
        (FACE_MODEL, FACE_MODEL_URL, "face landmarker"),
        (HAND_MODEL, HAND_MODEL_URL, "hand landmarker"),
    ]:
        if not os.path.exists(path):
            print(f"Downloading {name} model...")
            urllib.request.urlretrieve(url, path)
            print(f"  -> {path}")


def get_face_bbox(face_landmarks, w, h):
    """Extract bounding box from face landmarks with margin."""
    xs = [lm.x * w for lm in face_landmarks]
    ys = [lm.y * h for lm in face_landmarks]
    x_min = int(min(xs)) - FACE_BOX_MARGIN
    x_max = int(max(xs)) + FACE_BOX_MARGIN
    y_min = int(min(ys)) - FACE_BOX_MARGIN
    y_max = int(max(ys)) + FACE_BOX_MARGIN
    return max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)


def get_fingertips(hand_landmarks, w, h):
    """Get pixel coordinates of all fingertips."""
    tips = []
    for tip_id in FINGERTIP_IDS:
        lm = hand_landmarks[tip_id]
        tips.append((int(lm.x * w), int(lm.y * h)))
    return tips


def is_point_in_box(point, bbox):
    """Check if a point (x, y) is inside a bounding box (x1, y1, x2, y2)."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def play_alert(sound_file=None):
    """Play alert sound asynchronously, cross-platform."""
    src = sound_file or FALLBACK_SOUND
    if sys.platform.startswith("win"):
        try:
            import winsound

            if src and os.path.exists(src):
                winsound.PlaySound(
                    src, winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            else:
                # fallback short async beep (winsound.Beep blocks; prefer PlaySound with None)
                winsound.MessageBeep(winsound.SystemHand)
        except Exception:
            print("\a", end="", flush=True)
    elif sys.platform == "darwin":
        subprocess.Popen(
            ["afplay", src],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        for player in ("aplay", "paplay", "play"):
            if shutil.which(player):
                subprocess.Popen(
                    [player, src],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                break
        else:
            print("\a", end="", flush=True)


def _play_reward_internal(src):
    """Internal: play reward (called in background thread)."""
    if sys.platform.startswith("win"):
        try:
            import winsound

            if src and os.path.exists(src):
                winsound.PlaySound(
                    src, winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            else:
                winsound.MessageBeep(winsound.MB_ICONINFORMATION)
        except Exception:
            print("\a", end="", flush=True)
    elif sys.platform == "darwin":
        subprocess.Popen(
            ["afplay", src],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        for player in ("aplay", "paplay", "play"):
            if shutil.which(player):
                subprocess.Popen(
                    [player, src],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                break
        else:
            print("\a", end="", flush=True)


def play_reward(sound_file=None):
    """Kick off reward sound in a background thread (non-blocking)."""
    src = sound_file or FALLBACK_SOUND
    t = threading.Thread(
        target=_play_reward_internal, args=(src,), daemon=True
    )
    t.start()


def draw_debug(frame, face_bbox, fingertips, touching):
    """Draw debug overlay: face box, fingertips, and touch status."""
    x1, y1, x2, y2 = face_bbox
    color = (0, 0, 255) if touching else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for tip in fingertips:
        cv2.circle(frame, tip, 6, (255, 0, 255), -1)

    status = "TOUCHING!" if touching else "OK"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main():
    download_models()
    populate_alert_sounds()
    populate_reward_sounds()

    # Ensure snapshot directory exists (do this once before the loop)
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create snapshot dir {SNAPSHOT_DIR}: {e}")
        quit()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create detectors using Tasks API
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_HAND_TRACKING_CONFIDENCE,
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    consecutive_touch_frames = 0
    last_alert_time = 0.0
    show_debug = False
    paused = False
    touch_count = 0
    frame_ts = 0
    was_touching = False
    reward_timer = None

    print(
        "Face Touch Guard running. Press 'q' to quit, 'd' for debug, SPACE to pause."
    )

    while True:
        # When paused, release camera and wait for key
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = False
                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                print("  Resumed — camera on")
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        h, w = frame.shape[:2]

        touching_this_frame = False
        face_bbox = None
        all_fingertips = []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33  # ~30fps in milliseconds

        face_result = face_detector.detect_for_video(mp_image, frame_ts)
        hand_result = hand_detector.detect_for_video(mp_image, frame_ts)

        if face_result.face_landmarks and hand_result.hand_landmarks:
            face_bbox = get_face_bbox(face_result.face_landmarks[0], w, h)

            for hand_lms in hand_result.hand_landmarks:
                # ignore tiny detections (e.g., ears) by checking wrist -> index tip distance
                wrist = hand_lms[0]
                index_tip = hand_lms[8]
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                index_px = (int(index_tip.x * w), int(index_tip.y * h))
                hand_size = np.hypot(
                    wrist_px[0] - index_px[0], wrist_px[1] - index_px[1]
                )
                if hand_size < MIN_HAND_SIZE_PX:
                    continue

                tips = get_fingertips(hand_lms, w, h)
                all_fingertips.extend(tips)

                for tip in tips:
                    if is_point_in_box(tip, face_bbox):
                        touching_this_frame = True
                        break

        # Consecutive frame logic
        if touching_this_frame:
            consecutive_touch_frames += 1
        else:
            consecutive_touch_frames = 0

        # region Reward sound logic

        if was_touching and not touching_this_frame:
            sound = get_random_sound(REWARD_SOUNDS)
            # cancel any existing pending reward and start a new timer
            if reward_timer is not None:
                try:
                    reward_timer.cancel()
                except Exception:
                    pass
            reward_timer = threading.Timer(
                REWARD_DELAY_SECONDS, play_reward, args=(sound,)
            )
            reward_timer.daemon = True
            reward_timer.start()

        # if the user touches again before the reward timer fires, cancel the pending reward
        if touching_this_frame and reward_timer is not None:
            try:
                reward_timer.cancel()
            except Exception:
                pass
            reward_timer = None

        was_touching = touching_this_frame

        # region Alert sound logic

        # Alert if enough consecutive frames and cooldown elapsed
        now = time.time()
        if (
            consecutive_touch_frames >= CONSECUTIVE_FRAMES_NEEDED
            and now - last_alert_time > COOLDOWN_SECONDS
        ):
            sound = get_random_sound(ALERT_SOUNDS)
            play_alert(sound)
            last_alert_time = now
            touch_count += 1
            print(f"Face touch detected! (total: {touch_count})")

            # Save snapshot with debug overlay (always include debug marks on snapshot)
            try:
                snapshot = frame.copy()
                if face_bbox:
                    draw_debug(snapshot, face_bbox, all_fingertips, True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                fname = f"facetouch_{timestamp}.jpg"
                path = os.path.join(SNAPSHOT_DIR, fname)
                cv2.imwrite(path, snapshot)
            except Exception as e:
                print(f"Failed to save snapshot: {e}")

        # Draw debug overlay
        if show_debug and face_bbox:
            is_alert = consecutive_touch_frames >= CONSECUTIVE_FRAMES_NEEDED
            draw_debug(frame, face_bbox, all_fingertips, is_alert)
        elif show_debug:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        cv2.imshow("Face Touch Guard", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord(" "):
            paused = True
            consecutive_touch_frames = 0
            cap.release()
            cv2.destroyAllWindows()
            # Show a small window for key input while paused
            pause_img = np.zeros(
                (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
            )
            cv2.putText(
                pause_img,
                "PAUSED",
                (FRAME_WIDTH // 2 - 100, FRAME_HEIGHT // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                pause_img,
                "SPACE to resume | q to quit",
                (FRAME_WIDTH // 2 - 170, FRAME_HEIGHT // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (150, 150, 150),
                1,
            )
            cv2.imshow("Face Touch Guard", pause_img)
            print("  Paused — camera off. Press SPACE to resume.")

    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()
    hand_detector.close()
    print(f"\nSession ended. Total face touches detected: {touch_count}")


if __name__ == "__main__":
    main()
