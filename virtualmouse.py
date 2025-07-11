import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import sys
import traceback

# ✅ Global crash logger — logs to error_log.txt
def log_exceptions(exctype, value, tb):
    with open("error_log.txt", "w") as f:
        traceback.print_exception(exctype, value, tb, file=f)
    print("❌ An error occurred. See error_log.txt")
    input("Press Enter to exit...")

sys.excepthook = log_exceptions

def main():
    # ✅ Setup camera with higher resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ✅ MediaPipe hand detection with wide detection tolerance
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()

    print("✅ Started Virtual Mouse...")

    # App state
    left_clicking = False
    right_clicking = False
    prev_x, prev_y = 0, 0
    smoothening = 7
    click_threshold = 35
    click_delay = 0.5
    last_left_click_time = 0
    last_right_click_time = 0
    cursor_locked = False
    fist_closed = False

    def toggle_cursor_lock():
        nonlocal cursor_locked
        cursor_locked = not cursor_locked
        print(f"🖱️ Cursor movement: {'LOCKED' if cursor_locked else 'UNLOCKED'}")

    def is_fist(landmarks, frame_width, frame_height):
        tip_ids = [8, 12, 16, 20]
        folded_fingers = 0
        for tip_id in tip_ids:
            tip_y = landmarks[tip_id].y * frame_height
            lower_joint_y = landmarks[tip_id - 2].y * frame_height
            if tip_y > lower_joint_y:
                folded_fingers += 1
        return folded_fingers == 4

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb_frame)
        frame_height, frame_width, _ = frame.shape

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # Key points
            thumb_x = int(landmarks[4].x * frame_width)
            thumb_y = int(landmarks[4].y * frame_height)
            index_x = int(landmarks[8].x * frame_width)
            index_y = int(landmarks[8].y * frame_height)
            middle_x = int(landmarks[12].x * frame_width)
            middle_y = int(landmarks[12].y * frame_height)

            # 👊 Fist detection to toggle lock
            if is_fist(landmarks, frame_width, frame_height):
                if not fist_closed:
                    fist_closed = True
            else:
                if fist_closed:
                    toggle_cursor_lock()
                    fist_closed = False

            # 🖱️ Cursor movement (controlled by thumb)
            if not cursor_locked:
                screen_x = np.interp(thumb_x, [0, frame_width], [0, screen_width])
                screen_y = np.interp(thumb_y, [0, frame_height], [0, screen_height])
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

            # 📏 Distance between fingers
            dist_index = math.hypot(thumb_x - index_x, thumb_y - index_y)
            dist_middle = math.hypot(thumb_x - middle_x, thumb_y - middle_y)
            current_time = time.time()

            # 👈 Left Click (Thumb + Index)
            if dist_index < click_threshold:
                if not left_clicking and (current_time - last_left_click_time) > click_delay:
                    pyautogui.mouseDown()
                    left_clicking = True
                    last_left_click_time = current_time
                    cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 255, 0), -1)
            else:
                if left_clicking:
                    pyautogui.mouseUp()
                    left_clicking = False
                    last_left_click_time = current_time

            # 👉 Right Click (Thumb + Middle)
            if dist_middle < click_threshold:
                if not right_clicking and (current_time - last_right_click_time) > click_delay:
                    pyautogui.mouseDown(button='right')
                    right_clicking = True
                    last_right_click_time = current_time
                    cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 0, 255), -1)
            else:
                if right_clicking:
                    pyautogui.mouseUp(button='right')
                    right_clicking = False
                    last_right_click_time = current_time

        # Show cursor lock state
        cv2.putText(frame, f'Cursor: {"LOCKED" if cursor_locked else "UNLOCKED"}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Virtual Mouse", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🟢 Safe entry point
if __name__ == "__main__":
    main()
