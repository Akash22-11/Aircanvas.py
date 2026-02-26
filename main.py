import cv2
import numpy as np
import mediapipe as mp
import sys


# --- Configuration ---
width, height = 1280, 720
brush_thickness = 15
eraser_thickness = 50
draw_color = (255, 0, 255)
is_eraser = False  # Track eraser mode separately


# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


# --- Setup Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Warning: Camera at index 0 failed. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open any webcam.")
        print("1. Check if your webcam is plugged in.")
        print("2. Check if another app (Zoom/Teams) is using the camera.")
        sys.exit()

cap.set(3, width)
cap.set(4, height)

img_canvas = np.zeros((height, width, 3), np.uint8)
xp, yp = 0, 0

print("SUCCESS: Camera started!")
print("Press 'q' to Quit, 'c' to Clear")

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame from camera.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)


    # --- UI Toolbar ---
    cv2.rectangle(img, (40, 10), (140, 100), (255, 0, 0), -1)    # Blue
    cv2.rectangle(img, (160, 10), (260, 100), (0, 255, 0), -1)   # Green
    cv2.rectangle(img, (275, 10), (375, 100), (0, 0, 255), -1)   # Red
    cv2.rectangle(img, (390, 10), (490, 100), (50, 50, 50), -1)  
    cv2.putText(img, "ERASER", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Highlight selected color/tool in toolbar
    color_ranges = [
        ((40, 10), (140, 100), (255, 0, 0)),
        ((160, 10), (260, 100), (0, 255, 0)),
        ((275, 10), (375, 100), (0, 0, 255)),
        ((390, 10), (490, 100), (50, 50, 50)),  
    ]
    for i, ((x1r, y1r), (x2r, y2r), color) in enumerate(color_ranges):
        # Highlight eraser box if eraser mode is on, else highlight by draw_color match
        if i == 3 and is_eraser:
            cv2.rectangle(img, (x1r - 3, y1r - 3), (x2r + 3, y2r + 3), (255, 255, 255), 3)
        elif i != 3 and not is_eraser and draw_color == color:
            cv2.rectangle(img, (x1r - 3, y1r - 3), (x2r + 3, y2r + 3), (255, 255, 255), 3)

    # --- Hand Detection ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1], lm_list[8][2]    
                x2, y2 = lm_list[12][1], lm_list[12][2]  

                # Check which fingers are up
                fingers = []
                fingers.append(1 if lm_list[8][2] < lm_list[6][2] else 0)   
                fingers.append(1 if lm_list[12][2] < lm_list[10][2] else 0)  

                # --- Selection Mode: Two fingers up ---
                if fingers[0] == 1 and fingers[1] == 1:
                    xp, yp = 0, 0
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
                    if y1 < 100:
                        if 40 < x1 < 140:
                            draw_color = (255, 0, 0)
                            is_eraser = False
                        elif 160 < x1 < 260:
                            draw_color = (0, 255, 0)
                            is_eraser = False
                        elif 275 < x1 < 375:
                            draw_color = (0, 0, 255)
                            is_eraser = False
                        elif 390 < x1 < 490:
                            is_eraser = True  

                # --- Drawing Mode: One finger up ---
                if fingers[0] == 1 and fingers[1] == 0:
                    active_color = (0, 0, 0) if is_eraser else draw_color
                    cv2.circle(img, (x1, y1), 15, active_color, cv2.FILLED)

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    thickness = eraser_thickness if is_eraser else brush_thickness

                    cv2.line(img, (xp, yp), (x1, y1), active_color, thickness)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), active_color, thickness)

                    xp, yp = x1, y1

    # --- Fix canvas size if webcam resolution differs ---
    if img.shape[:2] != img_canvas.shape[:2]:
        img_canvas = cv2.resize(img_canvas, (img.shape[1], img.shape[0]))

    # --- Merge canvas onto webcam feed ---
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    cv2.imshow("Air Canvas", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        img_canvas = np.zeros((height, width, 3), np.uint8)

cap.release()
cv2.destroyAllWindows()
