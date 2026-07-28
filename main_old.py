import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None

prev_x = 0
prev_y = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        print("Hands:", len(results.multi_hand_landmarks))

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            if prev_x == 0 and prev_y == 0:
                prev_x = x
                prev_y = y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 255), 5)

            prev_x = x
            prev_y = y

    else:
        prev_x = 0
        prev_y = 0

    frame = cv2.add(frame, canvas)

    cv2.imshow("AI Air Canvas", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        canvas = np.zeros_like(frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()