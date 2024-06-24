import cv2
import mediapipe as mp

# Initialize hand landmarks detector
mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define screen width and height (replace with your screen resolution)
screen_width = 1280
screen_height = 720

# Define fingertip landmarks (based on hand landmark connections)
fingertips = [mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
              mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

def detect_fingers(image, results):
  """Detects fingers and displays fingertip positions and gestures (like/dislike).

  Args:
    image: The input image frame.
    results: The MediaPipe hand landmarks detection results.
  """

  # Create empty finger fold status array
  finger_fold_status = []

  # Iterate through fingertips and draw circles
  for landmark in fingertips:
    x = int(results.multi_hand_landmarks[0].landmark[landmark].x * screen_width)
    y = int(results.multi_hand_landmarks[0].landmark[landmark].y * screen_height)
    cv2.circle(image, (x, y), 5, (255, 0, 0), cv2.FILLED)  # Blue circle for fingertips

    # Check if finger is folded (thumb is handled separately later)
    if landmark != mp_hands.HandLandmark.THUMB_TIP:  # Skip thumb for now
      # Access previous landmark (e.g., base of the finger)
      base_x = int(results.multi_hand_landmarks[0].landmark[landmark - 2].x * screen_width)
      base_y = int(results.multi_hand_landmarks[0].landmark[landmark - 2].y * screen_height)

      # Check if fingertip X is smaller than base X (indicating a fold)
      if x < base_x:
        cv2.circle(image, (x, y), 5, (0, 255, 0), cv2.FILLED)  # Green circle for folded finger
        finger_fold_status.append(True)
      else:
        finger_fold_status.append(False)

  # Check for like gesture (all fingers folded and thumb up)
  if all(finger_fold_status) and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].y < results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y:
    print("Like!")
    cv2.putText(image, "LIKE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  # Check for dislike gesture (all fingers folded and thumb down)
  if all(finger_fold_status) and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].y > results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y:
    print("Dislike!")
    cv2.putText(image, "DISLIKE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

  # Count the number of extended fingers (excluding thumb)
  num_extended_fingers = sum(not status for status in finger_fold_status[1:])
  cv2.putText(image, f"Fingers: {num_extended_fingers}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

  # Draw hand connections if a hand is detected
  if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)