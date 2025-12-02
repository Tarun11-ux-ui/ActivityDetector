import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time


print("Loading YOLO model...")
model = YOLO("yolov8n.pt")


mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    min_detection_confidence=0.4
)

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam. Please check:")
    print("1. Camera is connected to the computer")
    print("2. No other app is using the camera")
    print("3. Camera permissions are granted")
    exit(1)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam opened successfully!")
print("Press 'q' to quit, 's' to take a screenshot")

def is_sleeping(landmarks):
    """Detect if head is down -> sleeping"""
    try:
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        
        # If nose is much lower than eyes -> head is down
        return nose.y > max(left_eye.y, right_eye.y) + 0.05
    except:
        return False

def is_hand_raised(landmarks):
    """Detect if hand is raised above shoulder"""
    try:
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Check if either wrist is above corresponding shoulder
        left_raised = left_wrist.y < left_shoulder.y - 0.1
        right_raised = right_wrist.y < right_shoulder.y - 0.1
        
        return left_raised or right_raised
    except:
        return False

def detect_phone_use(frame, yolo_results):
    """Check if anyone is using a phone"""
    phone_count = 0
    for box in yolo_results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        
        if label.lower() in ["cell phone", "phone", "mobile phone"] or cls == 67:
            phone_count += 1
    
    return phone_count

# Main loop
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame from camera")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # YOLO detection (people & phones)
        yolo_results = model(frame, verbose=False)
        
        total_students = 0
        attentive = 0
        sleeping = 0
        hand_raised = 0
        
        # Count persons detected
        for box in yolo_results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            
            if label.lower() in ["person", "people"] or cls == 0:
                total_students += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Phone detection
        phone_usage = detect_phone_use(frame, yolo_results)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection (for hand raise and sleeping)
        pose_results = pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Draw pose skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Check for sleeping
            if is_sleeping(landmarks):
                sleeping += 1
            else:
                attentive += 1
            
            # Check for hand raised
            if is_hand_raised(landmarks):
                hand_raised += 1
        
        # Face detection (optional - for attention analysis)
        face_results = face_mesh.process(frame_rgb)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh (light drawing)
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
        
        # Hand detection (optional)
        hands_results = hands.process(frame_rgb)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1)
                )
        
        # Calculate engagement score
        engagement = 0
        if total_students > 0:
            engagement = int((attentive / max(1, total_students)) * 100)
        
        # Display statistics
        y_offset = 30
        cv2.putText(frame, f"FPS: {int(frame_count / (time.time() - start_time))}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Total Students: {total_students}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Attentive: {attentive}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame, f"Sleeping: {sleeping}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Phone Usage: {phone_usage}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        y_offset += 30
        cv2.putText(frame, f"Hand Raised: {hand_raised}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame, f"Engagement: {engagement}%", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        # Display frame
        cv2.imshow("Smart Classroom Detector", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")