"""
AI Smart Classroom Analyzer - Backend Only (No UI)
Processes webcam feed and saves engagement metrics to JSON/CSV

Features:
- Real-time webcam processing
- Uses YOLO for person/phone detection
- Uses MediaPipe for face/pose/hand landmarks
- Detects: attentive, sleeping, using_phone, hand_raise
- Saves metrics to engagement_timeline.csv and realtime_data.json
"""

import cv2
import numpy as np
import time
import json
import csv
import math
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
from datetime import datetime

# ===== Configuration =====
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE = 0.35
EYE_CLOSED_THRESHOLD = 0.20
EYE_CLOSED_SECS = 2.0
HAND_RAISE_OFFSET = -0.15
PHONE_DISTANCE_THRESH = 0.25

# ===== Initialize Models =====
print("[*] Loading YOLO model...")
yolo = YOLO(YOLO_MODEL)

print("[*] Initializing MediaPipe...")
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, min_tracking_confidence=0.4)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.4)

# ===== Utility Functions =====
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def euclidean(a, b):
    """Calculate Euclidean distance"""
    return math.dist(a, b)

def eye_aspect_ratio(eye_coords):
    """Calculate eye aspect ratio (EAR)"""
    A = euclidean(eye_coords[1], eye_coords[5])
    B = euclidean(eye_coords[2], eye_coords[4])
    C = euclidean(eye_coords[0], eye_coords[3])
    if C == 0:
        return 0
    ear = (A + B) / (2.0 * C)
    return ear

class CentroidTracker:
    """Simple centroid-based object tracker"""
    def _init_(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array(rects)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(tuple(input_centroids[i]))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            assigned_rows = set()
            assigned_cols = set()

            for (r, c) in zip(rows, cols):
                if r in assigned_rows or c in assigned_cols:
                    continue
                object_id = object_ids[r]
                self.objects[object_id] = tuple(input_centroids[c])
                self.disappeared[object_id] = 0
                assigned_rows.add(r)
                assigned_cols.add(c)

            unassigned_rows = set(range(0, D.shape[0])).difference(assigned_rows)
            for r in unassigned_rows:
                object_id = object_ids[r]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unassigned_cols = set(range(0, D.shape[1])).difference(assigned_cols)
            for c in unassigned_cols:
                self.register(tuple(input_centroids[c]))

        return self.objects

# ===== Main Analyzer Class =====
class ClassroomAnalyzerBackend:
    def _init_(self):
        self.yolo = yolo
        self.ear_thresh = EYE_CLOSED_THRESHOLD
        self.ear_secs = EYE_CLOSED_SECS
        self.hand_offset = HAND_RAISE_OFFSET
        self.phone_dist = PHONE_DISTANCE_THRESH

        self.face_mesh = face_mesh
        self.pose = pose
        self.hands = hands
        self.tracker = CentroidTracker(max_disappeared=50)
        self.eye_closed_start = dict()
        
        # Timeline storage
        self.timeline = deque(maxlen=3600)
        self.csv_file = "engagement_timeline.csv"
        self.json_file = "realtime_data.json"
        
        # Initialize CSV file with headers
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'datetime', 'total_students', 'attentive', 'sleeping', 'using_phone', 'hand_raise', 'engagement_score'])
            writer.writeheader()

    def analyze_frame(self, frame, timestamp=None):
        """Analyze a single frame and return annotated frame + stats"""
        h, w = frame.shape[:2]
        results = self.yolo.predict(frame, imgsz=640, conf=CONFIDENCE, verbose=False)
        detections = results[0]

        persons = []
        phones = []
        
        for box, cls in zip(detections.boxes.xyxy, detections.boxes.cls):
            x1, y1, x2, y2 = [int(x) for x in box]
            label = self.yolo.model.names[int(cls)] if hasattr(self.yolo, 'model') else str(int(cls))
            if label.lower() in ['person', 'people'] or int(cls) == 0:
                persons.append((x1, y1, x2, y2))
                # Draw person bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if label.lower() in ['cell phone', 'phone', 'mobile phone'] or int(cls) in [67, 77]:
                phones.append((x1, y1, x2, y2))
                # Draw phone bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Phone", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        annotated = frame.copy()
        centroids = []
        for (x1, y1, x2, y2) in persons:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids.append((cx, cy))

        objects = self.tracker.update(centroids)

        # Process MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face_results = self.face_mesh.process(rgb)
            pose_results = self.pose.process(rgb)
            hands_results = self.hands.process(rgb)
        except Exception as e:
            print(f"[!] MediaPipe error: {e}")
            return annotated, self._default_stats(timestamp)

        id_to_status = dict()
        for oid in objects.keys():
            id_to_status[oid] = {'attentive': False, 'sleeping': False, 'using_phone': False, 'hand_raise': False}

        # Analyze faces
        face_landmarks_all = []
        if face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                lm_coords = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms.landmark]
                face_landmarks_all.append(lm_coords)

        pose_landmarks = None
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark

        hands_list = []
        if hands_results.multi_hand_landmarks:
            for hand_lms in hands_results.multi_hand_landmarks:
                hand_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
                hands_list.append(hand_coords)

        # Process face landmarks
        now = timestamp if timestamp is not None else time.time()
        for lm_coords in face_landmarks_all:
            fx = int(np.mean([p[0] for p in lm_coords]))
            fy = int(np.mean([p[1] for p in lm_coords]))
            
            best_id = None
            best_dist = float('inf')
            for oid, centroid in objects.items():
                d = math.hypot(centroid[0] - fx, centroid[1] - fy)
                if d < best_dist:
                    best_dist = d
                    best_id = oid

            if best_id is None:
                continue

            try:
                left_eye = [lm_coords[i] for i in LEFT_EYE_IDX]
                right_eye = [lm_coords[i] for i in RIGHT_EYE_IDX]
            except:
                left_eye, right_eye = None, None

            ear = 1.0
            if left_eye and right_eye:
                ear_l = eye_aspect_ratio(left_eye)
                ear_r = eye_aspect_ratio(right_eye)
                ear = (ear_l + ear_r) / 2.0

            # Sleeping detection
            if ear < self.ear_thresh:
                if best_id not in self.eye_closed_start:
                    self.eye_closed_start[best_id] = now
                else:
                    elapsed = now - self.eye_closed_start[best_id]
                    if elapsed >= self.ear_secs:
                        id_to_status[best_id]['sleeping'] = True
            else:
                if best_id in self.eye_closed_start:
                    del self.eye_closed_start[best_id]

            if ear >= self.ear_thresh:
                id_to_status[best_id]['attentive'] = True

        # Hand raise detection
        if pose_landmarks:
            try:
                left_sh = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_sh = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_wr = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wr = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                l_sh = (int(left_sh.x * w), int(left_sh.y * h))
                r_sh = (int(right_sh.x * w), int(right_sh.y * h))
                l_wr = (int(left_wr.x * w), int(left_wr.y * h))
                r_wr = (int(right_wr.x * w), int(right_wr.y * h))

                for oid, centroid in objects.items():
                    avg_sh_y = (l_sh[1] + r_sh[1]) / 2
                    if l_wr[1] < avg_sh_y + self.hand_offset * h or r_wr[1] < avg_sh_y + self.hand_offset * h:
                        id_to_status[oid]['hand_raise'] = True
            except:
                pass

        # Phone detection
        phone_centroids = []
        for (x1, y1, x2, y2) in phones:
            phone_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1))

        for oid, centroid in objects.items():
            cx, cy = centroid
            for (px, py, pw, ph) in phone_centroids:
                d = math.hypot(px - cx, py - cy)
                if pw == 0:
                    continue
                if d < max(pw, ph) * 1.2:
                    id_to_status[oid]['using_phone'] = True

        # Aggregate stats
        total = max(1, len(objects))
        counts = {'attentive': 0, 'sleeping': 0, 'using_phone': 0, 'hand_raise': 0}
        for oid, st in id_to_status.items():
            for k in counts.keys():
                if st.get(k):
                    counts[k] += 1

        attentive = counts['attentive']
        distracted = counts['sleeping'] + counts['using_phone']
        score = int(100 * (attentive - distracted * 0.5) / total)
        score = max(0, min(100, score))

        stats = {
            'total_students': total,
            'attentive': attentive,
            'sleeping': counts['sleeping'],
            'using_phone': counts['using_phone'],
            'hand_raise': counts['hand_raise'],
            'engagement_score': score,
            'timestamp': timestamp if timestamp is not None else time.time()
        }

        # Draw engagement metrics on frame
        y_offset = 30
        cv2.putText(frame, f"Engagement: {score}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        y_offset += 40
        cv2.putText(frame, f"Students: {total}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        y_offset += 35
        cv2.putText(frame, f"Attentive: {attentive}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 35
        cv2.putText(frame, f"Sleeping: {counts['sleeping']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 35
        cv2.putText(frame, f"Phone Use: {counts['using_phone']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        y_offset += 35
        cv2.putText(frame, f"Hand Raised: {counts['hand_raise']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return frame, stats

    def _default_stats(self, timestamp):
        """Return default stats when processing fails"""
        return {
            'total_students': 0,
            'attentive': 0,
            'sleeping': 0,
            'using_phone': 0,
            'hand_raise': 0,
            'engagement_score': 0,
            'timestamp': timestamp if timestamp is not None else time.time()
        }

    def save_stats(self, stats):
        """Save stats to CSV and JSON"""
        self.timeline.append(stats)
        
        # Save to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'datetime', 'total_students', 'attentive', 'sleeping', 'using_phone', 'hand_raise', 'engagement_score'])
            writer.writerow({
                'timestamp': stats['timestamp'],
                'datetime': datetime.fromtimestamp(stats['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                'total_students': stats['total_students'],
                'attentive': stats['attentive'],
                'sleeping': stats['sleeping'],
                'using_phone': stats['using_phone'],
                'hand_raise': stats['hand_raise'],
                'engagement_score': stats['engagement_score']
            })
        
        # Save to JSON (realtime)
        json_data = {
            'last_update': datetime.now().isoformat(),
            'current_stats': stats,
            'summary': {
                'total_frames': len(self.timeline),
                'avg_engagement': np.mean([s['engagement_score'] for s in self.timeline]) if self.timeline else 0
            }
        }
        with open(self.json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

    def print_stats(self, stats, fps=0):
        """Print stats to console"""
        print(f"[{datetime.fromtimestamp(stats['timestamp']).strftime('%H:%M:%S')}] "
              f"Students: {stats['total_students']} | "
              f"Attentive: {stats['attentive']} | "
              f"Sleeping: {stats['sleeping']} | "
              f"Phone: {stats['using_phone']} | "
              f"Hand Raised: {stats['hand_raise']} | "
              f"Engagement: {stats['engagement_score']}% | "
              f"FPS: {fps:.1f}")

# ===== Main Processing Loop =====
def main():
    print("[*] Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[!] ERROR: Cannot open webcam. Check connection and permissions.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[*] Webcam opened successfully!")
    print("[*] Saving results to: engagement_timeline.csv and realtime_data.json")
    print("[*] Camera window opened - Press 'q' to quit, 's' for screenshot...\n")
    
    analyzer = ClassroomAnalyzerBackend()
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame")
                break
            
            ts = time.time()
            display_frame, stats = analyzer.analyze_frame(frame, timestamp=ts)
            analyzer.save_stats(stats)
            
            # Display the frame with annotations
            cv2.imshow("Smart Classroom Analyzer - Press 'q' to quit", display_frame)
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                analyzer.print_stats(stats, fps)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[*] Quit requested...")
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, display_frame)
                print(f"[+] Screenshot saved: {screenshot_name}")
    
    except KeyboardInterrupt:
        print("\n\n[*] Stopping analysis...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final summary
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Analysis Complete!")
        print(f"{'='*80}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count / total_time:.2f}")
        print(f"Results saved to:")
        print(f"  - {analyzer.csv_file}")
        print(f"  - {analyzer.json_file}")
        print(f"{'='*80}\n")

if __name__ == "_main_":
    main()