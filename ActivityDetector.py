"""
AI Smart Classroom Analyzer - Streamlit App
Single-file prototype: ai_smart_classroom_analyzer.py

Features:
- Real-time webcam or uploaded video processing
- Uses YOLO (ultralytics) for person/phone detection
- Uses MediaPipe for face/pose/hand landmarks
- Rule engine to detect: attentive, sleeping, using_phone, hand_raise
- Engagement score computed per minute and shown as timeline
- Streamlit dashboard with controls and live annotated frames

Notes & Requirements:
- Python 3.8+
- pip install streamlit opencv-python-headless ultralytics mediapipe pandas numpy matplotlib
- For webcam support in Streamlit on some platforms, you may need streamlit-webrtc (optional)
- Model: ultralytics' YOLO will auto-download 'yolov8n.pt' if not present when using YOLO('yolov8n')

This is a prototype — for production use, add robust tracking (ByteTrack/DeepSORT), privacy safeguards,
proper model optimization, and deployment details.
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from ultralytics import YOLO
import mediapipe as mp
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timedelta
import tempfile
import os
import math

# -----------------------------
# Helpers & Config
# -----------------------------

st.set_page_config(page_title="AI Smart Classroom Analyzer", layout="wide")

@st.cache_resource
def load_yolo(model_name="yolov8n.pt"):
    # ultralyticsYOLO will download if not present
    return YOLO(model_name)

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Constants for rule thresholds (tweakable in UI)
DEFAULTS = {
    'eye_closed_ear_thresh': 0.20,  # eye aspect ratio threshold
    'eye_closed_secs': 2.0,
    'hand_raise_height_offset': -0.15,  # relative to shoulder vertical difference
    'phone_distance_thresh': 0.25,  # fraction of bbox width
}

# Utility: Eye Aspect Ratio using Mediapipe face mesh landmarks
# We'll pick a few landmarks for left/right eye
# Mediapipe face mesh indices: use approximate indices for eyelids
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]  # approximate
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Simple centroid tracker for short-term identity mapping
class CentroidTracker:
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
        # rects: list of centroids
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

# Eye aspect ratio

def euclidean(a, b):
    return math.dist(a, b)


def eye_aspect_ratio(eye_coords):
    # eye_coords: list of 6 (x,y)
    A = euclidean(eye_coords[1], eye_coords[5])
    B = euclidean(eye_coords[2], eye_coords[4])
    C = euclidean(eye_coords[0], eye_coords[3])
    # avoid division by zero
    if C == 0:
        return 0
    ear = (A + B) / (2.0 * C)
    return ear

# -----------------------------
# Detection + Analysis Pipeline
# -----------------------------

class ClassroomAnalyzer:
    def _init_(self, yolo_model, ear_thresh=DEFAULTS['eye_closed_ear_thresh'], ear_secs=DEFAULTS['eye_closed_secs'], hand_offset=DEFAULTS['hand_raise_height_offset'], phone_dist=DEFAULTS['phone_distance_thresh']):
        self.yolo = yolo_model
        self.ear_thresh = ear_thresh
        self.ear_secs = ear_secs
        self.hand_offset = hand_offset
        self.phone_dist = phone_dist

        # MediaPipe solutions
        self.face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.4)

        # Tracking
        self.tracker = CentroidTracker(max_disappeared=50)
        # for each tracked id, store last EAR timestamps when closed
        self.eye_closed_start = dict()
        # history for timeline
        self.timeline = deque(maxlen=3600)  # store per-second metrics for up to 1 hour

    def analyze_frame(self, frame, timestamp=None):
        # Returns annotated frame and a dict of stats for that frame
        h, w = frame.shape[:2]
        results = self.yolo.predict(frame, imgsz=640, conf=0.35, verbose=False)
        detections = results[0]

        persons = []
        phones = []
        for box, cls in zip(detections.boxes.xyxy, detections.boxes.cls):
            x1, y1, x2, y2 = [int(x) for x in box]
            label = self.yolo.model.names[int(cls)] if hasattr(self.yolo, 'model') else str(int(cls))
            if label.lower() in ['person', 'people'] or int(cls) == 0:
                persons.append((x1, y1, x2, y2))
            if label.lower() in ['cell phone', 'phone', 'mobile phone'] or int(cls) in [67, 77]:
                phones.append((x1, y1, x2, y2))

        # Prepare output map
        annotated = frame.copy()
        centroids = []
        for (x1, y1, x2, y2) in persons:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids.append((cx, cy))

        objects = self.tracker.update(centroids)

        # Run mediapipe face and pose on the whole image (cheap for small class sizes)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face_results = self.face_mesh.process(rgb)
            pose_results = self.pose.process(rgb)
            hands_results = self.hands.process(rgb)
        except Exception as e:
            st.error(f"MediaPipe processing error: {e}")
            return annotated, {'total_students': 0, 'attentive': 0, 'sleeping': 0, 'using_phone': 0, 'hand_raise': 0, 'engagement_score': 0, 'timestamp': time.time()}

        # Map face detections by approximate centroid to tracking IDs
        id_to_status = dict()
        # default status per id
        for oid in objects.keys():
            id_to_status[oid] = {'attentive': False, 'sleeping': False, 'using_phone': False, 'hand_raise': False}

        # Analyze faces for EAR (eyes) and gaze approx
        face_landmarks_all = []
        if face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                # landmark coords to image
                lm_coords = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms.landmark]
                face_landmarks_all.append(lm_coords)

        # analyze pose for hand raise using pose landmarks
        pose_landmarks = None
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark

        # analyze hands for phone proximity (approx)
        hands_list = []
        if hands_results.multi_hand_landmarks:
            for hand_lms in hands_results.multi_hand_landmarks:
                hand_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
                hands_list.append(hand_coords)

        # Basic association: for each face landmark set, find nearest tracker id
        for lm_coords in face_landmarks_all:
            fx = int(np.mean([p[0] for p in lm_coords]))
            fy = int(np.mean([p[1] for p in lm_coords]))
            # find nearest object id
            best_id = None
            best_dist = float('inf')
            for oid, centroid in objects.items():
                d = math.hypot(centroid[0] - fx, centroid[1] - fy)
                if d < best_dist:
                    best_dist = d
                    best_id = oid

            if best_id is None:
                continue

            # compute EAR using landmark indices
            try:
                left_eye = [lm_coords[i] for i in LEFT_EYE_IDX]
                right_eye = [lm_coords[i] for i in RIGHT_EYE_IDX]
            except Exception:
                left_eye, right_eye = None, None

            ear = 1.0
            if left_eye and right_eye:
                ear_l = eye_aspect_ratio(left_eye)
                ear_r = eye_aspect_ratio(right_eye)
                ear = (ear_l + ear_r) / 2.0

            # Update sleeping detection state
            now = timestamp if timestamp is not None else time.time()
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

            # Simple attentive heuristic: face roughly facing front if eye landmarks present and not sleeping
            if ear >= self.ear_thresh:
                id_to_status[best_id]['attentive'] = True

            # draw face box / landmarks
            for (x, y) in lm_coords[0:10]:
                cv2.circle(annotated, (x, y), 1, (0, 255, 0), -1)

        # Hand raise detection (use pose landmarks)
        if pose_landmarks:
            # get shoulders and wrists
            try:
                left_sh = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_sh = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_wr = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wr = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                # convert to pixel coords
                l_sh = (int(left_sh.x * w), int(left_sh.y * h))
                r_sh = (int(right_sh.x * w), int(right_sh.y * h))
                l_wr = (int(left_wr.x * w), int(left_wr.y * h))
                r_wr = (int(right_wr.x * w), int(right_wr.y * h))

                # For each tracked id, see if a wrist is above corresponding shoulder (simple)
                for oid, centroid in objects.items():
                    # compare vertical positions relative to centroid
                    # if wrist y is significantly above shoulder y -> hand raise
                    # normalize by image height
                    # We'll just check global: if either wrist y < shoulder y - offset
                    avg_sh_y = (l_sh[1] + r_sh[1]) / 2
                    if l_wr[1] < avg_sh_y + self.hand_offset * h or r_wr[1] < avg_sh_y + self.hand_offset * h:
                        id_to_status[oid]['hand_raise'] = True
            except Exception:
                pass

        # Using phone: check phone boxes near face centroids
        # Compute phone centroids
        phone_centroids = []
        for (x1, y1, x2, y2) in phones:
            phone_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1))

        for oid, centroid in objects.items():
            cx, cy = centroid
            # check phones close to this centroid
            for (px, py, pw, ph) in phone_centroids:
                d = math.hypot(px - cx, py - cy)
                # threshold by bbox width
                if pw == 0:
                    continue
                if d < max(pw, ph) * 1.2:  # near enough
                    id_to_status[oid]['using_phone'] = True

        # Aggregate stats
        total = max(1, len(objects))
        counts = {'attentive': 0, 'sleeping': 0, 'using_phone': 0, 'hand_raise': 0}
        for oid, st in id_to_status.items():
            for k in counts.keys():
                if st.get(k):
                    counts[k] += 1

        # compute engagement score (simple weighted)
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

        # Draw annotations: boxes for persons
        i = 0
        for (x1, y1, x2, y2) in persons:
            color = (0, 255, 0)
            # try find nearest id
            if i < len(objects):
                # can't rely on order; draw generic boxes
                pass
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            i += 1

        # Put text metrics
        cv2.putText(annotated, f"Engagement: {stats['engagement_score']}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(annotated, f"Total: {stats['total_students']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # push to timeline (per-second)
        ts = int(stats['timestamp'])
        self.timeline.append({'ts': ts, **stats})

        return annotated, stats

    def get_timeline_df(self):
        if len(self.timeline) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(list(self.timeline))
        df['dt'] = pd.to_datetime(df['ts'], unit='s')
        df = df.sort_values('ts')
        return df

# -----------------------------
# Streamlit App UI
# -----------------------------

st.title("⚡ AI Smart Classroom Analyzer — Student Engagement Detector")

with st.sidebar.expander("Model & Source Options", expanded=True):
    source_mode = st.radio("Video Source", ['Webcam (0)', 'Upload Video File'], index=0)
    conf = st.slider("YOLO Confidence", 0.1, 0.9, 0.35, 0.01)
    ear_thresh = st.slider("Eye aspect ratio (closed) threshold", 0.08, 0.35, DEFAULTS['eye_closed_ear_thresh'], 0.01)
    ear_secs = st.slider("Seconds eyes closed -> sleeping", 0.5, 5.0, DEFAULTS['eye_closed_secs'], 0.5)
    hand_offset = st.slider("Hand raise vertical offset (fraction of H)", -0.5, 0.0, DEFAULTS['hand_raise_height_offset'], 0.05)
    phone_dist = st.slider("Phone proximity threshold (fraction)", 0.05, 0.5, DEFAULTS['phone_distance_thresh'], 0.05)
    model_name = st.text_input("YOLO model name/path", value="yolov8n.pt")

start_button = st.button("Start Analysis")
stop_button = st.button("Stop")

# main area: video + metrics
col1, col2 = st.columns([2, 1])
video_placeholder = col1.empty()
metrics_placeholder = col2.empty()

# Load model (cached)
with st.spinner("Loading YOLO model..."):
    yolo = load_yolo(model_name)
    # set model conf
    try:
        yolo.model.conf = conf
    except Exception:
        pass

analyzer = ClassroomAnalyzer(yolo_model=yolo, ear_thresh=ear_thresh, ear_secs=ear_secs, hand_offset=hand_offset, phone_dist=phone_dist)

# Initialize session state for video capture
if 'processing' not in st.session_state:
    st.session_state.processing = False
    st.session_state.cap = None
    st.session_state.last_frame_time = 0

# Metrics display function

def show_metrics(stats):
    df = pd.DataFrame([stats])
    with metrics_placeholder.container():
        st.metric("Engagement %", f"{stats['engagement_score']}%")
        st.write(f"Total students (detected): {stats['total_students']}")
        st.write(f"Attentive: {stats['attentive']}  |  Sleeping: {stats['sleeping']}  |  On-phone: {stats['using_phone']}  |  Hand raises: {stats['hand_raise']}")
        # timeline small
        tdf = analyzer.get_timeline_df()
        if not tdf.empty:
            # resample per minute
            tdf2 = tdf.set_index('dt').resample('1T').mean()
            st.line_chart(tdf[['dt', 'engagement_score']].set_index('dt')['engagement_score'])

# Start / Stop handling

if start_button:
    if st.session_state.processing:
        st.warning("Already running")
    else:
        st.session_state.processing = True
        if source_mode == 'Upload Video File':
            up = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi', 'mkv'])
            if up is None:
                st.info("Please upload a video file before starting")
                st.session_state.processing = False
            else:
                # save to temp
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
                tfile.write(up.read())
                tfile.flush()
                st.session_state.cap = cv2.VideoCapture(tfile.name)
        else:
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("❌ Cannot open webcam. Check if camera is connected and permissions are granted.")
                st.session_state.processing = False

if stop_button:
    st.session_state.processing = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.success("✓ Stopped")

# Main processing loop
if st.session_state.processing and st.session_state.cap is not None:
    cap = st.session_state.cap
    frame_count = 0
    while st.session_state.processing:
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video or camera disconnected")
            st.session_state.processing = False
            break
        
        ts = time.time()
        annotated, stats = analyzer.analyze_frame(frame, timestamp=ts)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        video_placeholder.image(annotated_rgb, use_column_width=True)
        show_metrics(stats)
        
        frame_count += 1
        if frame_count % 30 == 0:
            st.write(f"Processed {frame_count} frames...")
        
        time.sleep(0.01)  # minimal sleep
    
    cap.release()
    st.session_state.cap = None

# Show timeline and export
st.markdown("---")
colA, colB = st.columns([2, 1])
with colA:
    st.header("Engagement Timeline")
    df = analyzer.get_timeline_df()
    if df.empty:
        st.info("No timeline data yet — start the analysis to populate metrics")
    else:
        df_plot = df.set_index('dt')['engagement_score']
        st.line_chart(df_plot)
        st.dataframe(df[['dt', 'total_students', 'attentive', 'sleeping', 'using_phone', 'hand_raise', 'engagement_score']].tail(200))
with colB:
    st.header("Export & Settings")
    if not df.empty:
        csv = df[['dt', 'total_students', 'attentive', 'sleeping', 'using_phone', 'hand_raise', 'engagement_score']].to_csv(index=False)
        st.download_button("Download CSV", csv, file_name=f"engagement_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

st.markdown("---")
st.write("Notes: This is a prototype. Accuracy depends on camera angle, lighting, and model performance. For better real-world results integrate ID tracking (ByteTrack), larger models, and calibrate thresholds per classroom.")

# End of file