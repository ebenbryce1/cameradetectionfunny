import cv2
import mediapipe as mp
import math
import numpy as np
import os

# --- CONFIGURATION ---
CAMERA_INDEX = 1  # Change to 0 or -1 if camera doesn't open
CONFIDENCE = 0.5  # Sensitivity of AI detection

# --- SETUP MEDIAPIPE HOLISTIC (Face + Body + Hands) ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- IMAGE LOADER ---
def load_image(name):
    path = f"{name}.png"
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Creating a blank placeholder.")
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

# Load all 6 images
images = {
    "point": load_image("point"),
    "type":  load_image("type"),
    "angry": load_image("angry"),
    "dead":  load_image("dead"),
    "bowtie": load_image("bowtie"),
    "george": load_image("george")
}

# --- HELPER: Overlay Function ---
def overlay_image(background, overlay, x, y, size_scale=1.0):
    if overlay is None: return background
    
    h, w = overlay.shape[:2]
    new_w = int(w * size_scale)
    new_h = int(h * size_scale)
    if new_w <= 0 or new_h <= 0: return background
    overlay = cv2.resize(overlay, (new_w, new_h))
    
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape

    if x >= bg_w or y >= bg_h: return background
    
    # Clip overlay if it goes off screen
    if x + ol_w > bg_w: ol_w = bg_w - x; overlay = overlay[:, :ol_w]
    if y + ol_h > bg_h: ol_h = bg_h - y; overlay = overlay[:ol_h, :]
    if x < 0: ol_w += x; overlay = overlay[:, -ol_w:]; x = 0
    if y < 0: ol_h += y; overlay = overlay[-ol_h:, :]; y = 0
    
    if ol_w <= 0 or ol_h <= 0: return background

    roi = background[y:y+ol_h, x:x+ol_w]

    # Blend with transparency
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            roi[:, :, c] = (1. - alpha) * roi[:, :, c] + alpha * overlay[:, :, c]
    else:
        roi = overlay

    background[y:y+ol_h, x:x+ol_w] = roi
    return background

# --- MAIN APP ---
cap = cv2.VideoCapture(CAMERA_INDEX)
# cap.set(3, 640) # Uncomment these if Pi lags
# cap.set(4, 480)

with mp_holistic.Holistic(min_detection_confidence=CONFIDENCE, min_tracking_confidence=CONFIDENCE) as holistic:
    print("Gestures Active! Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Mirror and process
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Default state
        current_img = None 
        status = "Waiting..."
        
        # We need Body landmarks (Pose) and Hand landmarks
        pose = results.pose_landmarks
        
        if pose:
            # Helper to get coordinates
            def get_pos(idx):
                return int(pose.landmark[idx].x * w), int(pose.landmark[idx].y * h)
            
            # Key Body Points
            nose = pose.landmark[0]
            left_ear = pose.landmark[7]
            right_ear = pose.landmark[8]
            
            left_wrist = pose.landmark[15]
            right_wrist = pose.landmark[16]
            left_shoulder = pose.landmark[11]
            right_shoulder = pose.landmark[12]
            left_hip = pose.landmark[23]
            right_hip = pose.landmark[24]

            # --- GESTURE LOGIC START ---

            # 1. DEAD (Arms Horizontal Out)
            # Check if wrists are far apart AND at shoulder height
            # (Wrist Y is close to Shoulder Y)
            arms_level = abs(left_wrist.y - left_shoulder.y) < 0.15 and abs(right_wrist.y - right_shoulder.y) < 0.15
            arms_wide = abs(left_wrist.x - right_wrist.x) > 0.6 # 60% of screen width
            
            # 2. GEORGE (Hands on Hips)
            # Wrists close to Hips vertically and horizontally
            l_hand_on_hip = abs(left_wrist.y - left_hip.y) < 0.15 and abs(left_wrist.x - left_hip.x) < 0.15
            r_hand_on_hip = abs(right_wrist.y - right_hip.y) < 0.15 and abs(right_wrist.x - right_hip.x) < 0.15
            
            # 3. BOWTIE (Hands near neck)
            # Wrists close to Shoulders vertically, but close to EACH OTHER horizontally
            hands_high = abs(left_wrist.y - left_shoulder.y) < 0.15 and abs(right_wrist.y - right_shoulder.y) < 0.15
            hands_close = abs(left_wrist.x - right_wrist.x) < 0.2

            # 4. TYPING (Hands face down in front)
            # Wrists below shoulders but above hips, hands close together
            hands_mid = left_wrist.y > left_shoulder.y and left_wrist.y < left_hip.y
            hands_close_mid = abs(left_wrist.x - right_wrist.x) < 0.25

            # 5. ANGRY (Head Tilt Down)
            # If Nose is significantly LOWER (higher Y value) than Ears, head is tilted down
            # Normal: Nose is roughly level with ears. Tilted down: Nose drops.
            nose_y = nose.y
            ears_y = (left_ear.y + right_ear.y) / 2
            head_tilt = (nose_y - ears_y) > 0.08  # Threshold for tilt

            # 6. POINT (1 Finger Up)
            # We check if ONE hand is detected and Index finger is up
            pointing = False
            if results.right_hand_landmarks or results.left_hand_landmarks:
                # Use whichever hand is visible (prefer Right)
                hand_lms = results.right_hand_landmarks if results.right_hand_landmarks else results.left_hand_landmarks
                if hand_lms:
                    # Index Tip (8) higher than Index PIP (6)
                    idx_tip = hand_lms.landmark[8].y
                    idx_pip = hand_lms.landmark[6].y
                    # Middle Tip (12) lower than Middle PIP (10) -> Finger curled
                    mid_tip = hand_lms.landmark[12].y
                    mid_pip = hand_lms.landmark[10].y
                    
                    if idx_tip < idx_pip and mid_tip > mid_pip:
                        pointing = True

            # --- DECISION TREE (Priority) ---
            if pointing:
                current_img = images["point"]
                status = "POINT"
            elif l_hand_on_hip and r_hand_on_hip:
                current_img = images["george"]
                status = "GEORGE (Hips)"
            elif arms_level and arms_wide:
                current_img = images["dead"]
                status = "DEAD (T-Pose)"
            elif hands_high and hands_close:
                current_img = images["bowtie"]
                status = "BOWTIE"
            elif head_tilt:
                current_img = images["angry"]
                status = "ANGRY (Head Down)"
            elif hands_mid and hands_close_mid:
                current_img = images["type"]
                status = "TYPING"

            # --- DRAW OVERLAY ---
            if current_img is not None:
                # We place the image relative to the Nose (Point 0)
                nx, ny = get_pos(0)
                
                # Offset: Move image -100 pixels Left, -150 pixels Up from nose
                frame = overlay_image(frame, current_img, nx - 100, ny - 150, size_scale=0.8)

            cv2.putText(frame, f"Gesture: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gesture Reactor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()