import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from collections import deque
import time

THRESHOLDS = {
    'DEPTH_MIN': 60,   'DEPTH_MAX': 135, 
    'BACK_MIN': 20,    'BACK_MAX': 85,
    'KNEE_RATIO': 0.5,
    
    # State Triggers
    'STAND_THRESH': 165, 
    'SQUAT_THRESH': 140  
}

# Keypoint Indices
L_SH=5; R_SH=6; L_HIP=11; R_HIP=12; 
L_KNEE=13; R_KNEE=14; L_ANK=15; R_ANK=16

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.LSTM(51, 128, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.s = nn.LSTM(51, 128, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attn = nn.MultiheadAttention(256, 4, dropout=0.3, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(512,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3))
        self.temp = nn.Sequential(nn.Linear(256,128), nn.Tanh(), nn.Linear(128,1))
        self.cls = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,2))
    
    def forward(self, f, s):
        fo, _ = self.f(f)
        so, _ = self.s(s)
        fa, _ = self.attn(fo, so, so)
        fused = self.fuse(torch.cat([fa, so], -1))
        aw = F.softmax(self.temp(fused), 1)
        return self.cls(torch.sum(aw * fused, 1))

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0: return 0
    cosine_angle = np.dot(v1, v2) / denom
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_specific_feedback(kps):
    """Diagnose the squat errors"""
    hip, knee, ankle = kps[R_HIP][:2], kps[R_KNEE][:2], kps[R_ANK][:2]
    shoulder = kps[R_SH][:2]
    
    depth_angle = calculate_angle(hip, knee, ankle)
    back_angle = calculate_angle(shoulder, hip, knee)
    knee_width = abs(kps[L_KNEE][0] - kps[R_KNEE][0])
    ankle_width = abs(kps[L_ANK][0] - kps[R_ANK][0])

    # Relaxed feedback checks
    if depth_angle > THRESHOLDS['DEPTH_MAX']: 
        return f"LOWER HIPS ({int(depth_angle)})"
    if depth_angle < THRESHOLDS['DEPTH_MIN']: 
        return f"TOO LOW ({int(depth_angle)})"
    if back_angle < THRESHOLDS['BACK_MIN'] or back_angle > THRESHOLDS['BACK_MAX']: 
        return "STRAIGHTEN BACK"
    if (knee_width / (ankle_width + 1e-6)) < THRESHOLDS['KNEE_RATIO']: 
        return "KNEES OUT"
    
    return "IMPROVE FORM"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on {device}...")

model = Model().to(device)
try:
    ckpt = torch.load('models/best.pth', map_location=device)
    model.load_state_dict(ckpt['m'])
    print(" Model Loaded Successfully")
except:
    print(" Error: 'best.pth' not found.")
    exit()
model.eval()

print("Loading YOLOv8...")
pose_model = YOLO('yolov8n-pose.pt')

# State Machine
q_kps = deque(maxlen=90)
squat_state = "IDLE"
feedback = "Please stand up"
box_color = (100, 100, 100)
reset_timer = 0
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = pose_model(frame, verbose=False)
    display_frame = results[0].plot()

    if len(results[0].keypoints.data) > 0:
        raw_kps = results[0].keypoints.data[0].cpu().numpy()
        
        # Normalize
        norm_kps = raw_kps.copy()
        hip_center = (raw_kps[L_HIP,:2] + raw_kps[R_HIP,:2]) / 2
        norm_kps[:, :2] -= hip_center
        q_kps.append(norm_kps.flatten())

        # State Machine
        r_hip, r_knee, r_ank = raw_kps[R_HIP][:2], raw_kps[R_KNEE][:2], raw_kps[R_ANK][:2]
        current_angle = calculate_angle(r_hip, r_knee, r_ank)

        if squat_state == "IDLE":
            feedback = "Please stand up"
            box_color = (100, 100, 100) 
            
            if current_angle > THRESHOLDS['STAND_THRESH']: 
                squat_state = "READY"
        
        elif squat_state == "READY":
            feedback = "Ready. Squat!"
            box_color = (255, 200, 0) 
            
            if current_angle < THRESHOLDS['SQUAT_THRESH']: 
                squat_state = "SQUATTING"

        elif squat_state == "SQUATTING":
            feedback = "Squatting..."
            box_color = (0, 200, 255) 
            
            if current_angle > THRESHOLDS['STAND_THRESH']: 
                squat_state = "COMPLETE"

        elif squat_state == "COMPLETE":
            if len(q_kps) >= 60:
                recent_data = list(q_kps)[-60:]
                input_seq = torch.FloatTensor(np.array(recent_data)).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(input_seq, input_seq)
                    probs = F.softmax(logits, dim=1)
                    pred_class = torch.argmax(probs).item()
                    conf = probs[0][pred_class].item()

                if pred_class == 1:
                    feedback = f"GOOD! ({conf:.0%})"
                    box_color = (0, 255, 0) 
                else:
                    reason = get_specific_feedback(raw_kps) 
                    feedback = f"{reason}"
                    box_color = (0, 0, 255) 
            
            squat_state = "COOLDOWN"
            reset_timer = time.time()

        elif squat_state == "COOLDOWN":
            if (time.time() - reset_timer) > 3.0: 
                squat_state = "IDLE" 

    cv2.rectangle(display_frame, (0,0), (640, 60), box_color, -1)
    cv2.putText(display_frame, feedback, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('AI Squat Coach', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()