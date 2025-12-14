## Project Overview
This project is an end-to-end computer vision system designed to act as a real-time personal trainer. It uses a standard webcam to analyze squat technique, and provide instant, corrective feedback (e.g., "Lower Hips", "Knees Inward").

Unlike simple static analysis tools, this system uses a temporal model (Bi-LSTM) to understand the full context of the movement, achieving 90% accuracy in distinguishing between correct and incorrect forms on a custom dataset.

##  Key Features
**Dataset:** Custom dataset of 20 subjects recorded from Front and Side views.
**Temporal Analysis:** Uses a Dual-Stream Bidirectional LSTM to analyze the entire repetition trajectory, not just single frames.
**Hybrid Evaluation:** Combines Deep Learning for robust binary classification ("Good/Bad") with Geometric Heuristics for explainable feedback ("Straighten Back").
**State Machine Logic:** Intelligent rep detection (`IDLE` -> `SQUATTING` -> `COMPLETE`) prevents false positives and ensures accurate counting.
**View-Invariant Preprocessing:** Normalization techniques make the model robust to user position and camera distance.
**Real-Time Feedback:** Delivers analysis in milliseconds using a lightweight architecture suitable for consumer CPUs/GPUs.

## System Architecture
**Input:** Real-time video feed via OpenCV.
**Pose Estimation:** **YOLOv8-Pose** extracts 17 skeletal keypoints per frame.
**State Management:** A logic-based state machine monitors vertical hip displacement to segment individual repetitions.
**Inference Engine:** Coordinates are centered relative to the hip midpoint. Bi-LSTM with Multi-Head Attention mechanism.
**Output:** Status (Good/Bad), and specific geometric corrections.

##  Tech Stack
**Language:** Python
**Deep Learning:** PyTorch
**Computer Vision:** Ultralytics YOLOv8, OpenCV
**Data Processing:** NumPy, SciPy, Scikit-Learn
**Visualization:** Matplotlib, Seaborn

## Project Structure
HUMAN-POSTURE-DETECTION/
models/
  best.pth             
notebooks/
  squat_training.ipynb 
results/
  dataset.png
  training.png
app.py                   # Main deployment script for real-time webcam  requirements.txt        
README.md                


