initial version of project

##  Project Overview
This project is an end-to-end computer vision system designed to act as a **real-time personal trainer**. It uses a standard webcam to analyze squat technique and provide instant, corrective feedback (e.g., "Lower Hips", "Knees Inward").

Unlike simple static analysis tools, this system uses a **temporal model (Bi-LSTM)** to understand the full context of the movement, achieving **90% accuracy** in distinguishing between correct and incorrect forms on a custom dataset.

##  Key Features
* **Custom Dataset:** Taken custom dataset of 20 subjects from front and side angles.
* **Temporal Analysis:** Uses a Dual-Stream Bidirectional LSTM to analyze the entire repetition trajectory, not just single frames.
* **Hybrid Evaluation:** Combines Deep Learning for robust binary classification ("Good/Bad") with Geometric Heuristics for explainable feedback ("Straighten Back").
* **View-Invariant Preprocessing:** Normalization techniques make the model robust to user position and camera distance.
* **Real-Time Feedback:** Delivers analysis in milliseconds using a lightweight architecture suitable for consumer CPUs/GPUs.

## System Architecture
1.  **Input:** Real-time video feed via OpenCV.
2.  **Pose Estimation:** **YOLOv8-Pose** extracts 17 skeletal keypoints per frame.
3.  **State Management:** A logic-based state machine monitors vertical hip displacement to segment individual repetitions.
4.  **Inference Engine:**  **Normalization:** Coordinates are centered relative to the hip midpoint.
    * **Sequence Processing:** Frames are buffered and interpolated to a fixed sequence length (60 frames).
    * **Model:** Bi-LSTM with Multi-Head Attention mechanism.
5.  **Output:** Visual Status (Good/Bad), and specific geometric corrections.

##  Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Computer Vision:** Ultralytics YOLOv8, OpenCV
* **Data Processing:** NumPy, SciPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

