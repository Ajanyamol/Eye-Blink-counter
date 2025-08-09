# Eye Blink Counter
## Team Name: Amigos
## Team Members:
 * Member 1: Ajanyamol Thomas - College of Engineering Adoor
 * Member 2: Ayisha Anuna - College of Engineering Adoor
## Project Description
[This project is a simple yet entertaining real-time and recorded video eye blink counter using a webcam and video files. It uses MediaPipe’s advanced facial landmark detection to track eye movements and count blinks by calculating the Eye Aspect Ratio (EAR). The program works by detecting when the eyes close and open again, incrementing the blink count accordingly]

## The Problem (that doesn't exist)
[no one really cares about how many times you blink your eyes]

## The Solution (that nobody asked for)
[here we are, counting blinks like it’s the next big thing in tech. This project is exactly what it says on the tin: a program that counts eye blinks using your webcam or a recorded video]

## Technical Details
### Technologies/Components Used
#### For Software:

Language used : python
Libraries used : 
* MediaPipe :A cross-platform framework by Google for building multimodal (video, audio, etc.) applied ML pipelines.Provides ready-to-use solutions like Face Mesh, which detects detailed facial landmarks in real-time, allowing accurate eye tracking and blink detection.
* OpenCV (Open Source Computer Vision Library):A powerful library for real-time computer vision and image processing tasks.Used here to capture video from your webcam or read video files, display the video frames, and draw landmarks on the frames.
* NumPy:A fundamental package for scientific computing in Python.Used for numerical operations such as calculating Euclidean distances needed for the Eye Aspect Ratio (EAR) calculation
## Implementation
### For Software:
### Installation

Follow these steps to set up the project on your local machine:

1. **Install Python 3.7 or higher**  
   Download and install from [python.org](https://www.python.org/downloads/).

2. **Clone or download this repository**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>


### Run:
python blink_counter.py
