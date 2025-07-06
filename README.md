# AI Surveillance System
**Version: 1.2**  
**Last Updated: July 06, 2025**  
**Author: Nawel-Bellil**

Welcome to the **AI Surveillance System**, a simple yet powerful tool designed to monitor spaces using a webcam or video files. It detects movement, recognizes familiar faces, and keeps a log of what it sees—all without needing complex coding skills to get started! This project is perfect for learning about AI in security or setting up basic surveillance at home or work.

## What Does It Do?
- **Spot Movement**: Watches a specific area and notices when something moves.
- **Recognize Faces**: Identifies people you’ve added to its memory.
- **Keep a Record**: Saves a log of events and can save video clips.

## How It Works 
Imagine the system as a helpful security guard with a checklist:
1. **Camera Check**: It looks at the video from your camera or a pre-recorded file.
2. **Motion Alert**: It checks if anything is moving in the area you care about.
3. **Face Check**: It looks for faces and matches them to people you’ve told it about.
4. **Log It**: It writes down what it sees and can save a picture if something important happens.
5. **Show You**: It displays everything on your screen with labels.

## Getting Started

### What You Need
- A computer with Windows, macOS, or Linux.
- A webcam (or a video file to test with).
- Anaconda installed (to manage Python easily).

### Installation
1. **Download the Project**:
   - Clone or download the files to your computer (e.g., `C:\Users\YourName\AI-surveillance-system`).
   - Open a terminal or command prompt and navigate there:
     ```bash
     cd C:\Users\YourName\AI-surveillance-system
     ```

2. **Set Up Python Environment**:
   - Install Anaconda if you don’t have it (from [anaconda.com](https://www.anaconda.com/)).
   - Create a new environment and install the needed tools:
     ```bash
     conda create -n surveillance_env python=3.10
     conda activate surveillance_env
     conda install -c conda-forge opencv-python numpy=1.26.4 dlib
     pip install face_recognition click
     ```

3. **Prepare Folders**:
   - Create these folders in the project directory:
     ```bash
     mkdir models known_faces surveillance_footage surveillance_footage\motion_events
     ```

4. **Set Up Configuration**:
   - Create a file named `config.env` with:
     ```
     CAMERA_SOURCE=0  # Use 0 for webcam, or a video file path like "demo_video.mp4"
     ROI_START_POINT=100,100  # Top-left corner of the area to watch
     ROI_END_POINT=540,380  # Bottom-right corner
     FACE_RECOGNITION_ENABLED=true
     FACE_RECOGNITION_THRESHOLD=0.6  # How strict the face match should be
     DETECTION_MODE=both  # Detects both motion and faces
     MIN_CONTOUR_AREA=5000  # Minimum size for motion to count
     PERSISTENCE_THRESHOLD=5  # How long motion must last
     ALERT_INTERVAL=10  # Time between alerts
     OUTPUT_DIR=surveillance_footage
     KNOWN_FACES_PATH=known_faces
     ```
   - Adjust `CAMERA_SOURCE` and other settings as needed.

## Using the System

### Step 1: Add Faces
- Run this to teach the system about people’s faces:
  ```bash
  "C:\Users\YourName\anaconda3\python.exe" register_face.py
  ```
- Type a name (e.g., "Alice") and press `SPACE` when their face is in view. Repeat for others.

### Step 2: Start Monitoring
- Launch the system:
  ```bash
  "C:\Users\YourName\anaconda3\python.exe" surveillance.py
  ```
- A window called "Surveillance Feed" will pop up. You’ll see the video with labels for motion or recognized faces.
- Press `ESC` to stop.

### Step 3: Check Logs
- Look in the `surveillance_footage` folder for a `security_log.txt` file. It lists what the system saw (e.g., "Motion detected" or "Recognized: Alice").

### Tips
- If the camera doesn’t work, check if it’s in use by another app.
- If faces aren’t recognized, try adjusting `FACE_RECOGNITION_THRESHOLD` or retaking the face photos with better lighting.

## Workflow Image
Here’s a simple picture to show how everything flows together:
![Workflow Diagram](https://github.com/user-attachments/assets/1bb8285e-a2d1-4040-a511-3cb416b6f788)

- **Camera Input**: Starts with your video source.
- **Motion Detection**: Flags any movement.
- **Face Recognition**: Matches faces to your list.
- **Logging**: Records everything.
- **Display**: Shows you the results.

## For Developers
If you know some coding, here’s more detail:
- The system uses Python 3.10 with OpenCV for video, `face_recognition` for faces, and a modular design.
- Each script (e.g., `face_detection.py`, `surveillance.py`) handles a specific task, making it easy to tweak.
- The pipeline is sequential for speed, with precomputed face data to save time.

## Contributing
Feel free to suggest improvements or fix bugs! Open an issue or pull request on the repository.

## License
[MIT License] - This project is open for use and modification under this license.


