import cv2
import os
import pickle
import numpy as np
from logger import log_event
from utils import ensure_directory

# Global variables for face detection/recognition
face_detector = None
face_recognition_loaded = False
known_face_encodings = []
known_face_names = []

def init_face_detector():
    """Initialize the face detector."""
    global face_detector
    # Use Haar cascade for basic face detection
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_detector = cv2.CascadeClassifier(face_cascade_path)
    
    if face_detector.empty():
        log_event("ERROR", "Failed to load face detector")
        return None
    
    log_event("INIT", "Face detector initialized")
    return face_detector

def load_face_recognition():
    """Load face recognition library if available."""
    global face_recognition_loaded
    
    try:
        global face_recognition
        import face_recognition
        face_recognition_loaded = True
        log_event("INIT", "Face recognition library loaded")
        return True
    except ImportError:
        log_event("WARNING", "face_recognition library not available. Using basic face detection only.")
        return False

def load_known_faces(known_faces_path='known_faces'):
    """Load known faces from the database."""
    global known_face_encodings, known_face_names, face_recognition_loaded
    
    # Ensure the directory exists
    ensure_directory(known_faces_path)
    
    # Check if face recognition is available
    if not face_recognition_loaded and not load_face_recognition():
        log_event("WARNING", "Face recognition not available. Cannot load known faces.")
        return False
    
    # Check for pre-computed encodings
    encodings_file = os.path.join(known_faces_path, "face_encodings.pickle")
    
    if os.path.exists(encodings_file):
        log_event("INFO", "Loading pre-computed face encodings...")
        try:
            with open(encodings_file, "rb") as f:
                data = pickle.load(f)
                known_face_encodings = data["encodings"]
                known_face_names = data["names"]
            log_event("INFO", f"Loaded {len(known_face_names)} known faces")
            return True
        except Exception as e:
            log_event("ERROR", f"Failed to load face encodings: {e}")
    
    # If no pre-computed encodings, process image files
    log_event("INFO", "No pre-computed encodings found. Processing image files...")
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Extract name from filename (e.g., "john_doe.jpg" -> "John Doe")
            name = os.path.splitext(filename)[0].replace('_', ' ').title()
            
            # Load image
            image_path = os.path.join(known_faces_path, filename)
            
            try:
                # Process the image with face_recognition
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    # Use the first detected face
                    face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    log_event("INFO", f"Loaded face: {name}")
                else:
                    log_event("WARNING", f"No face detected in {filename}")
            except Exception as e:
                log_event("ERROR", f"Error processing {filename}: {e}")
    
    # Save encodings to file for faster loading next time
    if known_face_encodings:
        try:
            log_event("INFO", "Saving face encodings to file...")
            data = {"encodings": known_face_encodings, "names": known_face_names}
            with open(encodings_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            log_event("ERROR", f"Failed to save face encodings: {e}")
    
    log_event("INFO", f"Loaded {len(known_face_names)} known faces")
    return len(known_face_names) > 0

def detect_faces(frame):
    """
    Basic face detection using Haar cascade.
    
    Args:
        frame: Input video frame
        
    Returns:
        List of detected face rectangles (x, y, w, h)
    """
    global face_detector
    
    if face_detector is None:
        face_detector = init_face_detector()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces

def recognize_faces(frame, threshold=0.6):
    """
    Detect and recognize faces in the frame.
    
    Args:
        frame: Input video frame
        threshold: Face recognition confidence threshold
        
    Returns:
        (display_frame, detected_faces): 
            - display_frame: Frame with annotations
            - detected_faces: List of face information dictionaries
    """
    global face_recognition_loaded, known_face_encodings, known_face_names
    
    # Create a copy of the frame for drawing
    display_frame = frame.copy()
    detected_faces = []
    
    # Check if face recognition is available
    if not face_recognition_loaded:
        if not load_face_recognition():
            # Fall back to basic detection
            faces = detect_faces(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                name = "Unknown"
                cv2.putText(display_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                detected_faces.append({
                    'name': name,
                    'location': (x, y, x+w, y+h),
                    'recognized': False
                })
                
            return display_frame, detected_faces
    
    # Load known faces if not already loaded
    if not known_face_encodings:
        load_known_faces()
    
    # Use face_recognition library
    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find all faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Loop through each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale face locations back to original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Try to match the face with known faces
        matches = []
        name = "Unknown"
        
        if known_face_encodings:
            # Compare face with known faces
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=threshold
            )
            
            # Use the first match
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
        
        # Draw rectangle and name
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(display_frame, name, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        
        # Add to detected faces list
        detected_faces.append({
            'name': name,
            'location': (left, top, right, bottom),
            'recognized': name != "Unknown"
        })
    
    return display_frame, detected_faces

def save_face_encoding(face_img, name, known_faces_path='known_faces'):
    """
    Process a face image and save its encoding to the database.
    
    Args:
        face_img: Face image to encode
        name: Name to associate with the face
        known_faces_path: Directory for storing face data
        
    Returns:
        Success status (boolean)
    """
    global face_recognition_loaded, known_face_encodings, known_face_names
    
    # Check if face recognition is available
    if not face_recognition_loaded and not load_face_recognition():
        log_event("ERROR", "Face recognition not available. Cannot add face.")
        return False
    
    # Ensure directory exists
    ensure_directory(known_faces_path)
    
    try:
        # Convert BGR to RGB (face_recognition uses RGB)
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = face_img
            
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_img)
        
        if not face_locations:
            log_event("ERROR", "No face detected in the image")
            return False
            
        # Use the first detected face
        face_encoding = face_recognition.face_encodings(rgb_img, [face_locations[0]])[0]
        
        # Add to known faces
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        
        # Save face image to disk
        filename = f"{name.lower().replace(' ', '_')}.jpg"
        filepath = os.path.join(known_faces_path, filename)
        cv2.imwrite(filepath, face_img)
        
        # Update encodings file
        encodings_file = os.path.join(known_faces_path, "face_encodings.pickle")
        data = {"encodings": known_face_encodings, "names": known_face_names}
        with open(encodings_file, "wb") as f:
            pickle.dump(data, f)
            
        log_event("INFO", f"Added {name} to known faces database")
        return True
        
    except Exception as e:
        log_event("ERROR", f"Failed to add face: {str(e)}")
        return False