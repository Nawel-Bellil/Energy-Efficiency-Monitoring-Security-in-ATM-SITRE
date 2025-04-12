import cv2
import os
import argparse
from utils import ensure_directory
from face_detection import init_face_detector, save_face_encoding

def register_from_webcam(known_faces_path='known_faces'):
    """Register a new face using the webcam."""
    # Initialize face detector
    face_detector = init_face_detector()
    if face_detector is None:
        print("Error: Failed to initialize face detector")
        return False
        
    # Ensure directory exists
    ensure_directory(known_faces_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
        
    # Get name for the face
    name = input("Enter name for the new face: ")
    if not name:
        print("Name cannot be empty")
        cap.release()
        return False
        
    print("Press SPACE to capture the face or ESC to cancel...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame")
            break
            
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Display instructions
        cv2.putText(frame, "Press SPACE to capture or ESC to cancel", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        # Show the frame
        cv2.imshow('Face Registration', frame)
        
        # Check for key press
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC
            print("Registration cancelled")
            break
            
        if key == 32:  # SPACE
            # Check if a face is detected
            if len(faces) == 0:
                print("No face detected. Please try again.")
                continue
                
            if len(faces) > 1:
                print("Multiple faces detected. Please ensure only one face is in frame.")
                continue
                
            # Extract and save the face region
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            # Save face encoding
            if save_face_encoding(face_img, name, known_faces_path):
                print(f"Face registered successfully: {name}")
            else:
                print("Failed to register face")
                
            break
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    return True

def register_from_file(image_path, known_faces_path='known_faces'):
    """Register a new face from an image file."""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return False
        
    # Initialize face detector
    face_detector = init_face_detector()
    if face_detector is None:
        print("Error: Failed to initialize face detector")
        return False
        
    # Load the image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return False
        
    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        print("No face detected in the image")
        return False
        
    if len(faces) > 1:
        print("Multiple faces detected in the image. Please use an image with only one face.")
        return False
        
    # Extract face region
    x, y, w, h = faces[0]
    face_img = frame[y:y+h, x:x+w]
    
    # Get name for the face
    name = input("Enter name for the new face: ")
    if not name:
        print("Name cannot be empty")
        return False
        
    # Save face encoding
    if save_face_encoding(face_img, name, known_faces_path):
        print(f"Face registered successfully: {name}")
        return True
    else:
        print("Failed to register face")
        return False

def main():
    """Main entry point for face registration tool."""
    parser = argparse.ArgumentParser(description='Face Registration Tool')
    parser.add_argument('--image', type=str, help='Path to image file (optional)')
    parser.add_argument('--output', type=str, default='known_faces', help='Output directory for known faces')
    
    args = parser.parse_args()
    
    if args.image:
        register_from_file(args.image, args.output)
    else:
        register_from_webcam(args.output)

if __name__ == "__main__":
    main()