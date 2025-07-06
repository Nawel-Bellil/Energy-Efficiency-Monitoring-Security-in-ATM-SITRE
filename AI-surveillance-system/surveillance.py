import cv2
import os
import time
import numpy as np
import argparse
from datetime import datetime

# Import modules
from utils import (
    load_env_file, parse_coordinates, ensure_directory, 
    get_timestamp, draw_timestamp, draw_roi, save_frame,
    str_to_bool
)
from logger import init_logger, log_event
from motion_detection import init_motion_detector, detect_motion, draw_motion_contours
from face_detection import init_face_detector, load_known_faces, recognize_faces

def load_config(config_file='config.env'):
    """Load configuration settings."""
    # Load settings from config file
    if os.path.exists(config_file):
        config = load_env_file(config_file)
    else:
        config = {}
        print(f"Warning: Config file {config_file} not found, using defaults.")
    
    # Camera settings
    camera_source = config.get('CAMERA_SOURCE', '0')
    if camera_source.isdigit():
        camera_source = int(camera_source)
    
    # Parse ROI coordinates
    roi_start_point_str = config.get('ROI_START_POINT', '')
    roi_end_point_str = config.get('ROI_END_POINT', '')
    
    roi_coords = parse_coordinates(roi_start_point_str + ',' + roi_end_point_str)
    if roi_coords and len(roi_coords) == 2:
        roi_start_point, roi_end_point = roi_coords
    else:
        roi_start_point, roi_end_point = None, None
    
    # Detection settings
    detection_mode = config.get('DETECTION_MODE', 'both')
    min_contour_area = int(config.get('MIN_CONTOUR_AREA', '5000'))
    persistence_threshold = int(config.get('PERSISTENCE_THRESHOLD', '5'))
    alert_interval = int(config.get('ALERT_INTERVAL', '10'))
    
    # Face recognition settings
    face_recognition_enabled = str_to_bool(config.get('FACE_RECOGNITION_ENABLED', 'true'))
    face_recognition_threshold = float(config.get('FACE_RECOGNITION_THRESHOLD', '0.6'))
    
    # Output settings
        # Output settings
    output_dir = config.get('OUTPUT_DIR', 'surveillance_footage')
    ensure_directory(output_dir)

    return {
        'camera_source': camera_source,
        'roi_start_point': roi_start_point,
        'roi_end_point': roi_end_point,
        'detection_mode': detection_mode,
        'min_contour_area': min_contour_area,
        'persistence_threshold': persistence_threshold,
        'alert_interval': alert_interval,
        'face_recognition_enabled': face_recognition_enabled,
        'face_recognition_threshold': face_recognition_threshold,
        'output_dir': output_dir
    }
def main():
    """Main entry point for surveillance system."""
    # Load configuration
    config = load_config()
    
    # Initialize logger
    init_logger(output_dir=config['output_dir'])
    
    # Initialize detectors
    init_face_detector()
    init_motion_detector()
    load_known_faces()  # Load known faces from known_faces directory
    
    # Open video capture
    cap = cv2.VideoCapture(config['camera_source'])
    if not cap.isOpened():
        log_event("ERROR", "Could not open camera")
        return
    
    log_event("INIT", f"Camera opened with source: {config['camera_source']}")
    
    # Variables for persistence
    motion_persistence = 0
    last_alert_time = time.time()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            log_event("ERROR", "Failed to grab frame")
            break
        
        # Draw timestamp
        frame = draw_timestamp(frame)
        
        # Draw ROI if specified
        if config['roi_start_point'] and config['roi_end_point']:
            frame = draw_roi(frame, config['roi_start_point'], config['roi_end_point'])
        
        # Detect motion
        motion_detected, mask, contours = detect_motion(
            frame,
            config['roi_start_point'],
            config['roi_end_point'],
            config['min_contour_area']
        )
        
        # Update persistence
        if motion_detected:
            motion_persistence += 1
        else:
            motion_persistence = max(0, motion_persistence - 1)
        
        # Trigger alert if motion persists
        current_time = time.time()
        if motion_persistence >= config['persistence_threshold']:
            if current_time - last_alert_time >= config['alert_interval']:
                log_event("ALERT", "Motion detected!")
                save_frame(frame, config['output_dir'], prefix="motion_alert")
                last_alert_time = current_time
        
        # Recognize faces if enabled
        if config['face_recognition_enabled']:
            display_frame, detected_faces = recognize_faces(
                frame,
                threshold=config['face_recognition_threshold']
            )
            frame = display_frame
            
            for face in detected_faces:
                if face['recognized']:
                    log_event("INFO", f"Recognized: {face['name']}")
                else:
                    log_event("INFO", "Unknown face detected")
        
        # Draw motion contours
        if contours:
            frame = draw_motion_contours(frame, contours, config['roi_start_point'], config['roi_end_point'])
        
        # Display the frame
        cv2.imshow('Surveillance Feed', frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    log_event("INFO", "Surveillance system stopped")

if __name__ == "__main__":
    main()