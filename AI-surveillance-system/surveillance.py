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
