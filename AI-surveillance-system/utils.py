import os
import cv2
import numpy as np
from datetime import datetime

def load_env_file(file_path):
    """Load environment variables from a file."""
    config = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                key, value = line.split('=', 1)
                config[key] = value
                
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def parse_coordinates(coord_str):
    """Parse coordinates from string to tuple."""
    if not coord_str or coord_str.lower() == 'none':
        return None
        
    try:
        coords = list(map(int, coord_str.split(',')))
        if len(coords) == 2:
            return tuple(coords)
        elif len(coords) == 4:
            return (tuple(coords[:2]), tuple(coords[2:]))
        else:
            return None
    except (ValueError, TypeError):
        return None

def ensure_directory(directory):
    """Ensure a directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def get_timestamp(format_str="%Y-%m-%d %H:%M:%S"):
    """Get current timestamp as string."""
    return datetime.now().strftime(format_str)

def get_file_timestamp():
    """Get timestamp format suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def draw_timestamp(frame, timestamp=None):
    """Draw timestamp on the frame."""
    if timestamp is None:
        timestamp = get_timestamp()
        
    cv2.putText(
        frame, timestamp, (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    return frame

def draw_roi(frame, roi_start, roi_end):
    """Draw ROI rectangle on the frame."""
    if roi_start is not None and roi_end is not None:
        cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)
    return frame

def extract_roi(frame, roi_start, roi_end):
    """Extract the region of interest from frame."""
    if roi_start is None or roi_end is None:
        return frame
        
    x1, y1 = roi_start
    x2, y2 = roi_end
    
    # Ensure coordinates are within frame bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    
    return frame[y1:y2, x1:x2]

def save_frame(frame, directory, prefix="detection"):
    """Save a frame to disk with timestamp in filename."""
    timestamp = get_file_timestamp()
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    
    # Ensure directory exists
    ensure_directory(directory)
    
    # Save the frame
    cv2.imwrite(filepath, frame)
    print(f"Saved: {filepath}")
    
    return filepath

def str_to_bool(value):
    """Convert string value to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes', '1', 'y'):
        return True
    elif value.lower() in ('false', 'no', '0', 'n'):
        return False
    else:
        return False