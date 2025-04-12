import cv2
import numpy as np
from utils import extract_roi
from logger import log_event

# Initialize the background subtractor
bg_subtractor = None

def init_motion_detector():
    """Initialize the motion detector."""
    global bg_subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    log_event("INIT", "Motion detector initialized")
    return bg_subtractor

def detect_motion(frame, roi_start=None, roi_end=None, min_area=5000):
    """
    Detect motion in the frame or region of interest.
    
    Args:
        frame: Input video frame
        roi_start: Top-left point of ROI (or None for full frame)
        roi_end: Bottom-right point of ROI (or None for full frame)
        min_area: Minimum contour area to consider as motion
        
    Returns:
        (motion_detected, mask, contours): 
            - motion_detected: Boolean indicating if motion was detected
            - mask: The processed foreground mask
            - contours: List of contours where motion was detected
    """
    global bg_subtractor
    if bg_subtractor is None:
        bg_subtractor = init_motion_detector()
    
    # Apply ROI if specified
    if roi_start is not None and roi_end is not None:
        # Extract ROI from the frame
        roi_frame = extract_roi(frame, roi_start, roi_end)
    else:
        roi_frame = frame
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(roi_frame)
    
    # Threshold the mask
    _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour exceeds the minimum area
    motion_detected = False
    motion_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            motion_detected = True
            motion_contours.append(contour)
    
    return motion_detected, thresh, motion_contours

def draw_motion_contours(frame, contours, roi_start=None, roi_end=None):
    """
    Draw motion contours on the frame.
    
    Args:
        frame: Input video frame
        contours: Contours to draw
        roi_start: Top-left point of ROI (for coordinate adjustment)
        roi_end: Bottom-right point of ROI (not directly used but included for consistency)
        
    Returns:
        Frame with contours drawn
    """
    if not contours:
        return frame
        
    # Create a copy of the frame to avoid modifying the original
    result = frame.copy()
    
    # If ROI was used, adjust contour coordinates
    if roi_start is not None:
        x_offset, y_offset = roi_start
        for contour in contours:
            # Shift contour to correct position in full frame
            shifted_contour = contour + np.array([x_offset, y_offset])
            cv2.drawContours(result, [shifted_contour], -1, (0, 255, 255), 2)
    else:
        # Draw contours directly if no ROI
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
        
    return result