import os
from datetime import datetime
from utils import ensure_directory

class Logger:
    """Simple logging utility."""
    
    def __init__(self, log_file=None, output_dir='surveillance_footage'):
        """Initialize logger with optional log file."""
        self.output_dir = ensure_directory(output_dir)
        
        if log_file is None:
            log_file = os.path.join(self.output_dir, 'security_log.txt')
            
        self.log_file = log_file
        
    def log(self, event_type, details=None):
        """Log an event to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details_str = f" - {details}" if details else ""
        log_entry = f"[{timestamp}] {event_type}{details_str}\n"
        
        # Print to console
        print(log_entry, end="")
        
        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")

# Create a global logger instance
security_logger = None

def init_logger(log_file=None, output_dir='surveillance_footage'):
    """Initialize the global logger."""
    global security_logger
    security_logger = Logger(log_file, output_dir)
    return security_logger

def log_event(event_type, details=None):
    """Log an event using the global logger."""
    global security_logger
    if security_logger is None:
        security_logger = Logger()
    security_logger.log(event_type, details)