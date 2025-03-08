import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
import os
import time

# STServo library import
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

sys.path.append("..")
from STservo_sdk import *

# Servo settings
STS_ID_AZIMUTH             = 1                 # Azimuth servo ID
STS_ID_ELEVATION           = 2                 # Elevation servo ID
BAUDRATE                   = 1000000           # STServo default baudrate
DEVICENAME                 = 'COM6'            # Serial port
STS_MOVING_SPEED           = 4000              # STServo moving speed
STS_MOVING_ACC             = 200               # STServo moving acc
STS_MINIMUM_POSITION       = 0                 # Minimum position value
STS_MAXIMUM_POSITION       = 4095              # Maximum position value

# Camera and tracking parameters
CAMERA_FOV_HORIZONTAL = 60  # Camera horizontal field of view in degrees
CAMERA_FOV_VERTICAL = 40    # Camera vertical field of view in degrees

# Servo control setup
portHandler = PortHandler(DEVICENAME)
packetHandler = sts(portHandler)

# Load YOLO model
model = YOLO("models/best_2.pt")

# Initialize current servo positions (center positions)
current_azimuth = 2048  # Center position for azimuth
current_elevation = 2048  # Center position for elevation

def initialize_servo_port():
    # Open port
    if portHandler.openPort():
        print("Succeeded to open the servo port")
    else:
        print("Failed to open the servo port")
        print("Press any key to terminate...")
        getch()
        return False

    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        return False
    
    return True

def move_servo(id, position):
    # Ensure position is within valid range
    position = max(STS_MINIMUM_POSITION, min(STS_MAXIMUM_POSITION, position))
    
    # Write servo position with default speed and acceleration
    sts_comm_result, sts_error = packetHandler.WritePosEx(id, position, STS_MOVING_SPEED, STS_MOVING_ACC)
    
    if sts_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(sts_comm_result))
        return False
    elif sts_error != 0:
        print("%s" % packetHandler.getRxPacketError(sts_error))
        return False
    
    return True

def pixel_offset_to_servo_angle(offset_x, offset_y, frame_width, frame_height):
    """
    Convert pixel offsets to servo angle adjustments.
    Returns the new absolute servo positions (0-4095).
    """
    global current_azimuth, current_elevation
    
    # Calculate normalized offsets (-1.0 to 1.0)
    normalized_x = offset_x / (frame_width / 2)
    normalized_y = offset_y / (frame_height / 2)
    
    # Calculate angle adjustments based on camera FOV
    angle_x = normalized_x * (CAMERA_FOV_HORIZONTAL / 2)
    angle_y = normalized_y * (CAMERA_FOV_VERTICAL / 2)
    
    # Convert to servo units (4096 units = 360 degrees, so ~11.38 units per degree)
    servo_units_per_degree = 4096 / 360
    adjustment_x = int(angle_x * servo_units_per_degree)
    adjustment_y = int(angle_y * servo_units_per_degree)
    
    # Calculate new servo positions (invert as needed for proper movement direction)
    new_azimuth = current_azimuth - adjustment_x  # Invert X axis if necessary
    new_elevation = current_elevation + adjustment_y  # Invert Y axis if necessary
    
    # Ensure positions stay within valid range
    new_azimuth = max(STS_MINIMUM_POSITION, min(STS_MAXIMUM_POSITION, new_azimuth))
    new_elevation = max(STS_MINIMUM_POSITION, min(STS_MAXIMUM_POSITION, new_elevation))
    
    return new_azimuth, new_elevation

def main():
    global current_azimuth, current_elevation
    
    # Initialize servo control
    if not initialize_servo_port():
        return
    
    # Center the turret at startup
    print("Centering turret...")
    move_servo(STS_ID_AZIMUTH, current_azimuth)
    move_servo(STS_ID_ELEVATION, current_elevation)
    time.sleep(1)  # Give servos time to move
    
    # Load video from camera
    cap = cv2.VideoCapture(0)
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera.")
        cap.release()
        portHandler.closePort()
        return
    
    frame_height, frame_width = frame.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)
    
    # Create NanoTracker
    params = cv2.TrackerNano_Params()
    folder = Path(__file__).parent.absolute()
    params.backbone = str(folder.joinpath('image_trackers/nanotrack_backbone_sim.onnx').absolute().resolve())
    params.neckhead = str(folder.joinpath('image_trackers/nanotrack_head_sim.onnx').absolute().resolve())
    tracker = cv2.TrackerNano_create(params)
    
    tracked_object = None  # Stores (x, y, w, h) of tracked object
    is_tracking = False  # Flag for tracking status
    
    # Distance estimation parameters
    KNOWN_DISTANCE = 1.0  # Meters
    KNOWN_BBOX_AREA = 5000  # Pixels
    K = KNOWN_DISTANCE * KNOWN_BBOX_AREA
    
    prev_center = None
    prev_size = None
    
    # User selects tracking mode
    print("Select tracking mode: [1] Automatic (YOLO)  [2] Manual (Click)")
    mode = input("Enter choice (1 or 2): ")
    
    manual_selection = (mode == "2")
    waiting_for_selection = manual_selection  # Pause video in manual mode
    selected_bbox = None  # Stores manually selected bounding box
    
    # Optical Flow parameters
    prev_gray = None
    camera_motion = np.array([0.0, 0.0])  # Estimated camera motion
    
    # Feature tracking parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_features = None
    
    def run_yolo_once(frame):
        results = model(frame)
        detections = results[0].boxes
        return detections
    
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Initialize optical flow
    prev_features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if manual_selection:
        cv2.imshow("Select Object", frame)
        print("Click on the object to track...")
    
    # Mouse callback for one-click object selection
    def select_object(event, x, y, flags, param):
        global selected_bbox, is_tracking, waiting_for_selection
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_bbox = (x - 15, y - 15, 30, 30)  # Simple bounding box around click point
            tracker.init(frame, selected_bbox)
            is_tracking = True
            waiting_for_selection = False  # Resume video playback
    
    if manual_selection:
        cv2.setMouseCallback("Select Object", select_object)
    
    # Wait for user selection in manual mode
    while manual_selection and waiting_for_selection:
        cv2.waitKey(1)  # Keep UI responsive
    
    cv2.destroyAllWindows()  # Close selection window before playback starts
    
    yolo_ran = False  # Flag to ensure YOLO runs only once in automatic mode
    last_servo_update = time.time()  # Time of last servo update
    
    # Main loop
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if video ends
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Estimate camera motion using optical flow
            if prev_gray is not None and prev_features is not None:
                next_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_features, None, **lk_params)
                if next_features is not None and prev_features is not None:
                    valid_motion = next_features[status == 1] - prev_features[status == 1]
                    if valid_motion.size > 0:
                        camera_motion = np.mean(valid_motion, axis=0)
            
            prev_gray = gray
            prev_features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            
            if not is_tracking:
                if not manual_selection and not yolo_ran:
                    # Run YOLO detection only once in automatic mode
                    detections = run_yolo_once(frame)
                    
                    if len(detections) > 0:
                        x1, y1, x2, y2 = map(int, detections.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        tracked_object = (x1, y1, w, h)
                        tracker.init(frame, tracked_object)
                        is_tracking = True
                        prev_center = (x1 + w // 2, y1 + h // 2)
                        prev_size = w * h
                        yolo_ran = True  # Stop running YOLO after detection
            
            else:
                # Update tracker
                success, tracked_object = tracker.update(frame)
                
                if success:
                    x, y, w, h = [int(v) for v in tracked_object]
                    bbox_center = (x + w // 2, y + h // 2)
                    bbox_area = w * h
                    estimated_distance = round(K / bbox_area, 2)
                    
                    # Draw bounding box with distance
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"Distance: {estimated_distance}m"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Calculate pixel offset from frame center
                    offset_x = bbox_center[0] - frame_center[0]
                    offset_y = bbox_center[1] - frame_center[1]
                    
                    # Display offset information
                    direction_text = f"Offset X: {offset_x}, Y: {offset_y}"
                    cv2.putText(frame, direction_text, (10, frame_height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Update servo positions at most every 0.1 seconds to avoid overwhelming the servos
                    current_time = time.time()
                    if current_time - last_servo_update >= 0.1:
                        # Convert pixel offsets to servo angles
                        new_azimuth, new_elevation = pixel_offset_to_servo_angle(
                            offset_x, offset_y, frame_width, frame_height)
                        
                        # Display servo positions
                        servo_text = f"Servo - Azimuth: {new_azimuth}, Elevation: {new_elevation}"
                        cv2.putText(frame, servo_text, (10, frame_height - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Move servos to track the object
                        move_servo(STS_ID_AZIMUTH, new_azimuth)
                        move_servo(STS_ID_ELEVATION, new_elevation)
                        
                        # Update current positions
                        current_azimuth = new_azimuth
                        current_elevation = new_elevation
                        
                        last_servo_update = current_time
                    
                    # Draw motion vector
                    if prev_center is not None:
                        adjusted_center = (bbox_center[0] - int(camera_motion[0]), 
                                        bbox_center[1] - int(camera_motion[1]))
                        arrow_color = (0, 255, 0) if bbox_area < prev_size else (0, 0, 255)
                        cv2.arrowedLine(frame, prev_center, adjusted_center, arrow_color, 3, tipLength=0.4)
                    
                    prev_center = bbox_center
                    prev_size = bbox_area
                    
                else:
                    # Lost tracking
                    is_tracking = False
                    yolo_ran = False  # Allow YOLO to run again to reacquire target
                    cv2.putText(frame, "Tracking lost!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show camera view with tracking information
            cv2.imshow("Drone Tracking Turret", frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        portHandler.closePort()
        print("Resources released, program terminated.")

if __name__ == "__main__":
    main()
