import cv2

# Initialize video capture
video = cv2.VideoCapture("abrams.mp4")

# Get video FPS
fps = video.get(cv2.CAP_PROP_FPS)
try:
    frame_delay = int(1000 / fps)  # Calculate the delay between frames in milliseconds
except:
    frame_delay = 25

# List to store multiple trackers and bounding boxes
saved_bboxes = []
current_tracker = None
current_bbox = None

# Function to start tracking a new object
def start_tracking():
    global current_tracker, current_bbox, saved_bboxes
    success, frame = video.read()
    if not success:
        return
    bbox = cv2.selectROI("Frame", frame, False)
    if bbox != (0, 0, 0, 0):
        current_tracker = cv2.TrackerKCF_create()
        current_tracker.init(frame, bbox)
        current_bbox = bbox
        saved_bboxes.append(bbox)

# Function to try to track with saved bboxes
def try_saved_bboxes():
    global current_tracker, current_bbox, saved_bboxes
    success, frame = video.read()
    if not success:
        return
    for bbox in saved_bboxes:
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, bbox)
        success, _ = tracker.update(frame)
        if success:
            current_tracker = tracker
            current_bbox = bbox
            break

# Get video frame dimensions
success, frame = video.read()
frame_height, frame_width = frame.shape[:2]

# Calculate center of the frame (static point)
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

while True:
    success, frame = video.read()
    if not success:
        break

    if current_tracker is not None:
        success, current_bbox = current_tracker.update(frame)
        if success:
            # Draw bounding box on the object
            (x, y, w, h) = [int(i) for i in current_bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate center of the green rectangle
            green_center_x = x + w // 2
            green_center_y = y + h // 2
            
            # Draw the center of the green rectangle
            cv2.circle(frame, (green_center_x, green_center_y), 5, (0, 255, 0), -1)  # Draw a small green circle at the center

            # Calculate the difference in coordinates
            delta_x = frame_center_x - green_center_x
            delta_y = frame_center_y - green_center_y

            # Display the difference in the top-right corner of the frame
            text = f"Move: x={delta_x}px, y={delta_y}px"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            current_tracker = None
            current_bbox = None
            try_saved_bboxes()

    # Draw the center point in the middle of the frame
    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)  # Draw a small blue circle at the center

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(frame_delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        start_tracking()

video.release()
cv2.destroyAllWindows()
