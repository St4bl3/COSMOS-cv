import cv2

def extract_frames(video_path):
    """Extracts frames from a video file and returns frames and FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return [], 0
        
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: # Check for invalid or zero FPS
        print(f"Warning: Video FPS reported as {fps}. Defaulting to 25 FPS.")
        fps = 25.0 # Default FPS

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} at {fps} FPS.")
    return frames, fps