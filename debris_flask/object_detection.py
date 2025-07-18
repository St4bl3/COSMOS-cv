import torch
import cv2
import numpy as np
from model_loader import detection_model, DEVICE # Import the loaded model and device

# --- IMPORTANT: Ensure this matches the label used during training for asteroids ---
# Assuming background is class 0, asteroid is class 1.
ASTEROID_LABEL_ID = 1 
# --- ---

CONFIDENCE_THRESHOLD = 0.99 # You can adjust this (e.g., 0.2 or 0.1 for testing)

def detect_asteroids_in_frame(frame_cv2):
    """
    Detects asteroids in a single video frame using the loaded PyTorch model.

    Args:
        frame_cv2: The video frame (OpenCV format, BGR).

    Returns:
        A tuple (detected_boxes, detected_scores)
        detected_boxes: List of [xmin, ymin, xmax, ymax] for detected asteroids.
        detected_scores: List of confidence scores for the detections.
    """
    if detection_model is None:
        # This check should ideally be done before starting the processing loop in app.py
        print("Error: Detection model is not loaded.")
        return [], []

    # 1. Convert frame to format expected by the model
    #    - BGR to RGB
    #    - HWC to CHW (Height, Width, Channels to Channels, Height, Width)
    #    - Normalize if required by your model (usually done by ToTensorV2)
    #    - Convert to PyTorch tensor
    
    img_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor (this also changes HWC to CHW and scales to [0,1])
    # Using a simple manual conversion to tensor. For more complex preprocessing
    # (like specific normalizations used during training), you might need to
    # replicate those steps or use torchvision.transforms.
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension and send to device

    detected_boxes = []
    detected_scores = []

    with torch.no_grad(): # Ensure gradients are not computed during inference
        prediction = detection_model(img_tensor)

    if prediction and prediction[0]['boxes'].numel() > 0:
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()

        # --- START OF ADDED DEBUG PRINT STATEMENTS ---
        # Construct a simple frame identifier, e.g., based on a global counter or a hash of the frame
        # For now, just printing shape as a basic identifier.
        # If you process sequentially, you could pass a frame_number to this function.
        print(f"Frame (shape {frame_cv2.shape}): Raw scores & labels from model output:")
        for i_score, score_val in enumerate(scores):
            print(f"  - Raw Score: {score_val:.4f}, Raw Label: {labels[i_score]}")
        # --- END OF ADDED DEBUG PRINT STATEMENTS ---

        for i in range(len(boxes)):
            if labels[i] == ASTEROID_LABEL_ID and scores[i] >= CONFIDENCE_THRESHOLD:
                box = boxes[i]
                # Ensure box coordinates are within image dimensions if necessary, though typically not an issue
                # xmin, ymin, xmax, ymax = box
                detected_boxes.append(box)
                detected_scores.append(scores[i])
                
                # --- Optional: Print details of boxes that pass the threshold ---
                # print(f"    -> Kept: Box {box}, Score {scores[i]:.4f}, Label {labels[i]}")
                # --- ---

    return detected_boxes, detected_scores


def draw_detections_on_frame(frame_cv2, boxes, scores):
    """
    Draws bounding boxes and scores on a video frame.

    Args:
        frame_cv2: The video frame (OpenCV format).
        boxes: List of [xmin, ymin, xmax, ymax] for detected objects.
        scores: List of confidence scores for the detections.

    Returns:
        The frame with detections drawn on it.
    """
    output_frame = frame_cv2.copy()
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box) # Convert to integers for drawing
        score = scores[i]
        
        # Draw rectangle
        cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Green box
        
        # Prepare label text
        label_text = f"Asteroid: {score:.2f}"
        
        # Put label text above the rectangle
        # Calculate text size to position it nicely
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Ensure text background is opaque
        text_ymin = ymin - text_height - baseline
        if text_ymin < 0: # Adjust if text goes off the top of the image
            text_ymin = ymin + baseline + 5 # Put below top of box if no space above
            
        cv2.rectangle(output_frame, (xmin, text_ymin - baseline), (xmin + text_width, text_ymin + text_height + baseline), (0, 255, 0), -1) # Green filled background for text
        cv2.putText(output_frame, label_text, (xmin, text_ymin + text_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Black text

    return output_frame

if __name__ == '__main__':
    # This part is for testing this script directly.
    # It won't run when app.py imports it.
    print("Running object_detection.py directly for testing...")

    # Ensure model is loaded (it should be by importing model_loader)
    if detection_model is None:
        print("Test Error: Model not loaded. Make sure model_loader.py runs correctly.")
    else:
        print(f"Test Info: Model loaded successfully on {DEVICE}.")
        
        # Create a dummy black image for testing
        # Replace this with `cv2.imread('your_test_image.jpg')` if you have one
        dummy_frame_height, dummy_frame_width = 480, 640
        dummy_frame = np.zeros((dummy_frame_height, dummy_frame_width, 3), dtype=np.uint8)
        
        print(f"Test Info: Created a dummy frame of size {dummy_frame.shape}")

        # Add some dummy features to make it slightly more interesting than pure black
        cv2.circle(dummy_frame, (100,100), 30, (255,0,0), -1) # Blue circle
        cv2.rectangle(dummy_frame, (200,200), (300,300), (0,0,255), -1) # Red square

        # Simulate a single frame detection
        print("Test Info: Calling detect_asteroids_in_frame...")
        test_boxes, test_scores = detect_asteroids_in_frame(dummy_frame)
        print(f"Test Info: Detected {len(test_boxes)} objects.")

        if test_boxes:
            for i, box in enumerate(test_boxes):
                print(f"  - Box: {box}, Score: {test_scores[i]:.4f}")
            
            output_test_frame = draw_detections_on_frame(dummy_frame, test_boxes, test_scores)
            
            # Try to display the image (requires a GUI environment)
            # cv2.imshow("Test Detection", output_test_frame)
            # cv2.waitKey(0) # Wait for a key press
            # cv2.destroyAllWindows()
            
            # Save the output image instead of displaying
            cv2.imwrite("test_detection_output.jpg", output_test_frame)
            print("Test Info: Saved test output to test_detection_output.jpg")
        else:
            print("Test Info: No objects detected in the dummy frame (as expected if model is specific).")
            # Save the original dummy frame if nothing detected
            cv2.imwrite("test_detection_output_no_detections.jpg", dummy_frame)
            print("Test Info: Saved dummy frame (no detections) to test_detection_output_no_detections.jpg")