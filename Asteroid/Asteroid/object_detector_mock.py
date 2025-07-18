import numpy as np
import time

class ObjectDetectorMock:
    def __init__(self):
        # Simulate an asteroid moving across the frame and slightly in depth
        self.pos_x = 50
        self.pos_y = 50
        self.size = 20
        self.vx = 2  # pixels per frame
        self.vy = 1.5
        self.vz_factor = 0.99 # Multiplicative factor for size change (simulating depth)
        self.frame_width = 640
        self.frame_height = 480

    def detect(self, frame):
        # Simulate some processing time
        time.sleep(0.05)

        # Update position
        self.pos_x += self.vx
        self.pos_y += self.vy
        self.size *= self.vz_factor

        # Bounce off edges (simple simulation)
        if self.pos_x + self.size / 2 > self.frame_width or self.pos_x - self.size / 2 < 0:
            self.vx *= -1
            self.vz_factor = 1 / self.vz_factor # Reverse depth effect
        if self.pos_y + self.size / 2 > self.frame_height or self.pos_y - self.size / 2 < 0:
            self.vy *= -1
            self.vz_factor = 1 / self.vz_factor

        # Ensure size doesn't get too small or too large
        self.size = np.clip(self.size, 10, 50)
        if self.size <=10.5 or self.size >=49.5: # Add a bit of margin to avoid rapid flipping
            self.vz_factor = 1 / self.vz_factor


        # Bounding box: [x_min, y_min, width, height]
        x_min = self.pos_x - self.size / 2
        y_min = self.pos_y - self.size / 2
        width = self.size
        height = self.size

        if np.random.rand() < 0.9: # Simulate detection 90% of the time
            return [{"bbox": [x_min, y_min, width, height], "confidence": 0.95, "label": "asteroid"}]
        else:
            return []

if __name__ == '__main__':
    # Example Usage
    detector = ObjectDetectorMock()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Dummy frame
    for i in range(50):
        detections = detector.detect(mock_frame)
        if detections:
            print(f"Frame {i}: Detected {detections[0]['label']} at {detections[0]['bbox']}")
        else:
            print(f"Frame {i}: No detection")