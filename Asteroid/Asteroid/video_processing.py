from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, LongType, ArrayType, IntegerType
import time
import numpy as np

class AsteroidPathProcessor:
    def __init__(self):
        self.spark = SparkSession.builder.appName("AsteroidPathProcessing") \
            .master("local[*]") \
            .config("spark.sql.streaming.schemaInference", "true") \
            .getOrCreate()

        self.bbox_schema = StructType([
            StructField("timestamp", LongType(), True),
            StructField("x_center", FloatType(), True),
            StructField("y_center", FloatType(), True),
            StructField("width", FloatType(), True),
            StructField("height", FloatType(), True)
        ])

        self.time_series_data = []
        self.initial_depth_estimate = 1000.0
        self.depth_scale_factor = 500.0
        self.prev_area = None
        self.prev_point_3d = None
        self.frame_width = 640
        self.frame_height = 480
        self.fov_horizontal_rad = np.deg2rad(60)

    def _estimate_depth(self, width, height):
        current_area = width * height
        depth_change_factor = 1.0

        if self.prev_area and self.prev_area > 0: # ensure prev_area is not zero
            area_ratio = current_area / self.prev_area
            depth_change_factor = 1 / (area_ratio**0.5) if area_ratio > 0 else 1.0
        else: # First frame or prev_area was zero
            depth_change_factor = 1.0


        self.prev_area = current_area
        if not self.prev_point_3d:
            estimated_depth = self.initial_depth_estimate
        else:
            estimated_depth = self.prev_point_3d[2] * depth_change_factor # Note: Z is negative, so depth is positive
            # If using negative Z for 'into the screen', then:
            # estimated_depth = abs(self.prev_point_3d[2]) * depth_change_factor
            # For now, let's assume depth is a positive value and Z becomes -depth
            estimated_depth = abs(self.prev_point_3d[2]) * depth_change_factor if self.prev_point_3d[2] != 0 else self.initial_depth_estimate


        return np.clip(estimated_depth, self.initial_depth_estimate / 5, self.initial_depth_estimate * 5)

    def _project_to_3d(self, x_center_img, y_center_img, estimated_depth):
        ndc_x = (x_center_img - self.frame_width / 2) / (self.frame_width / 2)
        ndc_y = (self.frame_height / 2 - y_center_img) / (self.frame_height / 2)

        aspect_ratio = self.frame_width / self.frame_height
        # Using estimated_depth (positive value) for calculations
        view_x = ndc_x * estimated_depth * np.tan(self.fov_horizontal_rad / 2) # Incorrect: aspect_ratio usually applied to x or tan argument for y
        # Corrected approach:
        # Half height of the view plane at distance `d` is `d * tan(fov_vertical / 2)`
        # Half width is `d * tan(fov_horizontal / 2)`
        # For FOV_H:
        view_x = ndc_x * estimated_depth * np.tan(self.fov_horizontal_rad / 2)
        # If fov_horizontal_rad is indeed horizontal FOV:
        # tan(fov_h / 2) = (view_plane_half_width) / depth
        # view_x = ndc_x * view_plane_half_width
        # view_y needs vertical FOV or calculate from horizontal FOV and aspect ratio
        fov_vertical_rad = 2 * np.arctan(np.tan(self.fov_horizontal_rad / 2) / aspect_ratio)
        view_y = ndc_y * estimated_depth * np.tan(fov_vertical_rad / 2)
        view_z = -estimated_depth # Negative Z for "into the screen"

        return view_x, view_y, view_z


    def process_bounding_box(self, timestamp_ms, bbox):
        if not bbox:
            return None

        x_min, y_min, width, height = bbox
        if width <= 0 or height <= 0: # Avoid division by zero or invalid areas
             return None
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        estimated_depth = self._estimate_depth(width, height)
        x_3d, y_3d, z_3d = self._project_to_3d(x_center, y_center, estimated_depth)

        current_point_3d = (x_3d, y_3d, z_3d)
        self.prev_point_3d = current_point_3d

        new_data = [(timestamp_ms, float(x_center), float(y_center), float(width), float(height),
                     float(x_3d), float(y_3d), float(z_3d))]
        self.time_series_data.extend(new_data)

        return {"timestamp": timestamp_ms, "point_3d": current_point_3d, "bbox_original": bbox}

    def get_all_3d_points(self):
        return [p[5:8] for p in self.time_series_data]

    def reset_state(self): # Added to clear state for new video processing
        self.time_series_data = []
        self.prev_area = None
        self.prev_point_3d = None
        # self.initial_depth_estimate remains same or could be reset too

    def stop_spark(self):
        self.spark.stop()

if __name__ == '__main__':
    processor = AsteroidPathProcessor()
    mock_detector = ObjectDetectorMock()
    mock_frame = np.zeros((processor.frame_height, processor.frame_width, 3), dtype=np.uint8)

    path_3d = []
    for i in range(100):
        timestamp = int(time.time() * 1000) + i * 33
        detections = mock_detector.detect(mock_frame)
        if detections:
            bbox_data = detections[0]['bbox']
            processed_data = processor.process_bounding_box(timestamp, bbox_data)
            if processed_data:
                print(f"Timestamp: {processed_data['timestamp']}, 3D Point: {processed_data['point_3d']}")
                path_3d.append(processed_data['point_3d'])
        time.sleep(0.03)

    all_points_df = processor.spark.createDataFrame(
        [(p[0], p[1], p[2]) for p in path_3d], ["x", "y", "z"]
    )
    print("Path collected in Spark DataFrame:")
    all_points_df.show(5)

    processor.stop_spark()