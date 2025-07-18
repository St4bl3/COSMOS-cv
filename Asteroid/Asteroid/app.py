from flask import Flask, render_template, Response, request, jsonify, stream_with_context
import cv2 # OpenCV for video processing
import time
import json
from object_detector_mock import ObjectDetectorMock
from video_processing import AsteroidPathProcessor
import os
import threading

VIDEO_UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
if not os.path.exists(VIDEO_UPLOAD_FOLDER):
    os.makedirs(VIDEO_UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = VIDEO_UPLOAD_FOLDER

detector = ObjectDetectorMock()
processor = AsteroidPathProcessor() # Instantiated globally
video_capture = None
processing_active_video_path = None # Track which video is being processed
stop_processing_flag = threading.Event()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def video_frame_generator(video_path_to_process):
    global video_capture, detector, processor, stop_processing_flag, processing_active_video_path

    if processing_active_video_path != video_path_to_process: # If it's a new video or first video
        processor.reset_state() # Reset processor state for the new video
        processing_active_video_path = video_path_to_process


    try:
        video_capture_local = cv2.VideoCapture(video_path_to_process) # Use local var for capture
        if not video_capture_local.isOpened():
            yield f"data: {{ \"error\": \"Could not open video file: {video_path_to_process}\" }}\n\n"
            return

        fps = video_capture_local.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        frame_count = 0

        # stop_processing_flag should be cleared by the calling function if a new stream starts

        while not stop_processing_flag.is_set():
            ret, frame = video_capture_local.read()
            if not ret:
                yield "data: { \"status\": \"Video processing complete.\" }\n\n"
                break

            timestamp_ms = int(video_capture_local.get(cv2.CAP_PROP_POS_MSEC))
            detections = detector.detect(frame) # Your actual model call here

            if detections:
                bbox = detections[0]['bbox']
                processed_output = processor.process_bounding_box(timestamp_ms, bbox)
                if processed_output:
                    point_3d_data = {
                        "timestamp": processed_output["timestamp"],
                        "point": processed_output["point_3d"],
                        "bbox_img": processed_output["bbox_original"]
                    }
                    yield f"data: {json.dumps(point_3d_data)}\n\n"

            time.sleep(frame_delay) # Adjust to control streaming rate vs. processing rate
            frame_count += 1

    except Exception as e:
        app.logger.error(f"Error in video processing stream: {e}")
        yield f"data: {{ \"error\": \"Error during processing: {str(e)}\" }}\n\n"
    finally:
        if 'video_capture_local' in locals() and video_capture_local.isOpened():
            video_capture_local.release()
        app.logger.info(f"Video processing generator for {video_path_to_process} finished.")
        if stop_processing_flag.is_set() or not ret : # if loop exited due to stop or end of video
             processing_active_video_path = None # Allow new video to reset processor state
        yield f"data: {{ \"status\": \"Stream closed for {os.path.basename(video_path_to_process)}.\" }}\n\n"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global stop_processing_flag # Allow modification

    if 'videoFile' not in request.files:
        return jsonify({"error": "No video file part"}), 400
    file = request.files['videoFile']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Stop any existing processing by setting the flag
        # The generator itself checks this flag
        stop_processing_flag.set()
        # Small delay to allow the existing generator to see the flag and exit
        time.sleep(0.2)


        filename = "uploaded_video_" + str(int(time.time())) + "." + file.filename.rsplit('.', 1)[1].lower()
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        app.logger.info(f"Video saved to {video_path}")

        # Clear the flag for the new stream that will be started by the client  
        stop_processing_flag.clear()

        return jsonify({"message": "Video uploaded successfully. Path created.", "video_path": video_path}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/stream_3d_path')
def stream_3d_path():
    video_path = request.args.get('video_path', None)
    if not video_path or not os.path.exists(video_path):
        def error_stream():
            yield f"data: {{ \"error\": \"Video path not provided or video not found.\" }}\n\n"
        return Response(error_stream(), mimetype='text/event-stream')

    # Important: Clear the flag for this new stream.
    # If a previous stream was stopped by flag, it needs to be clear for the new one.
    # The upload_video now handles this for subsequent uploads.
    # If client directly calls stream_3d_path, ensure flag is clear if no other stream expected.
    # For simplicity, let's assume upload_video sets the context for stop_processing_flag correctly.
    # If a stream is initiated for a NEW video path after a previous one was stopped,
    # the `upload_video` endpoint should have already cleared `stop_processing_flag`.
    # If it's for the SAME video path and the flag was set, this logic might need refinement
    # to decide if it's a "resume" or "restart". Current setup implies restart.

    app.logger.info(f"Starting stream for {video_path}. Stop flag is: {stop_processing_flag.is_set()}")
    return Response(stream_with_context(video_frame_generator(video_path)), mimetype='text/event-stream')


@app.route('/stop_stream', methods=['POST'])
def stop_stream_endpoint():
    global stop_processing_flag, processing_active_video_path
    app.logger.info("Stop stream request received.")
    stop_processing_flag.set()
    processing_active_video_path = None # Clear active video path on manual stop
    return jsonify({"message": "Processing stop signal sent. Stream will halt."}), 200


if __name__ == '__main__':
    # Consider Spark context lifecycle:
    # processor object (global) holds the SparkSession.
    # It's started when processor is initialized.
    # It should be stopped when the app exits.
    try:
        # `threaded=True` is important for handling concurrent requests (like SSE and other endpoints)
        app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
    finally:
        if processor:
             processor.stop_spark()
        app.logger.info("Flask app stopped, Spark session should be closed.")