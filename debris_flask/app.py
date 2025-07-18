from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import time
import uuid # For unique task IDs
from werkzeug.utils import secure_filename
import threading # For running processing in background (optional for better UI responsiveness)
import traceback # For detailed error logging

# Try to import project modules
try:
    from video_processing import extract_frames
    from object_detection import detect_asteroids_in_frame, draw_detections_on_frame
    from model_loader import detection_model # Check if model loaded
except ImportError as e:
    print(f"ImportError in app.py: {e}. Ensure all .py files are in the root directory.")
    # Define dummy functions if imports fail, so app can still start to show errors
    def extract_frames(p): return [], 0
    def detect_asteroids_in_frame(f): return [], []
    def draw_detections_on_frame(f, b, s): return f
    detection_model = None


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed_videos')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit, adjust as needed

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# In-memory store for task statuses (for simplicity)
# In a production app, you might use Redis or a database
task_statuses = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_job(task_id, input_video_path, output_video_name):
    output_video_path = os.path.join(app.config['PROCESSED_FOLDER'], output_video_name)
    
    try:
        if detection_model is None:
            task_statuses[task_id] = {"status": "Error", "message": "Detection model not loaded. Cannot process video."}
            print("Error: Detection model not loaded in process_video_job.")
            return

        task_statuses[task_id] = {"status": "Processing", "message": "Extracting frames...", "progress": 10}
        print(f"[Task {task_id}] Extracting frames from {input_video_path}")
        frames, fps = extract_frames(input_video_path)

        if not frames:
            task_statuses[task_id] = {"status": "Error", "message": "Could not extract frames from video."}
            print(f"[Task {task_id}] Error: No frames extracted.")
            if os.path.exists(input_video_path): os.remove(input_video_path)
            return

        height, width, _ = frames[0].shape
        # Common codecs: 'mp4v' (for .mp4), 'XVID' (for .avi)
        # H.264 is often 'avc1' or 'h264', but support varies. 'mp4v' is a safe bet for MP4.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        
        out_video = None # Initialize to None
        try:
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not out_video.isOpened():
                task_statuses[task_id] = {"status": "Error", "message": "Could not open VideoWriter. Check codec and permissions."}
                print(f"[Task {task_id}] Error: VideoWriter not opened for {output_video_path} with codec mp4v.")
                if os.path.exists(input_video_path): os.remove(input_video_path)
                return

            print(f"[Task {task_id}] VideoWriter opened for {output_video_path}. Processing {len(frames)} frames.")
            total_frames = len(frames)
            for i, frame in enumerate(frames):
                current_progress = 10 + int((i / total_frames) * 80) # Progress from 10% to 90%
                task_statuses[task_id] = {
                    "status": "Processing", 
                    "message": f"Detecting asteroids: Frame {i+1}/{total_frames}",
                    "progress": current_progress
                }
                if (i+1) % 10 == 0: # Log progress periodically
                     print(f"[Task {task_id}] Processing frame {i+1}/{total_frames}")

                boxes, scores = detect_asteroids_in_frame(frame)
                output_frame = draw_detections_on_frame(frame, boxes, scores)
                out_video.write(output_frame)
            
            task_statuses[task_id] = {"status": "Processing", "message": "Finalizing video...", "progress": 95}
            print(f"[Task {task_id}] Finished processing frames. Releasing video.")
        finally:
            if out_video is not None and out_video.isOpened(): # Check if it was opened before releasing
                out_video.release()
                print(f"[Task {task_id}] VideoWriter released.")
            # Verify file was created and has size
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                 task_statuses[task_id] = {"status": "Complete", "filename": output_video_name, "message": "Processing complete!", "progress": 100}
                 print(f"[Task {task_id}] Video processing complete. Output: {output_video_path}")
            else:
                task_statuses[task_id] = {"status": "Error", "message": f"Video processing failed or output file is empty/not created ({output_video_path}). Check console logs."}
                print(f"[Task {task_id}] Error: Output video file {output_video_path} not found or empty after processing.")

    except Exception as e:
        detailed_error = traceback.format_exc()
        task_statuses[task_id] = {"status": "Error", "message": f"An error occurred: {str(e)}"}
        print(f"[Task {task_id}] EXCEPTION during video processing: {str(e)}\n{detailed_error}")
    finally:
        # Clean up uploaded file
        if os.path.exists(input_video_path):
            try:
                os.remove(input_video_path)
                print(f"[Task {task_id}] Cleaned up input file: {input_video_path}")
            except Exception as e_rem:
                print(f"[Task {task_id}] Error removing uploaded file {input_video_path}: {e_rem}")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            # This case should ideally be caught by 'required' on the input field
            return render_template('index.html', error="No video file selected.")
        
        file = request.files['video']
        if file.filename == '':
            return render_template('index.html', error="No video file selected.")

        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            _, file_extension = os.path.splitext(original_filename)
            
            task_id = str(uuid.uuid4()) # Generate a unique task ID
            
            # Use a timestamp or task_id for unique filenames to avoid collisions
            input_video_name = f"input_{task_id}{file_extension}"
            output_video_name = f"processed_{task_id}.mp4" # Standardize output to mp4
            
            input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], input_video_name)
            
            try:
                file.save(input_video_path)
                print(f"Uploaded video saved to {input_video_path}")
            except Exception as e:
                print(f"Error saving uploaded file: {e}")
                return render_template('index.html', error=f"Error saving file: {e}")

            # Initialize status
            task_statuses[task_id] = {"status": "Pending", "message": "Video queued for processing...", "progress": 0}
            
            # Run processing in a separate thread so the request can return immediately
            # This is crucial for the progress bar to work without the page freezing
            thread = threading.Thread(target=process_video_job, args=(task_id, input_video_path, output_video_name))
            thread.start()
            
            # Redirect to the processing status page
            return redirect(url_for('processing_status_page', task_id=task_id))
        else:
            return render_template('index.html', error="Invalid file type. Allowed types: mp4, avi, mov, mkv.")
            
    return render_template('index.html', error=None)


@app.route('/processing/<task_id>')
def processing_status_page(task_id):
    return render_template('processing.html', task_id=task_id)


@app.route('/status/<task_id>')
def task_status(task_id):
    status = task_statuses.get(task_id, {"status": "Unknown", "message": "Task ID not found.", "progress": 0})
    return jsonify(status)


@app.route('/results/<filename>')
def display_video(filename):
    # Security check: ensure filename is safe and refers to a file in PROCESSED_FOLDER
    if ".." in filename or filename.startswith("/"): # Basic path traversal check
        return "Invalid filename", 400
    
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(video_path):
        return "Processed video not found. It may have failed or been cleaned up.", 404
        
    return render_template('results.html', filename=filename)


@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(video_path):
        return "Video not found", 404

    def generate():
        with open(video_path, "rb") as f:
            while True:
                chunk = f.read(4096) # Read in chunks
                if not chunk:
                    break
                yield chunk
    
    # --- ENSURE MIMETYPE AND HEADERS ---
    response = Response(generate(), mimetype='video/mp4')
    response.headers['Content-Disposition'] = 'inline; filename=' + filename
    # The following headers can help with streaming and caching behavior
    response.headers['Accept-Ranges'] = 'bytes'
    # response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    # response.headers['Pragma'] = 'no-cache'
    # response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    print("Starting Flask app...")
    if detection_model is None:
        print("WARNING: Object detection model was not loaded. Detection functionality will not work.")
    else:
        print("Object detection model seems to be loaded.")
    app.run(debug=True, threaded=True) # threaded=True is helpful for handling concurrent requests like status checks