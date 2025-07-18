import pytest
import os
import h5py
import numpy as np
import tempfile
from unittest.mock import patch
import threading
import time
import requests # For threaded_flask_server health check

# Attempt to import constants from your app.py
try:
    from app import CHANNELS as APP_CHANNELS, TARGET_SIZE as APP_TARGET_SIZE
except ImportError:
    # Fallback if app module is not easily importable here, define them manually
    # Ensure these match your app.py
    APP_CHANNELS = ['VIS', 'MIR', 'SWIR', 'WV', 'TIR1']
    APP_TARGET_SIZE = (256, 256)


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    """
    Session-scoped Flask app fixture.
    Patches UPLOAD_DIR and STATIC_DIR used by app.py to temporary directories.
    """
    mock_upload_dir_obj = tmp_path_factory.mktemp("uploads_session_pytest")
    mock_static_dir_obj = tmp_path_factory.mktemp("static_outputs_session_pytest")

    mock_upload_dir_str = str(mock_upload_dir_obj)
    mock_static_dir_str = str(mock_static_dir_obj)

    # Patch the module-level UPLOAD_DIR and STATIC_DIR in your app.py
    with patch('app.UPLOAD_DIR', mock_upload_dir_str), \
         patch('app.STATIC_DIR', mock_static_dir_str):

        from app import app as flask_app # Import app after patching

        flask_app.config.update({
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test_secret_key_for_pytest_session",
            "SERVER_NAME": "localhost.test:5000", # Default, can be overridden by server
             # Store these paths in config as well for potential access within tests
            "UPLOAD_DIR_CONFIG_TEST": mock_upload_dir_str,
            "STATIC_DIR_CONFIG_TEST": mock_static_dir_str,
        })
        
        # Diagnostic check for pickling (can be removed after confirming)
        # import pickle
        # try:
        #     pickle.dumps(flask_app)
        #     print("\n[DIAGNOSTIC] Flask app instance appears to be pickleable.\n")
        # except Exception as e:
        #     print(f"\n[DIAGNOSTIC] Flask app instance IS NOT pickleable: {e}\n")

        yield flask_app
    # Patches are automatically undone here. tmp_path_factory handles cleanup.


@pytest.fixture(scope="session")
def client(app):
    """A test client for the app, session-scoped."""
    return app.test_client()


@pytest.fixture
def dummy_h5_file_path(tmp_path):
    """
    Creates a dummy H5 file with the expected structure for testing.
    Returns the file path as a string.
    The filename includes a parsable date for testing date extraction.
    """
    h5_filename = tmp_path / "dummy_3RIMG_10MAR2025_0215_L1C_TEST.h5"
    raw_data_shape = (300, 300) # Example shape before resize

    with h5py.File(h5_filename, 'w') as f:
        for ch_name in APP_CHANNELS:
            img_data = np.random.randint(0, 10, size=raw_data_shape).astype(np.int16)
            f.create_dataset(f"IMG_{ch_name}", data=np.expand_dims(img_data, axis=0))

            # Corrected: use num instead of size for np.linspace
            lut_data = np.linspace(0, 100, num=1024, dtype=np.float32)
            
            if ch_name == 'VIS':
                f.create_dataset('IMG_VIS_ALBEDO', data=lut_data)
            elif ch_name == 'MIR':
                f.create_dataset('IMG_MIR_RADIANCE', data=lut_data)
            elif ch_name == 'SWIR':
                f.create_dataset('IMG_SWIR_RADIANCE', data=lut_data)
            elif ch_name == 'WV':
                f.create_dataset('IMG_WV_RADIANCE', data=lut_data)
            elif ch_name == 'TIR1':
                f.create_dataset('IMG_TIR1_TEMP', data=lut_data)
    
    return str(h5_filename)


@pytest.fixture(scope="session")
def threaded_flask_server(app):
    """
    Runs the Flask app in a separate thread for E2E tests.
    This avoids multiprocessing pickling issues on Windows with pytest-flask's live_server.
    Yields the base URL of the server.
    """
    host = "127.0.0.1"
    port = 5001 # Ensure this port is free or make it dynamic
    server_url = f"http://{host}:{port}"

    # Update app's SERVER_NAME if not already set appropriately for this port
    app.config["SERVER_NAME"] = f"{host}:{port}"

    def run_server():
        print(f"Starting Flask app on {server_url} in a thread for E2E tests...")
        try:
            # use_reloader=False is crucial for threaded execution within tests
            app.run(host=host, port=port, use_reloader=False, use_debugger=False, threaded=True)
        except Exception as e:
            print(f"Error starting Flask server in thread: {e}")


    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for the server to be responsive
    timeout = 15  # seconds
    health_check_url = f"{server_url}/upload" # A known working GET endpoint

    start_time = time.time()
    server_ready = False
    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_check_url, timeout=1)
            if response.status_code == 200:
                print(f"Flask server in thread started successfully at {health_check_url}")
                server_ready = True
                break
        except requests.ConnectionError:
            time.sleep(0.5) # Wait and retry
        except requests.Timeout:
            print(f"Connection timed out during health check for {health_check_url}")
            time.sleep(0.5)


    if not server_ready:
        raise RuntimeError(f"Flask server in thread did not start at {health_check_url} within {timeout} seconds.")

    yield server_url
    # Daemon thread will exit when the main pytest process finishes.
    # No explicit stop needed for daemon thread with simple Flask dev server for tests.
