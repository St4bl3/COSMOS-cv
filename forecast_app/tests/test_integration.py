import pytest
import os
from unittest.mock import patch, MagicMock
from io import BytesIO
import numpy as np

# app fixture is from conftest.py
# CHANNELS, TARGET_SIZE can be imported from app if needed for mock setups
from app import CHANNELS, TARGET_SIZE 

def test_home_redirect(client):
    """Test the GET / route redirects to /upload."""
    response = client.get('/')
    assert response.status_code == 302
    # The location will be a full URL when testing with the client
    assert response.location.endswith('/upload')

def test_upload_get(client):
    """Test the GET /upload route renders the upload form."""
    response = client.get('/upload')
    assert response.status_code == 200
    assert b"<title>INSAT-3D Next-Day Forecast</title>" in response.data
    assert b"<h1>INSAT-3D Next-Day Forecast</h1>" in response.data
    assert b'name="file"' in response.data # Check for file input more robustly

@patch('app.load_and_preprocess')
@patch('app.model.predict')
@patch('app.plt.figure')        # Mock the figure creation function
@patch('app.plt.axes')          # Mock axes creation (if called as plt.axes)
@patch('app.plt.colorbar')      # Mock colorbar creation (if called as plt.colorbar)
@patch('app.plt.close')         # Mock closing the figure (if called as plt.close)
def test_upload_post_success(mock_plt_close, mock_plt_colorbar, mock_plt_axes, mock_plt_figure, mock_model_predict, mock_load_preprocess, client, dummy_h5_file_path, app):
    """Test POST /upload with a successful file upload and mocked processing."""

    # Configure the mock for plt.figure() to return a new MagicMock each time it's called,
    # this new MagicMock will represent the 'fig' object.
    mock_figure_instance = MagicMock()
    mock_plt_figure.return_value = mock_figure_instance

    # Configure mock for load_and_preprocess
    dummy_preprocessed_data = np.random.rand(1, *TARGET_SIZE, len(CHANNELS)).astype(np.float32)
    mock_load_preprocess.return_value = dummy_preprocessed_data

    # Configure mock for model.predict
    dummy_prediction_data = np.random.rand(*TARGET_SIZE, len(CHANNELS)).astype(np.float32)
    mock_model_predict.return_value = np.expand_dims(dummy_prediction_data, axis=0)

    # Prepare file for upload using the dummy_h5_file_path
    with open(dummy_h5_file_path, 'rb') as h5_file_content:
        data = {
            'file': (h5_file_content, os.path.basename(dummy_h5_file_path))
        }
        response = client.post('/upload', data=data, content_type='multipart/form-data')

    assert response.status_code == 200, f"Expected status 200, got {response.status_code}. Response data: {response.data[:500]}"
    assert b"Forecast Results" in response.data, "Results page identifier not found."
    assert b"pred_VIS.png" in response.data, "VIS channel image reference not found."

    # Verify mocks were called
    expected_saved_path = os.path.join(app.config['UPLOAD_DIR_CONFIG_TEST'], "input.h5")
    mock_load_preprocess.assert_called_once_with(expected_saved_path)
    
    mock_model_predict.assert_called_once_with(dummy_preprocessed_data)
    
    # Assert that plt.figure was called for each channel
    assert mock_plt_figure.call_count == len(CHANNELS), \
        f"plt.figure was called {mock_plt_figure.call_count} times, expected {len(CHANNELS)}"

    # Assert that the savefig method on our mock_figure_instance was called for each channel
    # This assumes that each call to plt.figure() in the loop gets the same mock_figure_instance.
    # If a new mock is desired per loop, the setup of mock_plt_figure needs to change (e.g., side_effect)
    # For this setup, mock_figure_instance is one object, so its savefig is called multiple times.
    assert mock_figure_instance.savefig.call_count == len(CHANNELS), \
        f"fig.savefig was called {mock_figure_instance.savefig.call_count} times, expected {len(CHANNELS)}"

    # Assert that plt.close was called for each figure
    assert mock_plt_close.call_count == len(CHANNELS), \
        f"plt.close was called {mock_plt_close.call_count} times, expected {len(CHANNELS)}"


def test_upload_post_invalid_file_extension(client):
    """Test POST /upload with an invalid file extension (e.g., .txt)."""
    data = {
        'file': (BytesIO(b"dummy text file content"), 'test.txt')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data')

    assert response.status_code == 302, "Expected redirect for invalid file type."
    assert response.location.endswith('/upload'), "Redirect location incorrect."
    
    redirect_response = client.get(response.location)
    assert redirect_response.status_code == 200
    assert b"Please upload a valid .h5 file." in redirect_response.data, "Flash message for invalid file not found."

def test_upload_post_no_file(client):
    """Test POST /upload with no file submitted."""
    response = client.post('/upload', data={}, content_type='multipart/form-data')
    assert response.status_code == 302, "Expected redirect when no file is submitted."
    assert response.location.endswith('/upload'), "Redirect location incorrect."

    redirect_response = client.get(response.location)
    assert redirect_response.status_code == 200
    assert b"Please upload a valid .h5 file." in redirect_response.data, "Flash message for no file not found."

def test_upload_post_filename_parsing_error_results_in_no_dates(client, dummy_h5_file_path, app):
    """Test POST /upload with a file whose name doesn't match the date regex, ensure no dates shown."""
    non_matching_filename = "THIS_IS_A_WRONGLY_NAMED_FILE.h5"
    
    with open(dummy_h5_file_path, 'rb') as original_h5_content:
        data = {
            'file': (original_h5_content, non_matching_filename)
        }
        with patch('app.load_and_preprocess', return_value=np.random.rand(1, *TARGET_SIZE, len(CHANNELS)).astype(np.float32)), \
             patch('app.model.predict', return_value=np.expand_dims(np.random.rand(*TARGET_SIZE, len(CHANNELS)).astype(np.float32), axis=0)), \
             patch('app.plt.figure') as mock_fig_again, \
             patch('app.plt.close'): # Also mock close here
            
            mock_actual_fig = MagicMock()
            mock_fig_again.return_value = mock_actual_fig

            response = client.post('/upload', data=data, content_type='multipart/form-data')

    assert response.status_code == 200, "Expected to still get results page even with bad filename for dates."
    assert b"Forecast Results" in response.data 
    assert b"Input Time:" not in response.data, "Input time should not be displayed for unparsable filename."
    assert b"Forecast Time:" not in response.data, "Forecast time should not be displayed for unparsable filename."
    assert b"pred_VIS.png" in response.data
    # Check that savefig was still called on the mock figure
    assert mock_actual_fig.savefig.call_count == len(CHANNELS)
