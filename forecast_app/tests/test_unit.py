import pytest
import numpy as np
import h5py 
from datetime import datetime, timedelta
import re

from app import load_and_preprocess, CHANNELS, TARGET_SIZE, MEAN, STD

def test_load_and_preprocess(dummy_h5_file_path):
    """Test the H5 file loading and preprocessing function."""
    processed_data = load_and_preprocess(dummy_h5_file_path)

    assert processed_data.shape == (1, *TARGET_SIZE, len(CHANNELS)), \
        f"Expected shape {(1, *TARGET_SIZE, len(CHANNELS))}, but got {processed_data.shape}"
    assert processed_data.dtype == np.float32, \
        f"Expected dtype np.float32, but got {processed_data.dtype}"

    for i in range(len(CHANNELS)):
        channel_data = processed_data[0, :, :, i]
        assert np.any(channel_data != 0) or np.allclose(np.mean(channel_data), 0, atol=1e-1), \
            f"Channel {CHANNELS[i]} data might be all zeros or not normalized properly (mean check)."
        assert np.std(channel_data) > 1e-5, \
            f"Std dev for channel {CHANNELS[i]} is too low ({np.std(channel_data)}), possibly not normalized."

def test_filename_date_parsing():
    """Test the filename parsing logic (as used in the upload route)."""
    # Test case 1: Valid filename
    filename1 = "3RIMG_10MAR2025_0215_L1C_EXTRA.h5"
    match1 = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", filename1)
    assert match1 is not None, f"Regex did not match valid filename: {filename1}"
    date_str, time_str = match1.groups()
    dt_in = datetime.strptime(date_str + time_str, "%d%b%Y%H%M")
    dt_out = dt_in + timedelta(days=1)
    assert dt_in == datetime(2025, 3, 10, 2, 15)
    assert dt_out == datetime(2025, 3, 11, 2, 15)

    # Test case 2: Filename with different valid date (end of year)
    filename2 = "3RIMG_31DEC2024_2359_L1C_SOMETHING.h5"
    match2 = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", filename2)
    assert match2 is not None, f"Regex did not match valid filename: {filename2}"
    date_str2, time_str2 = match2.groups()
    dt_in2 = datetime.strptime(date_str2 + time_str2, "%d%b%Y%H%M")
    dt_out2 = dt_in2 + timedelta(days=1)
    assert dt_in2 == datetime(2024, 12, 31, 23, 59)
    assert dt_out2 == datetime(2025, 1, 1, 23, 59) 
    
    # Test case 3: Invalid filename (should not match the full pattern)
    filename_invalid = "INVALID_FILENAME_FORMAT.h5"
    match_invalid = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", filename_invalid)
    assert match_invalid is None, f"Regex matched an invalid filename: {filename_invalid}"

    # Test case 4: Malformed month part that the regex should not match
    # Using a non-alphabetic character in the month part
    filename_malformed_month_char = "3RIMG_31X#Z2024_1234_L1C.h5" 
    match_malformed_month_char = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", filename_malformed_month_char)
    assert match_malformed_month_char is None, f"Regex matched malformed month with invalid char: {filename_malformed_month_char}"

    # Test case 5: Month part with incorrect length
    filename_malformed_month_len = "3RIMG_31XY2024_1234_L1C.h5"
    match_malformed_month_len = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", filename_malformed_month_len)
    assert match_malformed_month_len is None, f"Regex matched malformed month with incorrect length: {filename_malformed_month_len}"

    # Test case 6: Malformed time part (non-digits)
    filename_malformed_time = "3RIMG_10MAR2024_ABCD_L1C.h5"
    match_malformed_time = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", filename_malformed_time)
    assert match_malformed_time is None, f"Regex matched malformed time: {filename_malformed_time}"
