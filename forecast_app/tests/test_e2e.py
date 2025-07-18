import pytest
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions # Renamed to avoid conflict
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# CHANNELS can be imported from app if needed for more detailed assertions
from app import CHANNELS as APP_CHANNELS, CHANNEL_DESCRIPTIONS as APP_CHANNEL_DESCRIPTIONS

@pytest.fixture(scope="module") # Changed to module to match threaded_flask_server if it becomes module-scoped
def browser():
    """Manages the Selenium WebDriver instance for a test module."""
    options = ChromeOptions()
    # options.add_argument("--headless") # Uncomment to run headless
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized") # Useful for visibility during debugging
    
    # Ensure chromedriver is in your PATH or specify its path using webdriver.Chrome(service=Service(executable_path=...))
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

@pytest.mark.e2e
def test_successful_upload_and_forecast(browser, threaded_flask_server, dummy_h5_file_path):
    """
    E2E test for uploading an H5 file and checking the results page.
    'threaded_flask_server' provides the base URL of the running app.
    'dummy_h5_file_path' is from conftest.py.
    """
    upload_page_url = f"{threaded_flask_server}/upload"
    print(f"Navigating to E2E test URL: {upload_page_url}") # For debugging
    browser.get(upload_page_url)

    # Wait for the upload page title
    WebDriverWait(browser, 10).until(
        EC.title_is("INSAT-3D Next-Day Forecast")
    )
    print("Upload page loaded.")

    # Find the file input element and send the dummy H5 file path
    file_input = browser.find_element(By.ID, "file") # Assuming your input has id="file"
    file_input.send_keys(dummy_h5_file_path) # send_keys needs an absolute path
    print(f"File selected: {dummy_h5_file_path}")

    # Find and click the submit button
    # Using a more specific XPath if multiple buttons exist
    submit_button = browser.find_element(By.XPATH, "//button[@type='submit' and contains(text(), 'Predict Next Day')]")
    submit_button.click()
    print("Submit button clicked.")

    # Wait for the results page to load by checking its title
    WebDriverWait(browser, 30).until( # Increased timeout for model prediction & plotting
        EC.title_is("Forecast Results")
    )
    print("Results page loaded.")

    # Assertions on the results page
    page_source = browser.page_source # Get page source once for multiple checks
    assert "Forecast Results" in page_source, "Results page heading not found."
    
    # Check for input and output datetime (based on dummy_h5_file_path's name "dummy_3RIMG_10MAR2025_0215_L1C_TEST.h5")
    assert "Input Time:" in page_source, "Input time label not found."
    assert "10 Mar 2025 02:15" in page_source, "Correct input date/time not found."
    assert "Forecast Time:" in page_source, "Forecast time label not found."
    assert "11 Mar 2025 02:15" in page_source, "Correct forecast date/time not found."

    # Check if images and descriptions for all channels are present
    for ch_name in APP_CHANNELS:
        # Check for the image with alt text
        img_element = browser.find_element(By.XPATH, f"//img[@alt='{ch_name}']")
        assert img_element.is_displayed(), f"Image for channel {ch_name} not displayed."
        img_src = img_element.get_attribute("src")
        assert f"pred_{ch_name}.png" in img_src, f"Image source for {ch_name} seems incorrect: {img_src}"
        print(f"Image for {ch_name} found and source looks okay.")

        # Check for channel caption (e.g., "VIS Forecast")
        caption_element = browser.find_element(By.XPATH, f"//div[@class='item']/p[@class='caption' and contains(text(), '{ch_name} Forecast')]")
        assert caption_element.is_displayed(), f"Caption for {ch_name} not found or incorrect."
        print(f"Caption for {ch_name} found.")

        # Check for channel description text (partial match)
        # Assuming APP_CHANNEL_DESCRIPTIONS is available
        description_snippet = APP_CHANNEL_DESCRIPTIONS[ch_name].split(':')[1][:30] # Get a snippet of the description
        desc_element_xpath = f"//div[@class='item']/p[@class='desc' and contains(text(), \"{description_snippet}\")]"
        try:
            desc_element = browser.find_element(By.XPATH, desc_element_xpath)
            assert desc_element.is_displayed(), f"Description for {ch_name} not displayed."
            print(f"Description for {ch_name} found.")
        except Exception as e:
            print(f"Could not find description for {ch_name} with snippet '{description_snippet}'. XPATH: {desc_element_xpath}")
            print(f"Page source snippet: {page_source[:2000]}") # Print some source to help debug
            raise e


    # Check for the "Back to Upload" link
    back_link = browser.find_element(By.LINK_TEXT, "← Back to Upload")
    assert back_link.is_displayed(), "Back to Upload link not found."
    print("Back to Upload link found.")

    # Optional: Give a moment to visually inspect if not running headless
    # time.sleep(2)
