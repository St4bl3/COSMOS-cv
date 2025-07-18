# COSMOS-CV: Satellite Image Analysis for Weather Forecasting and Space Debris Detection

## Overview

**COSMOS-CV** is the computer vision component of the COSMOS satellite management software. This project leverages deep learning and distributed computing to analyze satellite imagery for two critical tasks:

1.  **Weather Nowcasting:** Predicting near-future weather patterns using multi-channel satellite data from INSAT-3DR.
2.  **Space Debris Detection:** Identifying and tracking space debris in orbital images to ensure satellite safety.

The repository includes Jupyter notebooks for model training, evaluation, and Flask applications to serve the trained models for real-world use. The workflows are designed to handle large-scale satellite data, incorporating technologies like Apache Spark and Hadoop for efficient data processing.

## 🚀 Features

* **Multi-Task Weather Forecasting:**
    * Utilizes a multi-channel approach, incorporating VIS, MIR, SWIR, WV, and TIR1 bands from INSAT-3DR satellite data.
    * Employs a ConvLSTM model architecture for spatio-temporal forecasting.
    * Predicts cloud cover, convection, fog, moisture, and temperature trends.
    * Includes a Flask web application for uploading satellite data and visualizing next-day forecasts.

* **Space Debris Detection:**
    * Fine-tunes a pre-trained Faster R-CNN model with a ResNet-50 backbone for object detection.
    * Processes image datasets and annotations from HDFS using PySpark.
    * Includes a Flask application to process video streams and identify potential debris, calculating their trajectories.

* **Scalable Data Processing:**
    * Demonstrates the use of **PySpark** and **Hadoop (HDFS)** for preprocessing large volumes of satellite imagery and annotation data.
    * Notebooks are configured to read data directly from HDFS, making the pipeline scalable.

* **Model Serving with Flask:**
    * Two separate Flask applications are provided to serve the trained weather and debris detection models.
    * The weather app allows users to upload HDF5 files and view generated forecast maps.
    * The debris app processes video files to detect and track objects of interest.

## 🛠️ Technologies Used

* **Machine Learning/Deep Learning:** TensorFlow, Keras, PyTorch, Scikit-learn
* **Web Framework:** Flask
* **Big Data:** Apache Spark (PySpark), Hadoop (HDFS)
* **Data Handling:** Pandas, NumPy, H5Py
* **Visualization:** Matplotlib, Cartopy, Seaborn
* **Containerization (Implied):** The use of distributed systems suggests that the applications are designed to be containerized with tools like Docker.

## 📂 Project Structure

COSMOS-CV/
│
├── checkpoints/              # Saved model checkpoints
├── static/                   # Static assets for Flask apps (images, CSS)
├── templates/                # HTML templates for Flask apps
├── uploads/                  # Directory for user-uploaded files
│
├── COSMOS_weather.ipynb      # Notebook for training the weather forecasting model
├── COSMOS_debris.ipynb       # Notebook for training the space debris detection model
├── COSMOS_weather_eval.ipynb # Notebook for evaluating weather model performance
├── COSMOS_debris_hadoop.ipynb # Notebook demonstrating debris detection with Hadoop integration
│
├── app.py                    # Flask application for the weather forecasting service
├── app_debris.py             # Flask application for the space debris detection service
│
└── README.md                 # This file


## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/COSMOS-CV.git](https://github.com/your-username/COSMOS-CV.git)
    cd COSMOS-CV
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    *It is highly recommended to create a `requirements.txt` file with all the necessary packages.*
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have the correct versions of CUDA and cuDNN installed if you plan to use a GPU for training.*

4.  **Hadoop/Spark Setup (for HDFS notebooks):**
    * Ensure that you have a working Hadoop and Spark environment.
    * Configure the HDFS paths and WebHDFS URLs in the relevant Jupyter notebooks (`COSMOS_debris_hadoop.ipynb`).

## 🚀 Usage

### Weather Forecasting App

1.  **Place the trained model** (`cp-20.weights.h5`) in the `checkpoints/` directory.

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```

3.  **Access the application** in your web browser at `http://127.0.0.1:5000`.

4.  **Upload** the required sequence of HDF5 satellite data files to generate a next-day forecast.

### Space Debris Detection App

1.  **Place the trained debris detection model** in the appropriate directory as specified in `app_debris.py`.

2.  **Run the Flask application:**
    ```bash
    python app_debris.py
    ```

3.  **Upload a video file** containing satellite footage to detect and analyze potential space debris.

## 🧠 Model Training

The Jupyter notebooks in this repository provide a detailed walkthrough of the model training process.

### Weather Forecasting Model

* **Notebook:** `COSMOS_weather.ipynb`
* **Data:** INSAT-3DR HDF5 files containing multi-channel data.
* **Process:**
    1.  The notebook loads sequences of HDF5 files.
    2.  It preprocesses the data by converting digital numbers (DN) to physical values using Look-Up Tables (LUTs).
    3.  The data is then structured into input sequences and corresponding target frames for next-day prediction.
    4.  A ConvLSTM model is trained on this time-series data.
    5.  Checkpoints are saved during training to allow for resumption.

### Space Debris Detection Model

* **Notebook:** `COSMOS_debris.ipynb`
* **Data:** A dataset of images with corresponding bounding box annotations for space debris.
* **Process:**
    1.  The notebook defines a custom PyTorch `Dataset` to load images and annotations.
    2.  It uses PySpark to preprocess annotation data, especially when dealing with large CSV files stored on HDFS.
    3.  A Faster R-CNN model with a ResNet-50 backbone is fine-tuned on the debris dataset.
    4.  The training process includes data augmentation and saves model checkpoints after each epoch.

## 🤝 Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
