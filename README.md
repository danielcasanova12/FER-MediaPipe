# FER-MediaPipe

This repository contains a project for Facial Emotion Recognition (FER) using MediaPipe for facial landmark detection and custom models for emotion classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The **FER-MediaPipe** project aims to classify facial emotions based on landmarks extracted using MediaPipe. This model detects key facial points and feeds them into a custom-trained classifier to predict one of several emotional states.

Emotions detected include:
- Anger
- Contempt
- Disgust
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

## Features

- Facial landmark extraction using [MediaPipe](https://google.github.io/mediapipe/).
- Emotion classification using a custom deep learning model.
- Integration with popular datasets like FER2013 and AffectNet.
- Real-time facial emotion detection through a webcam or static images.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/danielcasanova12/FER-MediaPipe.git
    cd FER-MediaPipe
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install additional dependencies for MediaPipe and machine learning:
    ```bash
    pip install mediapipe opencv-python tensorflow
    ```

4. If you encounter issues with `dlib`, ensure that `CMake` is installed and configured in your PATH.

## Usage

### Running Emotion Detection on Webcam

To run the real-time emotion detection using your webcam, use the following command:

```bash
python webcam_emotion_detector.py
Running Emotion Detection on Static Images
You can also run the detection on a set of static images. Modify the paths in the script to point to your image directory and run:

bash
Copiar código
python static_image_emotion_detector.py
Training the Model
If you want to retrain the model with new data:

Organize your dataset following this structure:

bash
Copiar código
dataset/
├── train/
│   ├── anger/
│   ├── happy/
│   └── ... (other emotions)
└── test/
    ├── anger/
    ├── happy/
    └── ... (other emotions)
Run the training script:

python train_model.py
Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue to report bugs or suggest improvements.
