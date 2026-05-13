# AIProj2: Campus Sustainability Object Detection using YOLOv8

## Overview

AIProj2 is the second phase of a campus sustainability object detection project. The project uses YOLOv8 to detect and classify common sustainability-related objects in images. It builds on the original project by improving the dataset, expanding class coverage, refining preprocessing steps, and evaluating the model on both validation data and unseen test images.

The system is designed to support automatic identification of waste-related objects and bins, which can be useful in smart campus, recycling awareness, and environmental monitoring applications.

## Objectives

The main objectives of this project are to:

- Improve the original YOLOv8 object detection model.
- Expand and refine the dataset used for training.
- Detect five sustainability-related object classes.
- Train and evaluate a YOLOv8 model using a structured pipeline.
- Test the final model on unseen images.
- Save training outputs, evaluation results, and prediction examples.

## Detected Classes

The model is trained to detect the following classes:

1. Plastic bottles
2. Paper waste
3. Cans
4. General bins
5. Recycling bins

## Project Structure

```text
AIProj2/
│
├── custom_dataset2/        # Final prepared dataset, excluded from GitHub
├── dataset_sources/        # Raw/source datasets, excluded from GitHub
├── demo_images/            # Demo images used for presentation or testing
├── results/                # Model outputs, evaluation results, and predictions
├── scripts/                # Python scripts used throughout the project
├── test_images/            # Unseen images used for final testing
│
├── .gitignore              # Files and folders excluded from GitHub
└── README.md               # Project documentation

scripts/
├── build_dataset.py        # Builds the final YOLO-compatible dataset
├── check.py                # Checks dataset paths, labels, and structure
├── count.py                # Counts images and class distribution
├── evaluate.py             # Evaluates the trained model
├── merge.py                # Merges multiple dataset sources
├── split.py                # Splits data into training, validation, and testing sets
├── test_unseen.py          # Runs predictions on unseen test images
└── train.py                # Trains the YOLOv8 model

Methodology

The project follows a complete object detection pipeline:

Dataset collection
Images were collected from multiple sources to represent different waste objects and bin types.
Dataset cleaning
Irrelevant, duplicate, unclear, or low-quality images were removed where necessary.
Dataset merging
Multiple image sources were combined into a single dataset structure.
Dataset balancing
Underrepresented classes were improved to reduce class imbalance and model bias.
Annotation formatting
The dataset was prepared in YOLO format, where each image has a corresponding label file containing bounding box coordinates and class IDs.
Dataset splitting
The final dataset was split into training, validation, and testing sets.
Model training
YOLOv8 was trained using the prepared dataset and selected training configuration.
Model evaluation
The trained model was evaluated using standard object detection metrics.
Unseen image testing
The final model was tested on new images that were not used during training.
Evaluation Metrics

The model is evaluated using the following metrics:

Precision: Measures how many predicted detections were correct.
Recall: Measures how many actual objects the model successfully detected.
mAP@0.5: Measures detection accuracy at an IoU threshold of 0.5.
mAP@0.5:0.95: Measures detection accuracy across multiple IoU thresholds.
Confusion Matrix: Shows correct and incorrect class predictions.
Precision-Recall Curve: Shows the trade-off between precision and recall.
F1-Confidence Curve: Helps identify a suitable confidence threshold.
Training Output

YOLOv8 automatically saves important training and evaluation files, such as:

Best model weights
Last model weights
Training curves
Validation results
Confusion matrix
Precision-recall curve
F1-confidence curve
Prediction examples

These outputs are stored in the results/ directory.

Ignored Files and Folders

The following folders are excluded from GitHub because they contain large dataset files:

custom_dataset2/
dataset_sources/

This keeps the repository clean and avoids uploading large raw datasets.

Installation

Install the required Python dependencies using:

pip install -r requirements.txt

If requirements.txt is not available yet, the main package required is:

pip install ultralytics
Running the Project

To train the model:

python scripts/train.py

To evaluate the model:

python scripts/evaluate.py

To test the model on unseen images:

python scripts/test_unseen.py
Technologies Used
Python
YOLOv8
Ultralytics
OpenCV
Google Colab
GitHub
VS Code
Author

Moustafa Sherif

Project Status

Completed as Part 2 of the AI campus sustainability object detection project.