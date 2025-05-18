# Multi-Image Object Detection with YOLOv11

## Features

- **YOLOv11 Integration**: Uses the latest YOLOv11 model for accurate object detection
- **Multi-Image Processing**: Ability to upload and process multiple images at once
- **COCO Dataset Support**: Detects 80 common object classes from the COCO dataset
- **Visual Results**: Displays bounding boxes with labels inside them
- **Web Interface**: Simple and user-friendly Flask web application
- **GPU Acceleration**: Automatically uses CUDA if available for faster inference

# Installation Guide for Object Detection Application

This guide provides simple instructions to install and run the Object Detection application.

## Prerequisites - Must Install
- Python 3.8 or newer
- Internet connection (required for the first run to download model components)

## Installation

### Windows Users

1. Double-click the `install_and_run_windows.bat` file
2. The script will:
   - Check if Python is installed
   - Create a virtual environment
   - Install all required dependencies
   - Start the application automatically

3. When the application starts, open your web browser and go to: http://127.0.0.1:5000

### Mac/Linux Users

1. Open Terminal
2. Navigate to the project directory:

   cd path/to/project
 
3. Make the installation script executable:

   chmod +x install_and_run.sh

4. Run the installation script:

   ./install_and_run.sh

5. When the application starts, open your web browser and go to: http://127.0.0.1:5000

## Manual Installation (Alternative)

If you prefer to install dependencies manually:

1. Open Command Prompt (Windows) or Terminal (Mac/Linux)
2. Navigate to the project directory
3. Create a virtual environment (optional but recommended):
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install dependencies:

   pip install -r requirements.txt

5. Run the application:
   - Windows: `python -m flask run --debug`
   - Mac/Linux: `python3 -m flask run --debug`



## Usage

1. Start the Flask server (use for manual installation):
   - Insert this code to run Flask server:
      "flask run --debug"

2. Open your web browser and go to (use for manual installation):
      "http://127.0.0.1:5000"

3. Upload one or multiple images using the file selector

4. Click "Detect Objects" and wait for processing

5. View the results with bounding boxes and class labels for each detected object. 
   If you want to process other images, click "Process More Images" 


## Code Structure

- **app.py**: Main Flask application handling web routes and image uploads
  - Manages the web routes, file uploads, and rendering of templates
  - Processes multiple images and passes them to the detection utilities

- **procession.py**: Core detection functionality
  - `load_model()`: Loads the YOLOv11 model and configures it 
  - `get_prediction()`: Performs object detection on images
  - `draw_boxes()`: Draws bounding boxes and labels on detected objects

- **templates/**: HTML templates for the web interface
  - `index.html`: Upload page with multi-file selection
  - `results.html`: Results page showing all detected objects

## Model Configuration

The YOLOv11 model is configured with:

- Confidence threshold: 0.45 (minimum confidence for detection)
- IoU threshold: 0.50 (for Non-Maximum Suppression)
- Image size: 1280px (for inference)
- Auto-selects between CPU and CUDA (GPU) processing

