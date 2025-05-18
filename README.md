# Multi-Image Object Detection with YOLOv11

## Features

- **YOLOv11 Integration**: Uses the latest YOLOv11 model for accurate object detection
- **Multi-Image Processing**: Ability to upload and process multiple images at once
- **COCO Dataset Support**: Detects 80 common object classes from the COCO dataset
- **Visual Results**: Displays bounding boxes with labels inside them
- **Web Interface**: Simple and user-friendly Flask web application
- **GPU Acceleration**: Automatically uses CUDA if available for faster inference

### Setup Instruction

1. 
   ** If you install from Github, clone the repository:
   # Open Command Promt (Windows) /  Terminal (macOS/Linux)
   Direct to the path where you want to store application.
   Then insert these code respectively:
   
   "git clone https://github.com/Huygraf/Object-Detection.git"
   
   "cd Object_Detection"

   ** If you install from source folder, download archive file and extract them.

2. Install the required dependencies:
   - Direct to the path of your Application folder.
   - Insert this code to install required dependencies:
      "pip install -r requirements.txt"

## Usage

1. Start the Flask server:
   - Insert this code to run Flask server:
      "flask run --debug"

2. Open your web browser and go to:
      "http://127.0.0.1:5000"

3. Upload one or multiple images using the file selector

4. Click "Detect Objects" and wait for processing

5. View the results with bounding boxes and class labels for each detected object. 
   If you want to process other images, click "Process More Images" 


## Code Structure

- **app.py**: Main Flask application handling web routes and image uploads
  - Manages the web routes, file uploads, and rendering of templates
  - Processes multiple images and passes them to the detection utilities

- **util.py**: Core detection functionality
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

