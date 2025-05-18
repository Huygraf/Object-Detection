import os
import io
import base64
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from util import get_prediction, draw_boxes
import uuid # use for generating unique filename for uploaded images

# Create new Flask application
app = Flask(__name__)
# Specify the directory where uploaded and processed images will be stored
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Create endpoint
# Define the URLs that the application will respond 
@app.route('/')
# Active function when user navigates to the root URL
def index():
    return render_template('index.html')
# Associate the upload_image function with upload folder
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if reuqests include 'images' field
    if 'images' not in request.files:
        return redirect(url_for('index'))
    
    # List of uploaded images
    files = request.files.getlist('images')
    
    # If files has no element, return to home page
    if not files or files[0].filename == '':
        return redirect(url_for('index'))
    
    # List to store results
    # Include: file name, picture, number of detected object
    results = []
    
    # Check if the files are allowed image types
    allowed_extensions = {'jpg', 'jpeg', 'png', 'webp'}
    
    for file in files:
        # Check allowed type of file
        if '.' not in file.filename or file.filename.rsplit('.',1)[1].lower() not in allowed_extensions:
            continue
        
        # Process the image
        try:
            # Open file as an image with Pillow
            img = Image.open(file.stream)
            
            # Generate a UNIQUE filename
            filename = str(uuid.uuid4()) + '.jpg'
            # Create path to place storing processed image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Perform object detection
            boxes, labels, scores = get_prediction(img)
            
            # Draw bounding boxes on the image
            result_img = draw_boxes(img, boxes, labels, scores)
            
            # Save the result image
            result_img.save(img_path)
            
            # Convert the result image to base64 for display
            # HTML use base64 format
            buffered = io.BytesIO() # Temporarily store image data in memory
            
            # Save result image into memory stream
            result_img.save(buffered, format="JPEG") 
            
            # Retrive raw bytes from mamory stream 
            # Then encode to Base64 string
            # Final, convert to utf-8 string (use for HTML) 
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Add result to list
            results.append({
                'filename': file.filename,
                'img_data': img_str,
                'num_detections': len(boxes)
            })
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue
    # Save rendered results.html and results list
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True) 