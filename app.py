import os
import io
import base64
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from util import get_prediction, draw_boxes
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'images' not in request.files:
        return redirect(url_for('index'))
    
    files = request.files.getlist('images')
    
    if not files or files[0].filename == '':
        return redirect(url_for('index'))
    
    # List to store results
    results = []
    
    # Check if the files are allowed image types
    allowed_extensions = {'jpg', 'jpeg', 'png', 'webp'}
    
    for file in files:
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            continue
        
        # Process the image
        try:
            img = Image.open(file.stream)
            
            # Generate a unique filename
            filename = str(uuid.uuid4()) + '.jpg'
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Perform object detection
            boxes, labels, scores = get_prediction(img)
            
            # Draw bounding boxes on the image
            result_img = draw_boxes(img, boxes, labels, scores)
            
            # Save the result image
            result_img.save(img_path)
            
            # Convert the result image to base64 for display
            buffered = io.BytesIO()
            result_img.save(buffered, format="JPEG")
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
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True) 