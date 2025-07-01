import os
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

# Get the absolute path to the frontend directory
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, 
            static_folder=FRONTEND_DIR,
            static_url_path='',
            template_folder=FRONTEND_DIR)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
class_to_idx = None

try:
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
    else:
        checkpoint = torch.load(model_path, map_location=device)
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['class_to_idx']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        class_to_idx = checkpoint['class_to_idx']
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    if not model:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess the image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = list(class_to_idx.keys())[predicted.item()]
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({'prediction': predicted_class})
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000) 