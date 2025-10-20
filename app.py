from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import json
from datetime import datetime
import uuid
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
app.secret_key = 'pneumonia_detection_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('database', exist_ok=True)

# Updated PneumoniaClassifier class to match the trained model
class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        
        # Use ResNet50 as backbone (matching the trained model)
        self.backbone = models.resnet50(pretrained=False)
        
        # Replace the final fully connected layer with the same architecture
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Load model - CORRECTED VERSION
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading model on: {device}")
    
    # Create model instance
    model = PneumoniaClassifier(num_classes=2)
    
    # Load the checkpoint
    checkpoint = torch.load('models/best_improved_model.pth', map_location=device)
    
    # Check if it's a checkpoint dictionary or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # It's a checkpoint dictionary, extract the model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from checkpoint - Val Acc: {checkpoint.get('val_acc', 'Unknown')}%")
    else:
        # It's directly the state_dict
        model.load_state_dict(checkpoint)
        print("✅ Model state_dict loaded directly")
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, device

# Initialize model and device
model, device = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'PNEUMONIA']

def save_analysis_history(prediction_data):
    """Save analysis results to JSON database"""
    try:
        history_file = 'database/analysis_history.json'
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add unique ID and timestamp
        prediction_data['id'] = str(uuid.uuid4())[:8]
        prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        history.append(prediction_data)
        
        # Keep only last 100 analyses
        if len(history) > 100:
            history = history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        return prediction_data['id']
    except Exception as e:
        print(f"Error saving history: {e}")
        return None

def get_analysis_history():
    """Get analysis history from database"""
    try:
        history_file = 'database/analysis_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def analyze_image(image, filename="unknown"):
    """Analyze image and return detailed results"""
    try:
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get class probabilities
        normal_prob = probabilities[0][0].item() * 100
        pneumonia_prob = probabilities[0][1].item() * 100
        
        # Determine result
        prediction = class_names[predicted.item()]
        confidence_percent = confidence.item() * 100
        
        # Risk assessment
        if pneumonia_prob > 70:
            risk_level = "HIGH"
            recommendation = "Consult a healthcare professional immediately"
            icon = "🚨"
        elif pneumonia_prob > 40:
            risk_level = "MEDIUM"
            recommendation = "Recommend further medical evaluation"
            icon = "⚠️"
        else:
            risk_level = "LOW"
            recommendation = "No immediate action needed, but regular checkups recommended"
            icon = "✅"
        
        # Create annotated image
        annotated_image = create_annotated_image(image, prediction, confidence_percent)
        
        # Convert image to base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'prediction': prediction,
            'confidence': round(confidence_percent, 2),
            'normal_probability': round(normal_prob, 2),
            'pneumonia_probability': round(pneumonia_prob, 2),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'icon': icon,
            'image': f"data:image/png;base64,{img_str}",
            'filename': filename,
            'model_used': 'ResNet18 v2.1',
            'analysis_time': datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def create_annotated_image(image, prediction, confidence):
    """Create an annotated version of the image with results"""
    # Resize image for display
    display_image = image.resize((400, 400))
    
    # Create a new image with extra space for text
    annotated = Image.new('RGB', (500, 500), color='white')
    annotated.paste(display_image, (50, 50))
    
    # Add text annotations
    draw = ImageDraw.Draw(annotated)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Add title
    draw.text((150, 10), "Pneumonia Analysis", fill='black', font=title_font)
    
    # Add prediction with color
    color = 'green' if prediction == 'NORMAL' else 'red'
    draw.text((50, 460), f"Result: {prediction}", fill=color, font=font)
    draw.text((50, 480), f"Confidence: {confidence:.1f}%", fill=color, font=font)
    
    return annotated

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Routes
@app.route('/')
def index():
    """Home page"""
    history = get_analysis_history()
    stats = {
        'total_analyses': len(history),
        'normal_count': len([h for h in history if h.get('prediction') == 'NORMAL']),
        'pneumonia_count': len([h for h in history if h.get('prediction') == 'PNEUMONIA']),
        'recent_analyses': history[-5:] if history else []
    }
    return render_template('index.html', stats=stats)

@app.route('/upload')
def upload_page():
    """Single image upload page"""
    return render_template('upload.html')

@app.route('/batch')
def batch_upload_page():
    """Batch upload page"""
    return render_template('batch_upload.html')

@app.route('/history')
def history_page():
    """Analysis history page"""
    history = get_analysis_history()
    return render_template('history.html', history=history)

@app.route('/about')
def about_page():
    """About page"""
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze_single_image():
    """Analyze a single image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG)'})
    
    try:
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Analyze image
        result = analyze_image(image, file.filename)
        
        if result is None:
            return jsonify({'error': 'Error analyzing image'})
        
        # Save to history
        save_analysis_history(result)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in analyze_single_image: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch_images():
    """Analyze multiple images"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'})
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'})
    
    results = []
    errors = []
    
    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            try:
                image = Image.open(file.stream).convert('RGB')
                result = analyze_image(image, file.filename)
                
                if result:
                    # Add batch info
                    result['batch_index'] = i + 1
                    result['batch_total'] = len(files)
                    results.append(result)
                    
                    # Save to history
                    save_analysis_history(result)
                else:
                    errors.append(f"Error analyzing {file.filename}")
                    
            except Exception as e:
                errors.append(f"Error processing {file.filename}: {str(e)}")
        else:
            errors.append(f"Invalid file: {file.filename}")
    
    return jsonify({
        'success': True,
        'total_processed': len(results),
        'total_errors': len(errors),
        'results': results,
        'errors': errors
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image = Image.open(file.stream).convert('RGB')
        result = analyze_image(image, file.filename)
        
        if result:
            # Save to history
            save_analysis_history(result)
            
            # Return minimal API response
            api_response = {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'normal_probability': result['normal_probability'],
                'pneumonia_probability': result['pneumonia_probability'],
                'risk_level': result['risk_level'],
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(api_response)
        else:
            return jsonify({'error': 'Analysis failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def api_history():
    """API endpoint to get analysis history"""
    history = get_analysis_history()
    return jsonify({'history': history})

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """API endpoint to get statistics"""
    history = get_analysis_history()
    
    stats = {
        'total_analyses': len(history),
        'normal_count': len([h for h in history if h.get('prediction') == 'NORMAL']),
        'pneumonia_count': len([h for h in history if h.get('prediction') == 'PNEUMONIA']),
        'average_confidence': np.mean([h.get('confidence', 0) for h in history]) if history else 0,
        'last_analysis': history[-1] if history else None
    }
    
    return jsonify(stats)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    try:
        history_file = 'database/analysis_history.json'
        if os.path.exists(history_file):
            os.remove(history_file)
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/result/<result_id>')
def result_detail(result_id):
    """Detailed result page for a specific analysis"""
    history = get_analysis_history()
    result = next((h for h in history if h.get('id') == result_id), None)
    
    if result:
        return render_template('results.html', result=result)
    else:
        return render_template('error.html', message="Result not found"), 404

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', message="Page not found"), 404

if __name__ == '__main__':
    print("🚀 Starting Pneumonia Detection System...")
    print("📍 Access the application at: http://localhost:5000")
    print("📊 Model loaded and ready for analysis!")
    app.run(debug=True, host='0.0.0.0', port=5000)