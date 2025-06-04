import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from config import Config
from llm_service import LLMService
from dfine_service import DFineService

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# Initialize services
llm_service = LLMService()
dfine_service = DFineService()

# Create necessary directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        prompt = request.form.get('prompt', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not prompt.strip():
            return jsonify({'error': 'Please provide a description of what you want to detect'}), 400
        
        if file and Config.allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            try:
                # Get detection parameters from LLM
                print(f"Processing prompt: {prompt}")
                parameters = llm_service.parse_user_prompt(prompt)
                print(f"LLM generated parameters: {parameters}")
                print(f"Detection mode: {parameters.get('detection_mode', 'unknown')}")
                print(f"Count only: {parameters.get('count_only', False)}")
                print(f"ROI detection: {parameters.get('roi_detection', False)}")
                
                # Use D-Fine for detection
                results = dfine_service.process_media(file_path, parameters)
                
                if 'error' in results:
                    return jsonify({'error': results['error']}), 500
                
                print(f"Processing completed successfully: {results.get('filename', 'No output file')}")
                
                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return jsonify({
                    'success': True,
                    'results': results,
                    'parameters': parameters,
                    'prompt': prompt
                })
                
            except Exception as processing_error:
                print(f"Processing error: {processing_error}")
                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'Processing failed: {str(processing_error)}'}), 500
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload an image or video file.'}), 400
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/outputs/<filename>')
def uploaded_file(filename):
    """Serve output files with proper headers"""
    try:
        file_path = os.path.join(Config.OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/jpeg', as_attachment=False)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-llm', methods=['POST'])
def test_llm():
    """Test LLM service endpoint"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        parameters = llm_service.parse_user_prompt(prompt)
        return jsonify({'parameters': parameters})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting AI-Powered Object Detection System...")
    print(f"🌐 Available at: http://{Config.HOST}:{Config.PORT}")
    print("\n📋 Supported Detection Modes:")
    print("   ✓ Count-only mode: 'count people in the image'")
    print("   ✓ Specific region counting: 'count people in specific area'")
    print("   ✓ Video tracking: 'track people in video' (with unique counting)")
    print("   ✓ Standard detection: 'find all people and cars'")
    print("   ✓ Area filtering: 'detect cars in foreground/background/center'")
    print("\n🎯 Region Types Supported:")
    print("   • Specific regions (foreground/background/center/left/right/top/bottom)")
    print("   • Custom areas based on description")
    print("\n📹 Video Processing:")
    print("   • First frame analysis (quick)")
    print("   • Every second analysis (detailed with unique person tracking)")
    print("   • Frame-by-frame display in frontend")
    print("\n🔧 Features:")
    print("   • Unique person counting (no duplicate counting across frames)")
    print("   • Processed frame visualization")
    print("   • Simplified detection modes")
    print("\n🔧 Powered by D-Fine + Groq Llama AI")
    print("🎯 Enhanced with D-Fine for superior object detection accuracy!")
    
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT) 
    