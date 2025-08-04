from flask import Flask, render_template, request, jsonify, Response
import subprocess
import sys
import os
import json
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_tests', methods=['POST'])
def run_tests():
    try:
        # First, check if model exists, if not train it
        if not os.path.exists('maqam_identifier_model.joblib'):
            try:
                print("Starting model training...")
                train_result = subprocess.run([sys.executable, 'maqam_identifier.py'], 
                                            capture_output=True, text=True, timeout=600)
                
                if train_result.returncode != 0:
                    return jsonify({
                        'status': 'error',
                        'output': train_result.stdout,
                        'error': f"Training failed: {train_result.stderr}"
                    })
                
                print("Model training completed successfully")
            except subprocess.TimeoutExpired:
                return jsonify({
                    'status': 'error',
                    'output': '',
                    'error': 'Model training timed out (10+ minutes)'
                })
        
        # Check if the model was actually created
        if not os.path.exists('maqam_identifier_model.joblib'):
            return jsonify({
                'status': 'error',
                'output': '',
                'error': 'Model file was not created after training'
            })
        
        # Run the test script
        try:
            print("Running tests...")
            result = subprocess.run([sys.executable, 'test_maqam_identifier.py'], 
                                  capture_output=True, text=True, timeout=600)
            
            return jsonify({
                'status': 'success',
                'output': result.stdout,
                'error': result.stderr
            })
        except subprocess.TimeoutExpired:
            return jsonify({
                'status': 'error',
                'output': '',
                'error': 'Test execution timed out (10+ minutes)'
            })
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'output': '',
            'error': f"Server error: {str(e)}"
        })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
