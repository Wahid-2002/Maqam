from flask import Flask, render_template, request, jsonify
import subprocess
import sys
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_tests', methods=['POST'])
def run_tests():
    try:
        # First, check if model exists, if not train it
        if not os.path.exists('maqam_identifier_model.joblib'):
            train_result = subprocess.run([sys.executable, 'maqam_identifier.py'], 
                                        capture_output=True, text=True, timeout=300)
            if train_result.returncode != 0:
                return jsonify({
                    'status': 'error',
                    'output': '',
                    'error': f"Training failed: {train_result.stderr}"
                })
        
        # Run the test script
        result = subprocess.run([sys.executable, 'test_maqam_identifier.py'], 
                              capture_output=True, text=True, timeout=300)
        
        return jsonify({
            'status': 'success',
            'output': result.stdout,
            'error': result.stderr
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            'status': 'error',
            'output': '',
            'error': 'Test execution timed out'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'output': '',
            'error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
