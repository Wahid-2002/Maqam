from flask import Flask, render_template, request, jsonify, Response
import os
import librosa
import numpy as np
import joblib
import threading
import time
import subprocess
import sys
import logging
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier  # Much lighter than Random Forest

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the maqams
maqams = ['hijaz', 'rast', 'bayati', 'saba', 'nahawand', 'kurd', 'ajam', 'sikah']

# Global variables
model = None
model_lock = threading.Lock()
training_in_progress = False
training_status = ""
training_progress = 0
training_start_time = None
training_end_time = None
training_error = None
training_files_processed = 0
training_total_files = 0

# Function to extract minimal features from an audio file
def extract_features(file_path):
    try:
        # Load only 5 seconds of audio at 11025 Hz (half the sample rate)
        y, sr = librosa.load(file_path, sr=11025, duration=5)
        
        # Extract only chroma features (most important for maqam identification)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Extract only 5 MFCCs (reduced from 13)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Combine features (only 12 + 5 = 17 features total)
        features = np.concatenate([chroma_mean, mfcc_mean])
        
        return features
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

# Function to load the dataset with batching
def load_dataset():
    global training_files_processed, training_total_files
    
    features = []
    labels = []
    
    # Count total files first for progress tracking
    training_total_files = 0
    for maqam in maqams:
        maqam_path = os.path.join('arabic_music_dataset', maqam)
        if os.path.exists(maqam_path):
            for file in os.listdir(maqam_path):
                if file.endswith('.mp3') or file.endswith('.wav'):
                    training_total_files += 1
    
    # Process files with memory management
    training_files_processed = 0
    for maqam in maqams:
        maqam_path = os.path.join('arabic_music_dataset', maqam)
        if os.path.exists(maqam_path):
            for file in os.listdir(maqam_path):
                if file.endswith('.mp3') or file.endswith('.wav'):
                    file_path = os.path.join(maqam_path, file)
                    
                    # Update status with current file
                    global training_status
                    training_status = f"Processing {maqam}/{file}..."
                    
                    feature_vector = extract_features(file_path)
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(maqam)
                    
                    training_files_processed += 1
                    # Update progress
                    training_progress = int((training_files_processed / training_total_files) * 100)
                    
                    # Force garbage collection every 5 files
                    if training_files_processed % 5 == 0:
                        import gc
                        gc.collect()
    
    return np.array(features), np.array(labels)

# Function to train the model
def train_model():
    global model, training_in_progress, training_status, training_progress, training_start_time, training_end_time, training_error
    
    with model_lock:
        if training_in_progress:
            return "Training already in progress"
        
        training_in_progress = True
        training_status = "Initializing training..."
        training_progress = 0
        training_start_time = datetime.now()
        training_end_time = None
        training_error = None
        training_files_processed = 0
        training_total_files = 0
    
    try:
        logger.info("Starting model training")
        
        # Load the dataset
        training_status = "Loading dataset..."
        X, y = load_dataset()
        
        if len(X) == 0:
            raise Exception("No valid audio files found in the dataset")
        
        training_status = f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features"
        logger.info(training_status)
        
        # Train a simple Decision Tree classifier (much lighter than Random Forest)
        training_status = "Training Decision Tree classifier..."
        logger.info(training_status)
        training_progress = 70  # Update progress
        
        from sklearn.model_selection import train_test_split
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a very simple model
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)  # Very shallow tree
        clf.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(clf, 'maqam_identifier_model.joblib')
        logger.info("Model saved")
        
        # Load the model into memory
        model = clf
        
        training_status = "Model trained and saved successfully"
        training_progress = 100
        logger.info(training_status)
        
        training_end_time = datetime.now()
        return "Model trained successfully"
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        training_status = f"Error: {str(e)}"
        training_error = str(e)
        training_end_time = datetime.now()
        return f"Error: {str(e)}"
    finally:
        with model_lock:
            training_in_progress = False

# Initialize the model
def initialize_model():
    global model
    try:
        if os.path.exists('maqam_identifier_model.joblib'):
            model = joblib.load('maqam_identifier_model.joblib')
            logger.info("Model loaded successfully")
        else:
            logger.info("Model not found, training...")
            train_model()
    except Exception as e:
        logger.error(f"Error initializing model: {e}")

# Routes
@app.route('/')
def index():
    # Get list of test files
    test_dir = 'arabic_music_dataset/test'
    test_files = []
    
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) 
                     if f.endswith('.mp3') or f.endswith('.wav')]
    
    return render_template('index.html', test_files=test_files)

@app.route('/run_tests', methods=['POST'])
def run_tests():
    global model
    
    try:
        # First, check if model exists, if not train it
        if model is None and not os.path.exists('maqam_identifier_model.joblib'):
            train_result = train_model()
            if "Error" in train_result:
                return jsonify({
                    'status': 'error',
                    'output': '',
                    'error': f"Training failed: {train_result}"
                })
        
        # Load the model if not already loaded
        if model is None:
            model = joblib.load('maqam_identifier_model.joblib')
        
        # Get test files
        test_dir = 'arabic_music_dataset/test'
        if not os.path.exists(test_dir):
            return jsonify({
                'status': 'error',
                'output': '',
                'error': 'Test directory not found'
            })
        
        test_files = [f for f in os.listdir(test_dir) 
                     if f.endswith('.mp3') or f.endswith('.wav')]
        
        if len(test_files) == 0:
            return jsonify({
                'status': 'error',
                'output': '',
                'error': 'No test files found'
            })
        
        # Process each test file
        results = []
        for filename in test_files:
            file_path = os.path.join(test_dir, filename)
            
            # Extract features
            features = extract_features(file_path)
            if features is None:
                results.append({
                    'file': filename,
                    'error': 'Could not extract features'
                })
                continue
            
            # Reshape features for prediction
            features = features.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get confidence scores
            confidence_scores = model.predict_proba(features)[0]
            confidence = confidence_scores[list(model.classes_).index(prediction)]
            
            results.append({
                'file': filename,
                'prediction': prediction,
                'confidence': f"{confidence:.2%}"
            })
        
        # Format output
        output = "Test Results:\n"
        output += "=" * 50 + "\n"
        
        for result in results:
            if 'error' in result:
                output += f"File: {result['file']}\n"
                output += f"Error: {result['error']}\n"
                output += "-" * 30 + "\n"
            else:
                output += f"File: {result['file']}\n"
                output += f"Predicted Maqam: {result['prediction']}\n"
                output += f"Confidence: {result['confidence']}\n"
                output += "-" * 30 + "\n"
        
        # Add summary
        output += "\nSummary:\n"
        predictions = [r['prediction'] for r in results if 'prediction' in r]
        if predictions:
            unique, counts = np.unique(predictions, return_counts=True)
            for maqam, count in zip(unique, counts):
                percentage = (count / len(predictions)) * 100
                output += f"{maqam}: {count} files ({percentage:.1f}%)\n"
        
        return jsonify({
            'status': 'success',
            'output': output,
            'error': ''
        })
    except Exception as e:
        logger.error(f"Error in run_tests: {str(e)}")
        return jsonify({
            'status': 'error',
            'output': '',
            'error': f"Server error: {str(e)}"
        })

@app.route('/train', methods=['POST'])
def start_training():
    global training_status, training_error
    
    # Reset error state
    training_error = None
    
    # Start training in a separate thread
    thread = threading.Thread(target=train_model)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'Training started',
        'message': 'Model training has started in the background'
    })

@app.route('/training_status')
def get_training_status():
    try:
        # Calculate elapsed time
        elapsed_time = ""
        if training_start_time:
            if training_end_time:
                elapsed = (training_end_time - training_start_time).total_seconds()
            else:
                elapsed = (datetime.now() - training_start_time).total_seconds()
            
            minutes, seconds = divmod(int(elapsed), 60)
            elapsed_time = f"{minutes}m {seconds}s"
        
        # Calculate files per second if training is in progress
        files_per_second = ""
        if training_in_progress and training_files_processed > 0:
            elapsed = (datetime.now() - training_start_time).total_seconds()
            if elapsed > 0:
                files_per_second = f"{training_files_processed / elapsed:.2f} files/sec"
        
        return jsonify({
            'status': training_status,
            'in_progress': training_in_progress,
            'progress': training_progress,
            'elapsed_time': elapsed_time,
            'files_processed': training_files_processed,
            'total_files': training_total_files,
            'files_per_second': files_per_second,
            'error': training_error
        })
    except Exception as e:
        logger.error(f"Error in training_status: {str(e)}")
        return jsonify({
            'status': f"Error: {str(e)}",
            'in_progress': False,
            'progress': 0,
            'elapsed_time': "",
            'files_processed': 0,
            'total_files': 0,
            'files_per_second': "",
            'error': str(e)
        })

@app.route('/predict/<filename>')
def predict(filename):
    global model
    
    if model is None:
        return jsonify({'error': 'Model not available. Please train the model first.'})
    
    file_path = os.path.join('arabic_music_dataset/test', filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': f'File {filename} not found in test directory.'})
    
    # Extract features
    features = extract_features(file_path)
    
    if features is None:
        return jsonify({'error': f'Could not extract features from {filename}.'})
    
    try:
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get confidence scores
        confidence_scores = model.predict_proba(features)[0]
        confidence = confidence_scores[list(model.classes_).index(prediction)]
        
        # Get top 3 predictions
        top_indices = np.argsort(confidence_scores)[-3:][::-1]
        top_predictions = [
            {
                'maqam': model.classes_[i],
                'confidence': f"{confidence_scores[i]:.2%}"
            } for i in top_indices
        ]
        
        return jsonify({
            'file': filename,
            'prediction': prediction,
            'confidence': f"{confidence:.2%}",
            'top_predictions': top_predictions
        })
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/predict_all', methods=['POST'])
def predict_all():
    global model
    
    if model is None:
        return jsonify({'error': 'Model not available. Please train the model first.'})
    
    test_dir = 'arabic_music_dataset/test'
    
    if not os.path.exists(test_dir):
        return jsonify({'error': 'Test directory not found.'})
    
    results = []
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            file_path = os.path.join(test_dir, filename)
            
            # Extract features
            features = extract_features(file_path)
            
            if features is not None:
                try:
                    # Reshape features for prediction
                    features = features.reshape(1, -1)
                    
                    # Make prediction
                    prediction = model.predict(features)[0]
                    
                    # Get confidence scores
                    confidence_scores = model.predict_proba(features)[0]
                    confidence = confidence_scores[list(model.classes_).index(prediction)]
                    
                    results.append({
                        'file': filename,
                        'prediction': prediction,
                        'confidence': f"{confidence:.2%}"
                    })
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    results.append({
                        'file': filename,
                        'error': str(e)
                    })
    
    return jsonify({'results': results})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize the model
    initialize_model()
    
    # Run the app
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
