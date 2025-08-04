import os
import librosa
import numpy as np
import joblib
import json

# Load the trained model
model_path = 'maqam_identifier_model.joblib'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please run the training script first.")
    exit(1)

print("Loading trained model...")
try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Rest of the script remains the same...

# Define the maqams (same as in training)
maqams = ['hijaz', 'rast', 'bayati', 'saba', 'nahawand', 'kurd', 'ajam', 'sikah']

# Function to extract features (same as in training)
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Extract features
        # 1. Chroma features (for maqam identification)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # 2. MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # 3. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroids)
        spectral_centroid_std = np.std(spectral_centroids)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)
        
        # 4. Tempo features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 5. Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
        
        # Combine all features
        features = np.concatenate([
            chroma_mean, chroma_std,
            mfcc_mean, mfcc_std,
            [spectral_centroid_mean, spectral_centroid_std],
            [spectral_bandwidth_mean, spectral_bandwidth_std],
            [tempo],
            tonnetz_mean, tonnetz_std
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load test dataset
def load_test_dataset():
    test_path = 'arabic_music_dataset/test'
    features = []
    file_names = []
    
    if os.path.exists(test_path):
        for file in os.listdir(test_path):
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_path = os.path.join(test_path, file)
                feature_vector = extract_features(file_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    file_names.append(file)
    
    return np.array(features), file_names

# Load test dataset
print("Loading test dataset...")
X_test, file_names = load_test_dataset()
if len(X_test) == 0:
    print("No test files found. Please add some test files to the 'arabic_music_dataset/test' folder.")
else:
    print(f"Test dataset loaded: {X_test.shape[0]} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Calculate confidence scores
    confidence_scores = model.predict_proba(X_test)
    
    # Display results
    print("\nTest Results:")
    print("=" * 50)
    for i, (file_name, prediction, confidence) in enumerate(zip(file_names, predictions, confidence_scores)):
        # Get the confidence for the predicted class
        predicted_class = prediction
        confidence = confidence[list(model.classes_).index(predicted_class)]
        
        print(f"File: {file_name}")
        print(f"Predicted Maqam: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 30)
    
    # Summary statistics
    print("\nSummary:")
    unique, counts = np.unique(predictions, return_counts=True)
    for maqam, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        print(f"{maqam}: {count} files ({percentage:.1f}%)")
