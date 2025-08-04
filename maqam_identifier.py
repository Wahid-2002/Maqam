import os
import librosa
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the dataset path
dataset_path = 'arabic_music_dataset'

# Define the maqams (folders) we want to identify
maqams = ['hijaz', 'rast', 'bayati', 'saba', 'nahawand', 'kurd', 'ajam', 'sikah']

# Function to extract features from an audio file
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

# Load the dataset
def load_dataset():
    features = []
    labels = []
    
    for maqam in maqams:
        maqam_path = os.path.join(dataset_path, maqam)
        if os.path.exists(maqam_path):
            for file in os.listdir(maqam_path):
                if file.endswith('.mp3') or file.endswith('.wav'):
                    file_path = os.path.join(maqam_path, file)
                    feature_vector = extract_features(file_path)
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(maqam)
    
    return np.array(features), np.array(labels)

# Load the dataset
print("Loading dataset...")
X, y = load_dataset()
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
print("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier
print("Evaluating classifier...")
y_pred = clf.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=maqams, yticklabels=maqams)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Save the model
joblib.dump(clf, 'maqam_identifier_model.joblib')
print("Model saved as maqam_identifier_model.joblib")
