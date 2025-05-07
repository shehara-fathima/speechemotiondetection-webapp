import os
import librosa
import numpy as np
import pandas as pd
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tqdm import tqdm

# Map RAVDESS emotion codes to labels
RAVDESS_EMOTION_MAP = {
    "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
}

# Function to extract features
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        mfcc_length=13
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_length)       
        # Take the mean across time (optional, reduces variability)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc = np.concatenate((mfcc_mean, mfcc_std))

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
        energy = np.mean(librosa.feature.rms(y=y).T, axis=0)
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        pitches = pitch[pitch > 0]
        avg_pitch = np.mean(pitches) if pitches.size > 0 else 0
        # avg_pitch = np.mean(pitch[pitch > 0]) if pitch.any() else 0
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

        return mfcc, spectral_centroid, spectral_rolloff, energy, avg_pitch,chroma,mel
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None



# Function to parse emotion from filenames
def parse_emotion(file_name, dataset):
    if dataset == "Crema":
        # Extract emotion from CREMA-D filenames
        try:
            return file_name.split('_')[2]  # Example: '1001_DFA_ANG_XX.wav' -> 'ANG'
        except IndexError:
            return None  # Handle unexpected file name format
    elif dataset == "Ravdess":
        # Extract emotion from RAVDESS filenames
        try:
            code = file_name.split('-')[2]  # Example: '03-01-05-01-02-01-12.wav' -> '05'
            return RAVDESS_EMOTION_MAP.get(code, None)  # Map numeric code to emotion
        except IndexError:
            return None
    return None


# Function to process audio files and create a DataFrame
def process_dataset(folder_path, dataset):
    data = []
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features:
                mfcc, centroid, rolloff, energy, pitch,chroma,mel = features
                emotion = parse_emotion(file_name, dataset)
                feature_row = list(mfcc) + [centroid, rolloff, energy, pitch] + list(chroma) + list(mel) + [emotion]
                data.append(feature_row)

    return data

# Paths to CREMA-D and RAVDESS folders
crema_path = r"C:\Users\User\Desktop\Shehara\PROJECTS\Emotion detector\Crema"
ravdess_path = r"C:\Users\User\Desktop\Shehara\PROJECTS\Emotion detector\Ravdess"

# Process datasets
crema_data = process_dataset(crema_path, dataset="Crema")
ravdess_data = process_dataset(ravdess_path, dataset="Ravdess")

# Convert to DataFrame
mfcc_cols = [f"MFCC_{i}" for i in range(26)]         # MFCC mean + std (13 + 13)
chroma_cols = [f"Chroma_{i}" for i in range(12)]     # 12 chroma features
mel_cols = [f"Mel_{i}" for i in range(128)]          # 128 mel bands
columns = mfcc_cols + ["Spectral_Centroid", "Spectral_Rolloff", "Energy", "Pitch"] + chroma_cols + mel_cols + ["Emotion"]
crema_df = pd.DataFrame(crema_data, columns=columns)
ravdess_df = pd.DataFrame(ravdess_data, columns=columns)

# Combine datasets
final_df = pd.concat([crema_df, ravdess_df], ignore_index=True)
final_df.loc[final_df['Emotion'] == 'ANG', 'Emotion'] = 'Angry'
final_df.loc[final_df['Emotion'] == 'DIS', 'Emotion'] = 'Disappointed'
final_df.loc[final_df['Emotion'] == 'FEA', 'Emotion'] = 'Fearful'
final_df.loc[final_df['Emotion'] == 'HAP', 'Emotion'] = 'Happy'
final_df.loc[final_df['Emotion'] == 'SAD', 'Emotion'] = 'Sad'
final_df.loc[final_df['Emotion'] == 'NEU', 'Emotion'] = 'Neutral'

# Save to CSV
final_df.to_csv("audio_features.csv", index=False)
print("Feature extraction complete! Data saved to audio_features.csv.")
