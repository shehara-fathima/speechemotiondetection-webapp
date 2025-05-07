import streamlit as st
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
import json
import pandas as pd

# Function to load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_lottieur(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottiefile(r'C:\Users\User\Desktop\Shehara\PROJECTS\Emotion detector\lottiefile.json')
st_lottie(lottie_animation, height=300, key="emotion")

# Load the models
histgradient_model = joblib.load('histgradientmodel.pkl')
keras_model = load_model('kerasmodel.h5')

# Load the encoder and scaler
label_encoder = joblib.load('label_encoder.pkl')  # Load label encoder
scaler = joblib.load('scaler.pkl')  # Load scaler

# Emotion labels and reverse encoding
emotion_labels = {
    0: 'Angry',
    1: 'Disappointed',
    2: 'Fearful',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad'
}

# Function to clean columns by removing square brackets and converting to numeric
def clean_features(df):
    columns_to_clean = ['Spectral_Centroid', 'Spectral_Rolloff', 'Energy']
    for col in columns_to_clean:
        # Check if the column contains strings
        if df[col].apply(lambda x: isinstance(x, str)).all():
            df[col] = df[col].str.replace(r'[\[\]]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # If it's a list or array, extract the first element
        elif df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).all():
            df[col] = df[col].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    return df

# Feature extraction
def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # Extract features using librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc = np.concatenate((mfcc_mean, mfcc_std))

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
    energy = np.mean(librosa.feature.rms(y=y).T, axis=0)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitch[pitch > 0]) if pitch.any() else 0
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    features = np.concatenate([mfcc, spectral_centroid, spectral_rolloff, energy, [avg_pitch], chroma, mel])
    return features

# Preprocess features (scaling and encoding)
def preprocess_features(features):
    # Apply Standard Scaling
    features_scaled = scaler.transform(features.reshape(1, -1)).flatten()  # Use the loaded scaler to scale the features
    return features_scaled

def predict_emotion(model, features):
    # Make a prediction using the selected model
    features_reshaped = features.reshape(1, -1)  # Reshape for model input
    if model == 'HistGradientBoosting Classifier':
        prediction = histgradient_model.predict(features_reshaped)
    elif model == 'Tensorflow Keras Sequential':
        prediction = keras_model.predict(features_reshaped)
        prediction = np.argmax(prediction, axis=1)  # For multi-class classification
    return emotion_labels[int(prediction)]

# Streamlit app UI
st.title("Emotion Detection from Speech")
st.write("Upload an audio file and select a model to predict emotion.")

# Audio file upload
audio_file = st.file_uploader("Upload Audio File", type=["wav"])

# Model selection
model_choice = st.selectbox("Select Model", ["HistGradientBoosting Classifier", "Tensorflow Keras Sequential"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')  # Play the uploaded audio file

    # Extract features from the audio file
    features = extract_features(audio_file)

    # Preprocess features by cleaning and scaling
    features_cleaned = pd.DataFrame([features], columns=[['MFCC_0', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13', 'MFCC_14', 'MFCC_15', 'MFCC_16', 'MFCC_17', 'MFCC_18', 'MFCC_19', 'MFCC_20', 'MFCC_21', 'MFCC_22', 'MFCC_23', 'MFCC_24', 'MFCC_25', 'Spectral_Centroid', 'Spectral_Rolloff', 'Energy', 'Pitch', 'Chroma_0', 'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4', 'Chroma_5', 'Chroma_6', 'Chroma_7', 'Chroma_8', 'Chroma_9', 'Chroma_10', 'Chroma_11', 'Mel_0', 'Mel_1', 'Mel_2', 'Mel_3', 'Mel_4', 'Mel_5', 'Mel_6', 'Mel_7', 'Mel_8', 'Mel_9', 'Mel_10', 'Mel_11', 'Mel_12', 'Mel_13', 'Mel_14', 'Mel_15', 'Mel_16', 'Mel_17', 'Mel_18', 'Mel_19', 'Mel_20', 'Mel_21', 'Mel_22', 'Mel_23', 'Mel_24', 'Mel_25', 'Mel_26', 'Mel_27', 'Mel_28', 'Mel_29', 'Mel_30', 'Mel_31', 'Mel_32', 'Mel_33', 'Mel_34', 'Mel_35', 'Mel_36', 'Mel_37', 'Mel_38', 'Mel_39', 'Mel_40', 'Mel_41', 'Mel_42', 'Mel_43', 'Mel_44', 'Mel_45', 'Mel_46', 'Mel_47', 'Mel_48', 'Mel_49', 'Mel_50', 'Mel_51', 'Mel_52', 'Mel_53', 'Mel_54', 'Mel_55', 'Mel_56', 'Mel_57', 'Mel_58', 'Mel_59', 'Mel_60', 'Mel_61', 'Mel_62', 'Mel_63', 'Mel_64', 'Mel_65', 'Mel_66', 'Mel_67', 'Mel_68', 'Mel_69', 'Mel_70', 'Mel_71', 'Mel_72', 'Mel_73', 'Mel_74', 'Mel_75', 'Mel_76', 'Mel_77', 'Mel_78', 'Mel_79', 'Mel_80', 'Mel_81', 'Mel_82', 'Mel_83', 'Mel_84', 'Mel_85', 'Mel_86', 'Mel_87', 'Mel_88', 'Mel_89', 'Mel_90', 'Mel_91', 'Mel_92', 'Mel_93', 'Mel_94', 'Mel_95', 'Mel_96', 'Mel_97', 'Mel_98', 'Mel_99', 'Mel_100', 'Mel_101', 'Mel_102', 'Mel_103', 'Mel_104', 'Mel_105', 'Mel_106', 'Mel_107', 'Mel_108', 'Mel_109', 'Mel_110', 'Mel_111', 'Mel_112', 'Mel_113', 'Mel_114', 'Mel_115', 'Mel_116', 'Mel_117', 'Mel_118', 'Mel_119', 'Mel_120', 'Mel_121', 'Mel_122', 'Mel_123', 'Mel_124', 'Mel_125', 'Mel_126', 'Mel_127']
    ])  # Create a DataFrame for feature columns

    features_cleaned = clean_features(features_cleaned)  # Clean the features (remove square brackets and convert to numeric)
    
    # Preprocess the features by scaling
    features_scaled = preprocess_features(features_cleaned.values.flatten())

    # Predict emotion using the selected model
    if st.button("Predict Emotion"):
        emotion = predict_emotion(model_choice, features_scaled)
        st.markdown(
            f"""
            <div style='text-align:center; padding: 1rem; border-radius: 10px;
                        background-color:#ffdddd; color:#b30000; font-size:24px;
                        font-weight:bold;'>
                🎯 {emotion}
            </div>
            """,
            unsafe_allow_html=True
        )

