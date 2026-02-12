"""
Setup script to create LSTM model and MinMaxScaler for risk prediction.
Run once: python setup_models.py
"""

import os
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_scaler():
    """Create and save MinMaxScaler."""
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    
    # Sample sensor ranges: temp 15-35, humidity 30-90, oxygen 19-22, sound 20-100
    sample = np.array([
        [15, 30, 19, 20],
        [35, 90, 22, 100],
        [25, 65, 21, 50],
        [20, 50, 20.5, 35],
        [30, 80, 20, 70]
    ])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(sample)
    joblib.dump(scaler, MODELS_DIR / 'scaler.save')
    print("Scaler saved to models/scaler.save")


def create_lstm_model():
    """Create and save LSTM model for risk prediction."""
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(1, 4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Generate dummy training data
    np.random.seed(42)
    X = np.random.randn(500, 1, 4).astype(np.float32) * 0.5 + 0.5
    y = np.clip(np.random.rand(500, 1) * 0.8 + 0.1, 0, 1).astype(np.float32)
    
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    model.save(MODELS_DIR / 'lstm_model.keras')
    print("LSTM model saved to models/lstm_model.keras")


if __name__ == '__main__':
    create_scaler()
    create_lstm_model()
    print("Done. YOLOv8 model (yolov8s.pt) will auto-download on first run.")
