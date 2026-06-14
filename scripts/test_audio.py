import os
import sys
import argparse
import numpy as np
import torch
import joblib

# Add project root to sys.path to enable src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import process_file
from src.models import DeepfakeAudioCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Test deepfake audio detection on a single WAV file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the WAV audio file to test.")
    parser.add_argument("--model_type", type=str, choices=['rf', 'cnn'], default='rf', 
                        help="Model type to use for classification ('rf' for Random Forest, 'cnn' for PyTorch CNN).")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Optional custom path to the trained model file.")
    return parser.parse_args()

def predict_rf(audio_path, model_path):
    """
    Runs prediction using the Random Forest classifier.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
    model = joblib.load(model_path)
    
    # Extract 1D features
    feat_1d, _ = process_file(audio_path)
    
    # Predict
    prob = model.predict_proba([feat_1d])[0, 1] # Probability of being deepfake
    prediction = "Deepfake (AI-Generated)" if prob >= 0.5 else "Genuine (Human)"
    confidence = prob if prob >= 0.5 else (1.0 - prob)
    
    return prediction, confidence

def predict_cnn(audio_path, model_path):
    """
    Runs prediction using the PyTorch CNN model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
    # Instantiate model structure
    model = DeepfakeAudioCNN()
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Extract 2D features
    _, feat_2d = process_file(audio_path)
    
    # Format for PyTorch (batch, channel, n_mfcc, time_steps)
    feat_tensor = torch.tensor(feat_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        logit = model(feat_tensor).squeeze(0)
        prob = torch.sigmoid(logit).item()
        
    prediction = "Deepfake (AI-Generated)" if prob >= 0.5 else "Genuine (Human)"
    confidence = prob if prob >= 0.5 else (1.0 - prob)
    
    return prediction, confidence

def main():
    args = parse_args()
    
    base_dir = os.path.join("D:\\", "MARS_ML")
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at '{args.audio}'")
        return
        
    # Resolve default model paths
    if args.model_path is None:
        if args.model_type == 'rf':
            model_path = os.path.join(base_dir, "models", "rf_detector.pkl")
        else:
            model_path = os.path.join(base_dir, "models", "deepfake_detector.pth")
    else:
        model_path = args.model_path
        
    print(f"\nAnalyzing: {args.audio}")
    print(f"Using Model: {args.model_type.upper()} loaded from {model_path}")
    
    try:
        if args.model_type == 'rf':
            pred, conf = predict_rf(args.audio, model_path)
        else:
            pred, conf = predict_cnn(args.audio, model_path)
            
        print("\n" + "="*40)
        print(f"  PREDICTION RESULT  ")
        print("="*40)
        print(f"  Classification: {pred}")
        print(f"  Confidence:     {conf * 100:.2f}%")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Error executing prediction: {e}")

if __name__ == "__main__":
    main()
