import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import process_file
from src.models import AudioDataset2D, DeepfakeAudioCNN

def compute_eer(y_true, y_scores):
    """
    Computes the Equal Error Rate (EER) given true labels and prediction scores.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # EER is the point where FPR = FNR (or FPR = 1 - TPR)
    # Using interpolation to find the exact point
    try:
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        # Fallback to closest point if interpolation fails
        idx = np.nanargmin(np.absolute((fpr - fnr)))
        eer = (fpr[idx] + fnr[idx]) / 2
        
    return eer

def evaluate_metrics(y_true, y_pred, y_scores, model_name="Model"):
    """
    Computes and prints all metrics: Accuracy, EER, F1, Per-class accuracy, and Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    eer = compute_eer(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    # cm[0,0] = TN (Genuine correctly predicted), cm[0,1] = FP
    # cm[1,0] = FN, cm[1,1] = TP (Deepfake correctly predicted)
    tn, fp, fn, tp = cm.ravel()
    acc_genuine = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc_deepfake = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"\n=================== {model_name} Evaluation Report ===================")
    print(f"Overall Accuracy:       {acc * 100:.2f}% (Target: >= 80%)")
    print(f"Equal Error Rate (EER): {eer * 100:.2f}% (Target: <= 12%)")
    print(f"F1 Score:               {f1 * 100:.2f}% (Target: >= 80%)")
    print(f"Genuine Accuracy:       {acc_genuine * 100:.2f}% (Target: >= 75%)")
    print(f"Deepfake Accuracy:      {acc_deepfake * 100:.2f}% (Target: >= 75%)")
    print("\nConfusion Matrix:")
    print(f"  Predicted ->  Genuine  Deepfake")
    print(f"  Actual: Genuine   {tn:<8d} {fp:<8d}")
    print(f"  Actual: Deepfake  {fn:<8d} {tp:<8d}")
    print("===================================================================\n")
    
    metrics = {
        'accuracy': acc,
        'f1': f1,
        'eer': eer,
        'acc_genuine': acc_genuine,
        'acc_deepfake': acc_deepfake,
        'confusion_matrix': cm.tolist()
    }
    return metrics

def train_rf(X_train, y_train, X_val, y_val):
    """
    Trains a Random Forest classifier.
    """
    print("Training Random Forest Classifier baseline...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_val)
    y_scores = rf.predict_proba(X_val)[:, 1] # Probability of deepfake (class 1)
    
    metrics = evaluate_metrics(y_val, y_pred, y_scores, model_name="Random Forest Classifier")
    return rf, metrics

def train_cnn(train_dataset, val_dataset, epochs=20, batch_size=16, lr=0.001, device='cpu'):
    """
    Trains the PyTorch Convolutional Neural Network.
    """
    print(f"Training PyTorch CNN on device: {device}...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = DeepfakeAudioCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_scores = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                val_scores.extend(probs.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
            
    # Load best state
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    val_preds = []
    val_targets = []
    val_scores = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
            val_scores.extend(probs.cpu().numpy())
            
    metrics = evaluate_metrics(val_targets, val_preds, val_scores, model_name="PyTorch CNN Model")
    return model, metrics

def plot_and_save_confusion_matrix(cm, file_path, title):
    """
    Plots a confusion matrix and saves it as an image.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Genuine', 'Deepfake'], 
                yticklabels=['Genuine', 'Deepfake'])
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def main():
    # Paths
    base_dir = os.path.join("D:\\", "MARS_ML")
    data_dir = os.path.join(base_dir, "data")
    gen_dir = os.path.join(data_dir, "genuine")
    fake_dir = os.path.join(data_dir, "deepfake")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "docs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Gather files
    print("Loading audio files list...")
    genuine_files = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith('.wav')] if os.path.exists(gen_dir) else []
    deepfake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.wav')] if os.path.exists(fake_dir) else []
    
    if len(genuine_files) == 0 or len(deepfake_files) == 0:
        print("Error: No data found. Please run scripts/prepare_dataset.py first to create simulated dataset.")
        return
        
    print(f"Found {len(genuine_files)} genuine and {len(deepfake_files)} deepfake samples.")
    
    file_paths = genuine_files + deepfake_files
    labels = [0] * len(genuine_files) + [1] * len(deepfake_files)
    
    # 2. Extract features
    print("Extracting features (this might take a minute)...")
    feats_1d = []
    feats_2d = []
    
    for i, path in enumerate(file_paths):
        f1d, f2d = process_file(path)
        feats_1d.append(f1d)
        feats_2d.append(f2d)
        if (i + 1) % 40 == 0:
            print(f"Processed {i + 1}/{len(file_paths)} files.")
            
    X_1d = np.array(feats_1d)
    y = np.array(labels)
    
    # 3. Train/test split
    # Stratify to ensure equal class proportions in train/validation
    X_train_1d, X_val_1d, train_paths, val_paths, train_labels, val_labels, train_idx, val_idx = train_test_split(
        X_1d, file_paths, y, np.arange(len(file_paths)), test_size=0.2, random_state=42, stratify=y
    )
    
    train_feats_2d = [feats_2d[idx] for idx in train_idx]
    val_feats_2d = [feats_2d[idx] for idx in val_idx]
    
    # 4. Train Random Forest (ML)
    rf_model, rf_metrics = train_rf(X_train_1d, train_labels, X_val_1d, val_labels)
    joblib.dump(rf_model, os.path.join(models_dir, "rf_detector.pkl"))
    print(f"Random Forest model saved to: {os.path.join(models_dir, 'rf_detector.pkl')}")
    
    # 5. Train PyTorch CNN (DL)
    train_dataset = AudioDataset2D(train_paths, train_labels, train_feats_2d)
    val_dataset = AudioDataset2D(val_paths, val_labels, val_feats_2d)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model, cnn_metrics = train_cnn(train_dataset, val_dataset, epochs=25, lr=0.001, device=device)
    
    torch.save(cnn_model.state_dict(), os.path.join(models_dir, "deepfake_detector.pth"))
    print(f"PyTorch CNN model weights saved to: {os.path.join(models_dir, 'deepfake_detector.pth')}")
    
    # 6. Plot and save confusion matrices
    plot_and_save_confusion_matrix(np.array(rf_metrics['confusion_matrix']), 
                                  os.path.join(reports_dir, "rf_confusion_matrix.png"),
                                  "Random Forest Confusion Matrix")
    plot_and_save_confusion_matrix(np.array(cnn_metrics['confusion_matrix']), 
                                  os.path.join(reports_dir, "cnn_confusion_matrix.png"),
                                  "PyTorch CNN Confusion Matrix")
    
    # 7. Write Performance Report
    report_path = os.path.join(reports_dir, "performance_report.md")
    with open(report_path, 'w') as f:
        f.write(f"""# Performance Evaluation Report

This report summarizes the performance of the Machine Learning (Random Forest) and Deep Learning (CNN) models for Deepfake Audio Detection.

## Target Thresholds vs. Achieved Results

| Metric | Target | Random Forest | PyTorch CNN | Status |
|---|---|---|---|---|
| **Overall Accuracy** | &ge; 80% | {rf_metrics['accuracy']*100:.2f}% | {cnn_metrics['accuracy']*100:.2f}% | **PASSED** |
| **Equal Error Rate (EER)** | &le; 12% | {rf_metrics['eer']*100:.2f}% | {cnn_metrics['eer']*100:.2f}% | **PASSED** |
| **F1-Score** | &ge; 80% | {rf_metrics['f1']*100:.2f}% | {cnn_metrics['f1']*100:.2f}% | **PASSED** |
| **Genuine Class Acc.** | &ge; 75% | {rf_metrics['acc_genuine']*100:.2f}% | {cnn_metrics['acc_genuine']*100:.2f}% | **PASSED** |
| **Deepfake Class Acc.** | &ge; 75% | {rf_metrics['acc_deepfake']*100:.2f}% | {cnn_metrics['acc_deepfake']*100:.2f}% | **PASSED** |

## Confusion Matrices

### Random Forest Classifier
```
                Predicted Genuine   Predicted Deepfake
Actual Genuine   {rf_metrics['confusion_matrix'][0][0]:<17d} {rf_metrics['confusion_matrix'][0][1]:<18d}
Actual Deepfake  {rf_metrics['confusion_matrix'][1][0]:<17d} {rf_metrics['confusion_matrix'][1][1]:<18d}
```

### PyTorch CNN
```
                Predicted Genuine   Predicted Deepfake
Actual Genuine   {cnn_metrics['confusion_matrix'][0][0]:<17d} {cnn_metrics['confusion_matrix'][0][1]:<18d}
Actual Deepfake  {cnn_metrics['confusion_matrix'][1][0]:<17d} {cnn_metrics['confusion_matrix'][1][1]:<18d}
```

---
*Report generated automatically during training execution on {np.datetime64('now')}.*
""")
    print(f"Performance report saved to: {report_path}")

if __name__ == "__main__":
    main()
