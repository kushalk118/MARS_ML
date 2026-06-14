# Performance Evaluation Report

This report summarizes the performance of the Machine Learning (Random Forest) and Deep Learning (CNN) models for Deepfake Audio Detection.

## Target Thresholds vs. Achieved Results

| Metric | Target | Random Forest | PyTorch CNN | Status |
|---|---|---|---|---|
| **Overall Accuracy** | &ge; 80% | 100.00% | 100.00% | **PASSED** |
| **Equal Error Rate (EER)** | &le; 12% | 0.00% | 0.00% | **PASSED** |
| **F1-Score** | &ge; 80% | 100.00% | 100.00% | **PASSED** |
| **Genuine Class Acc.** | &ge; 75% | 100.00% | 100.00% | **PASSED** |
| **Deepfake Class Acc.** | &ge; 75% | 100.00% | 100.00% | **PASSED** |

## Confusion Matrices

### Random Forest Classifier
```
                Predicted Genuine   Predicted Deepfake
Actual Genuine   20                0                 
Actual Deepfake  0                 20                
```

### PyTorch CNN
```
                Predicted Genuine   Predicted Deepfake
Actual Genuine   20                0                 
Actual Deepfake  0                 20                
```

---
*Report generated automatically during training execution on 2026-06-14T17:55:16.*
