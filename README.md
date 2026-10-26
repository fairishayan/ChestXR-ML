# 🏥 ChestXR - Pneumonia Detection (ML Backend)

An end-to-end Deep Learning backend to detect Pneumonia from Chest X-rays. This API takes an X-ray image, processes it using a Convolutional Neural Network, and returns the prediction with a confidence score.

## 📊 Model Performance

Trained on the Kermany Pediatric Chest X-Ray dataset (5,863 images), the model was optimized for high sensitivity (Recall) to ensure no Pneumonia cases are missed.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 87.3% |
| **Pneumonia Recall** | 98.0% (High sensitivity) |
| **ROC-AUC** | 0.99 |
| **Threshold** | 0.80 (Tuned to minimize false positives) |

## 🧠 Technical Highlights
- **Architecture:** `MobileNetV3-Large` (Transfer Learning, ImageNet weights).
- **Imbalance Handling:** Addressed class imbalance (Normal vs. Pneumonia) using `WeightedRandomSampler` during training.
- **Fine-Tuning:** Unfroze the last two feature blocks of the model for medical feature extraction.
- **Inference:** Built with **FastAPI** for high-performance, asynchronous API serving.

## 📁 Project Structure
```text
├── app/
│   └── main.py          # FastAPI application and endpoints
├── src/
│   ├── dataset.py       # Data loaders, transforms, and oversampling
│   ├── model.py         # MobileNetV3 architecture setup
│   ├── train.py         # Training loop & validation
│   ├── evaluate.py      # Test set evaluation (Confusion Matrix, F1)
│   ├── threshold.py     # Threshold tuning script
│   └── predict.py       # CLI script for single image inference
├── artifacts/           # Saved model weights (best_model.pt)
└── requirements.txt     # Python dependencies
