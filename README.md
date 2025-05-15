# Pneumonia Detection Using CNN

This project uses Convolutional Neural Network (CNN) to classify chest X-ray images as either normal or showing pneumonia. 

## 📁 Structure 
- `data/`: Chest X-ray dataset from Kaggle
- `notebooks/`: Exploratory analysis
- `src/`: Model training and utility scripts
- `models/`: Saved trained models
- `outputs/`: Visualizations and predictions

## 🚀 How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/train_cnn.py`

## 🧠 Model
- Architecture: ResNet18 (pretrained)
- Loss: Cross Entropy
- Optimizer: Adam

## 📊 Evaluation
- Accuracy
- Precision / Recall
- Confusion Matrix

## 📦 Dataset
[Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

