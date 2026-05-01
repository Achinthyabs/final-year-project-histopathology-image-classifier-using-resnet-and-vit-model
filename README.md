# Histopathology Classification using ResNet-8

This project predicts histopathology tissue classes using a trained ResNet-8 model.

## Files
- `app.py` → Streamlit prediction app
- `evaluate_cnn.py` → evaluation script
- `train_resnet8.py` → training script
- `resnet8_12class_model.keras` → trained model
- `requirements.txt` → dependencies

## Classes
ADI, Adenocarcinoma, BACK, DEB, High-grade IN, LYM, Low-grade IN, MUC, MUS, NORM, Polyp, STR

## Setup
1. Install Python 3.10 or 3.11
2. Open terminal in project folder
3. Create virtual environment:
   ```bash
   python -m venv venv

   venv\Scripts\activate

   pip install -r requirements.txt