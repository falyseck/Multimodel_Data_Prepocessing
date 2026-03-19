# Multimodel Data Preprocessing

This project is a beginner-friendly multimodal machine learning system that combines:
- Face features from images
- Voice features from audio
- Customer behavior data (social + transactions)

The goal is to simulate **secure authentication + smart recommendation** in one flow.

## How the system functions

1. **Image processing**
   - `notebooks/image_processing.ipynb` loads images by person, applies augmentation, and extracts histogram features.
   - Output: `data/processed/image_features.csv`

2. **Audio processing**
   - `notebooks/audio_processing.ipynb` loads voice samples, applies augmentation, and extracts MFCC/energy features.
   - Output: `data/processed/audio_features.csv`

3. **Customer data merge**
   - `notebooks/data_merge.ipynb` merges social and transaction data, handles missing values, and prepares model-ready data.
   - Output: `data/processed/merged_customer_data.csv`

4. **Model training**
   - `notebooks/model_training.ipynb` trains 3 models:
     - `face_model.pkl`
     - `voice_model.pkl`
     - `product_model.pkl`

video demo
https://youtu.be/6d_DWc_dmdA

5. **Simulation runtime**
   - `simulation.py` loads trained models and processed datasets.
   - User enters face and voice identity candidates.
   - If face and voice predictions match, access is granted.
   - After access, the system predicts a recommended product category.
   - If predictions do not match, access is denied.

## Project structure
- `data/` raw and processed datasets
- `models/` trained model files (`.pkl`)
- `notebooks/` preprocessing + training notebooks
- `report/` project report
- `simulation.py` interactive simulation script

## Quick run idea
1. Run preprocessing notebooks to generate processed CSV files.
2. Run training notebook to generate model files in `models/`.
3. Execute `simulation.py` to test multimodal authentication and recommendation.

This repository is designed as a clear end-to-end ML prototype for learning and demonstration.