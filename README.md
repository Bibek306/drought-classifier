# Drought Classification System

A 5-class drought severity classifier trained on 2.7M NOAA meteorological records using a TensorFlow neural network.

The model predicts drought severity levels (No Drought → D4 Exceptional) from weather and environmental features such as precipitation, humidity, temperature, pressure, and wind speed.

## Highlights
- Trained on large-scale NOAA climate data
- Time-based train/test split to prevent temporal leakage
- Balanced sampling to address 17:1 class imbalance
- Neural network built with TensorFlow/Keras
- Deployed with Streamlit on Hugging Face Spaces

## Tech Stack
TensorFlow • Scikit-learn • Pandas • Streamlit • Hugging Face

## Model
Input(18) → Dense(128) → Dense(64) → Dense(32) → Softmax(5)

## Results
- Macro F1 Score: 0.22
- D4 Exceptional Recall: 36%
- Accuracy: 49% (imbalanced dataset)

## Live Demo
https://huggingface.co/spaces/Bibek360/drought-classifier
