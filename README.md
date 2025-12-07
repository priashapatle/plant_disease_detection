🌱 Plant Disease Detection Using Deep Learning

EfficientNet-B0 Based Image Classifier + FastAPI Backend + Streamlit Dashboard + Grad-CAM Heatmaps

📌 Project Overview

This project detects plant leaf diseases using deep learning.
It includes:

✔ EfficientNet-B0 trained model
✔ FastAPI backend for predictions
✔ Streamlit dashboard UI
✔ Grad-CAM heatmaps for explainability
✔ User-friendly interface to upload images

📂 Repository Structure
plant_disease_detection/
├── app/                → backend + dashboard
├── assets/             → logo + sample images
├── data/               → training/validation dataset
├── models/             → trained models
├── src/                → model training, inference, gradcam

🚀 Features
✔ Deep Learning Model

EfficientNet-B0

Trained on PlantVillage — New Plant Diseases Dataset (Augmented)

38 plant disease categories

✔ Backend (FastAPI)

/predict endpoint

Accepts image file and returns prediction + confidence

✔ Dashboard (Streamlit)

Upload leaf image

Shows predicted disease & confidence

Displays Grad-CAM heatmap

Simple and modern UI

✔ Explainability

Grad-CAM heatmaps show which area of the leaf the model focuses on.
