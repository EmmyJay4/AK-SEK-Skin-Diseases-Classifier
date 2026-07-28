# AK vs SEK Skin Lesion Classifier

A Streamlit web application for binary image classification of skin lesions: Actinic Keratosis (AK) vs Seborrheic Keratoses (SEK), built for GET 324 (AI/ML) Mini-Project - Group CV14.

## Dataset
Sourced from Kaggle: FYP Skin Disease Dataset by bilalmanzoor2.
https://www.kaggle.com/datasets/bilalmanzoor2/fyp-skin-disease-dataset

## Model
Transfer learning model (MobileNetV3Small, ImageNet weights, fine-tuned head) achieving 93.4% test accuracy.

## How to Use
1. Visit the deployed app link.
2. Upload a skin lesion image (jpg/jpeg/png).
3. View the predicted class and confidence scores.

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Team
Isong, Oyokunyi Etim-James 
23/EG/CV/066

## Challenges
- Add challenges encountered
