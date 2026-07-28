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

## Team — Group CV14
- Johnson, Emmanuel Okon — 23/EG/CV/016
- George, Richard Basil — 23/EG/CV/076
- Timothy, Ntiense Ima-Abasi — 23/EG/CV/036
- Isong, Oyokunyi Etim-James — 23/EG/CV/066
- Etokakpan, Nkutmfon Okokon — 23/EG/CV/006
- Thompson, Emaediong Ubong — 23/EG/CV/046
- Idobo, Goodnews Ita — 23/EG/CV/056
- Jack, George Boma — 23/EG/CV/086

## Challenges and Solutions
- Python/TensorFlow incompatibility on deployment: Streamlit Cloud defaulted to Python 3.14, which had no compatible TensorFlow build. Fixed by explicitly selecting Python 3.12 in Advanced Settings at deploy time.
- Blank deployed page with no server errors: traced to an accidentally truncated app.py (reduced to 7 lines during editing). Fixed by restoring the complete source.
- GitHub authentication failures: a fine-grained access token lacked write permissions; resolved by switching to a classic token with full repo scope.
- Class imbalance (2,491 SEK vs 722 AK) initially hurt AK recall in the custom CNN (around 21 percent); the transfer learning model closed this gap substantially through better feature representations from pretraining.
- Custom UI design: built a fully bespoke interface (branded header, medical disclaimer, styled result cards, confidence metrics) using Streamlit markdown and CSS rather than default components.
