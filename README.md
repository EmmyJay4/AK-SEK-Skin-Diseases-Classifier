# AK vs SEK Skin Lesion Classifier 🩺

An end-to-end deep learning pipeline that tells Actinic Keratosis (AK) and Seborrheic Keratoses (SEK) apart from a single uploaded photo — built, trained, deployed, and debugged from scratch by Group CV14 for the GET 324 (AI/ML) Mini-Project, Department of Civil Engineering, University of Uyo.

Two competing architectures were trained and benchmarked head-to-head before picking a winner, and the final model is live behind a custom-designed Streamlit interface featuring a branded header, a medical disclaimer banner, and styled result cards — no generic templates, no default styling.

## Live App
https://ak-sek-skin-diseases-classifier.streamlit.app

## Dataset
Sourced from Kaggle: FYP Skin Disease Dataset by bilalmanzoor2 — 2,491 SEK images and 722 AK images, isolated from the full multi-class dataset and re-split into train/val/test sets.
https://www.kaggle.com/datasets/bilalmanzoor2/fyp-skin-disease-dataset

## The Approach
Two models were trained and compared:

| Model | Test Accuracy | Notes |
|---|---|---|
| Custom CNN (from scratch) | 87.9% | 3-block Conv2D architecture, batch norm + dropout |
| MobileNetV3Small (transfer learning) | 93.8% | ImageNet-pretrained backbone, fine-tuned head — selected for deployment |

Training used GPU acceleration on Google Colab, with data augmentation (flip, rotation, zoom), early stopping, and adaptive learning rate reduction.

## How to Use
1. Visit the deployed app link above.
2. Upload a skin lesion image (jpg/jpeg/png).
3. Instantly view the predicted class (AK or SEK) with confidence scores for both.

## Run Locally
## Run Locally
pip install -r requirements.txt
streamlit run app.py


## Team — Group CV14

| # | Name | Registration Number | GitHub Username |
|---|------|---------------------|------------------|
| 1 | Johnson, Emmanuel Okon | 23/EG/CV/016 | EmmyJay4 |
| 2 | George, Richard Basil | 23/EG/CV/076 | imohabasi97-ship-it |
| 3 | Timothy, Ntiense Ima-Abasi | 23/EG/CV/036 | Timmy2006-maker |
| 4 | Isong, Oyokunyi Etim-James | 23/EG/CV/066 | isongjames40-hash |
| 5 | Etokakpan, Nkutmfon Okokon | 23/EG/CV/006 | Nkutmfonokokon-deep |
| 6 | Thompson, Emaediong Ubong | 23/EG/CV/046 | EmaediongThompson |
| 7 | Idobo, Goodnews Ita | 23/EG/CV/056 | goodnewsidobo-spec |
| 8 | Jack, George Boma | 23/EG/CV/086 | jackgeorge193-blip |

## Challenges and Solutions
- Python/TensorFlow incompatibility on deployment: Streamlit Cloud defaulted to Python 3.14, which had no compatible TensorFlow build. Fixed by explicitly selecting Python 3.12 in Advanced Settings at deploy time.
- Blank deployed page with no server errors: traced to an accidentally truncated app.py (reduced to 7 lines during editing). Fixed by restoring the complete source.
- GitHub authentication failures: a fine-grained access token lacked write permissions; resolved by switching to a classic token with full repo scope.
- Class imbalance (2,491 SEK vs 722 AK) initially hurt AK recall in the custom CNN (around 21 percent); the transfer learning model closed this gap substantially through better feature representations from pretraining.
- Custom UI design: built a fully bespoke interface (branded header, medical disclaimer, styled result cards, confidence metrics) using Streamlit markdown and CSS rather than default components.
