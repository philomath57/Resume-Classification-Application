# Resume Classification and Job Matching Application

This Streamlit application classifies resumes into predefined job categories, compares them with a given job description, extracts matched skills, and provides model interpretability using LIME.

Features

- Resume Classification: Predicts the most likely job role from ten categories (e.g., Python Developer, Java Developer, Project Manager) using a pre-trained TF-IDF + XGBoost model.

- Prediction Confidence: Displays a bar chart showing confidence scores for all categories.

- Job Description Matching: Calculates similarity between the resume and job description using spaCy’s semantic similarity and visualizes it via a pie chart.

- Skill Extraction: Uses spaCy’s PhraseMatcher to detect predefined technical and soft skills mentioned in the resume.

- Model Explainability: Integrates LIME to highlight words or phrases contributing most to the classification, with both quick and detailed views.

# How to Use

- Enter the resume content in the provided text area.

- Optionally, enter a job description for similarity scoring.

- Click Show Prediction Results to view classification, similarity, matched skills, and LIME explanations.

# Requirements

- Python 3.x

- Libraries: streamlit, nltk, spacy, lime, matplotlib, joblib

This tool helps recruiters and candidates understand resume-job fit and the reasoning behind predictions.
