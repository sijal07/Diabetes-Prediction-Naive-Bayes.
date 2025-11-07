# Diabetes-Prediction-Naive-Bayes.





# Naive Bayes Classifier for Diabetes Prediction

This project implements a Gaussian Naive Bayes classifier to predict diabetes outcomes using the Pima Indians Diabetes Dataset.

## Overview

The notebook demonstrates a complete machine learning workflow:
- Data loading and preprocessing
- Feature scaling using StandardScaler
- Model training with Gaussian Naive Bayes
- Prediction and evaluation
- Visualization of decision boundaries

## Dataset

The project uses the `diabetes.csv` dataset which contains medical predictor variables and a target variable indicating whether the patient has diabetes.

### Features:
- Multiple medical measurements (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- Target variable: Outcome (0 = No Diabetes, 1 = Diabetes)

## Implementation Details

### Libraries Used
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning algorithms and preprocessing

### Key Steps

1. **Data Loading**: Load the diabetes dataset and split into features (X) and target (y)
2. **Train-Test Split**: 75% training data, 25% testing data with random state for reproducibility
3. **Feature Scaling**: Standardize features using StandardScaler for better model performance
4. **Model Training**: Train Gaussian Naive Bayes classifier on the scaled training data
5. **Predictions**: 
   - Single sample prediction
   - Test set predictions
6. **Visualization**: 2D decision boundary plot using the first two features

## Model Performance

The Gaussian Naive Bayes classifier is implemented and tested on the diabetes dataset. The visualization shows the decision boundaries for the first two features, demonstrating how the model separates the classes.

## Usage

To run this project:

1. Ensure you have the required libraries installed:
```bash
pip install numpy pandas matplotlib scikit-learn
Place the diabetes.csv file in your working directory

Run the Jupyter notebook cells sequentially

File Structure
text
├── NAIVE BAYES.ipynb          # Main notebook with implementation
├── diabetes.csv               # Dataset file
└── README.md                  # This file
Results
The model provides predictions for diabetes outcomes based on patient medical data. The decision boundary visualization helps understand how the classifier distinguishes between diabetic and non-diabetic cases using the selected features.

