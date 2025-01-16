# Predicting UEFA Champions League Match Outcome in FIFA

This repository contains the implementation and supporting files for predicting the outcomes of UEFA Champions League matches using data from FIFA. The project leverages machine learning to generate predictions and insights based on team and player data.

---

## Project Files

### **1. `app.py`**
- **Purpose**: The main application script responsible for hosting the prediction service.
- **Functionality**:
  - Accepts API requests with two soccer teams as input.
  - Interacts with the trained model to compute the winning probabilities for the teams.
  - Returns the predicted winning team and best squad suggestions via a RESTful API.
- **Usage**: Run this file to deploy the service locally or on a server.

---

### **2. `champs.csv`**
- **Purpose**: Contains the dataset used for training the machine learning model.
- **Details**:
  - Includes match data such as team names, player ratings, and past match outcomes.
  - Used in data preprocessing and model training.

---

### **3. `final-project.py`**
- **Purpose**: The exploratory and data preprocessing script.
- **Functionality**:
  - Performs data cleaning and feature engineering on the `champs.csv` dataset.
  - Evaluates initial models and visualizes performance metrics.
  - Includes exploratory data analysis (EDA) to understand key patterns in the data.
- **Usage**: Run this script to preprocess data and experiment with preliminary models.

---

### **4. `model_generator.py`**
- **Purpose**: The model training and saving script.
- **Functionality**:
  - Trains the machine learning model using the processed data.
  - Saves the trained model for deployment in the `app.py` script.
  - Implements hyperparameter tuning for optimizing model performance.
- **Usage**: Run this script to generate and save a trained model file.

---

### **5. `report.pdf`**
- **Purpose**: A comprehensive project report detailing the methodology, results, and analysis.
- **Contents**:
  - Overview of the project objective and problem statement.
  - Description of the dataset, preprocessing steps, and feature selection.
  - Details on the model training, evaluation metrics, and key findings.
  - Suggestions for future improvements and enhancements.

---
