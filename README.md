# Customer Churn Prediction System

A machine-learning project that estimates whether a bank customer is likely to stay or exit. The project provides a Streamlit web interface that accepts customer details and displays a predicted outcome with churn and stay probabilities.

> **Project status:** Local Streamlit execution is demonstrated in the project screenshots. Model-wide evaluation metrics and the complete training/preprocessing workflow should be confirmed from the notebook and source code before being reported as verified results.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Application Inputs](#application-inputs)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Run the Application](#run-the-application)
- [Machine-Learning Workflow](#machine-learning-workflow)
- [Example Output](#example-output)
- [Known Technical Note](#known-technical-note)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Purpose](#project-purpose)

## Overview

Customer churn refers to a customer discontinuing a service or relationship with a business. This project demonstrates how a classification model can be connected to a web application so a user can enter customer attributes and view an estimated churn outcome.

The interface identifies **Random Forest** as the model algorithm. Predictions are intended for learning and demonstration, not as a guarantee of customer behaviour or as the sole basis for business decisions.

## Features

- Interactive Streamlit interface
- Customer information input form
- Predicted stay/exit outcome
- Churn and stay probability display
- Model-confidence indicator shown by the interface
- Random Forest classification model

## Application Inputs

The supplied interface displays fields for:

- Credit score
- Age
- Tenure in years
- Account balance
- Number of products
- Credit-card status
- Active-member status
- Estimated salary
- Country
- Gender

The exact input encoding and the final feature columns passed to the model should be checked in `app.py` and the training notebook. User-facing fields and model features are not necessarily the same count because categorical fields may be encoded into multiple columns.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Random Forest classifier
- StandardScaler (as indicated by the observed runtime warning)

The project also includes a Jupyter notebook and saved model/preprocessing files, as listed in the repository structure below. Confirm the exact filenames in the repository before relying on this example structure.

## Project Structure

```text
Customer-Churn-Prediction/
├── app.py
├── random_forest_churn_model.pkl
├── scaler.pkl
├── Churn_Modelling.csv
├── Python_Implementation_for_churn_prediction.ipynb
├── requirements.txt
└── README.md
```

## Setup and Installation

### 1. Open the project directory

Open Anaconda Prompt or a terminal and navigate to the folder containing `app.py` and `requirements.txt`.

### 2. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

If you use a dedicated Python environment, activate it before installing the dependencies.

## Run the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, commonly:

```text
http://localhost:8501
```

Open the displayed local URL in your browser. If Streamlit prints a different port, use the URL shown in your terminal.

## Machine-Learning Workflow

The project is intended to follow a workflow similar to the one below. Confirm the training steps and their order against the actual notebook before treating every step as verified.

```text
Customer dataset
       ↓
Data preparation and preprocessing
       ↓
Categorical encoding / feature preparation
       ↓
Scaling, if required by the implementation
       ↓
Trained classification model
       ↓
Prediction through the Streamlit interface
       ↓
Predicted outcome and probability display
```

## Example Output

In the supplied application screenshot, one example prediction displays:

- **Predicted outcome:** Customer Will Stay
- **Churn probability:** 7.00%
- **Stay probability:** 93.00%
- **Model confidence:** 93.00%

These values describe the specific example shown in the screenshot. They are not the model's overall accuracy, and probability calibration has not been independently verified.

## Known Technical Note

The supplied terminal screenshot shows a scikit-learn warning indicating that input data does not have valid feature names even though `StandardScaler` was fitted with feature names.

Before presenting the project as final, inspect the training notebook and `app.py` to ensure that the same feature names, column order, encoding, and preprocessing are used during training and prediction. Where appropriate, a scikit-learn `Pipeline` can help keep preprocessing and prediction steps together. Retest the application after any change.

## Limitations

- The dataset details and complete preprocessing process should be documented from the actual project files.
- Do not publish model accuracy, precision, recall, F1-score, or other metrics unless they are calculated and reproducible from a suitable evaluation procedure.
- A prediction is an estimate and does not guarantee that a customer will stay or leave.
- Customer information should be handled responsibly and should not be exposed in screenshots or public examples.

## Future Improvements

- Add batch prediction from a CSV file
- Add prediction history, where appropriate
- Build a customer analytics dashboard
- Add feature-importance analysis or SHAP explanations
- Evaluate the model using a held-out test set and report suitable classification metrics
- Resolve and test the preprocessing feature-name warning
- Consider deployment after validation and security review

## Project Purpose

This project demonstrates an end-to-end machine-learning application concept: taking customer attributes, generating a classification output, and presenting that output through an interactive Streamlit interface.

**For educational and project-demonstration purposes.**

## References

- Project repository: https://github.com/YashKharat-9124/ML-Projects-/tree/main/Churn%20Prediction
- Streamlit documentation: https://docs.streamlit.io/
- scikit-learn documentation: https://scikit-learn.org/stable/
- Pandas documentation: https://pandas.pydata.org/docs/
