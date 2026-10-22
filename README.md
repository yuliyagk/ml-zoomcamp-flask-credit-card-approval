## ML Model Evaluation and Deployment with Flask (ML Zoomcamp)

This project is part of the Machine Learning Zoomcamp by DataTalksClub.

It explores credit card approval prediction using logistic regression, evaluates classification performance and provides a Flask API for serving a saved model.

## Project Overview

The goal of this project is to:

* Prepare numerical and categorical credit application data
* Train a logistic regression model
* Evaluate predictions using ROC AUC, precision, recall and F1
* Explore classification thresholds and regularization
* Apply five-fold cross-validation
* Save models and feature encoders with Pickle
* Serve predictions through a Flask API

## Dataset

The training scripts use the AER credit card dataset.

The target variable is `card`, encoded as:

* `1` — application approved
* `0` — application not approved

The training workflow uses features including reports, age, income, expenditure, dependents, home ownership and employment status.

Dataset:
https://github.com/alexeygrigorev/datasets/blob/master/AER_credit_card_data.csv

## Tech Stack

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* Logistic Regression
* DictVectorizer
* Pickle
* Flask
* Requests
* Docker configuration

## Workflow

Data → Preparation → Feature Encoding → Model Training → Evaluation → Serialization → Flask API → Predictions

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask requests
```

The repository does not include pinned dependency versions. Loading the saved models may require versions compatible with the original training environment.

### 2. Explore model evaluation

```bash
python evalutionmlbookcamp.py
```

The script includes:

* A train/validation/test split
* Feature-level ROC AUC analysis
* Logistic regression training
* Precision, recall and F1 calculations across thresholds
* Five-fold cross-validation
* Comparison of regularization settings

### 3. Run a local prediction

The repository includes `model1.bin` and `dv.bin`, which are loaded by the prediction scripts.

```bash
python predict_model.py
```

This prints a prediction probability for a sample input.

### 4. Start the Flask API

```bash
python predict_service.py
```

The service starts at:

```text
http://localhost:9696
```

It exposes a POST endpoint at `/predict`.

The application uses Flask's development server with debug mode enabled and is intended for local experimentation.

### 5. Make a prediction

Run the included example client:

```bash
python test.py
```

Or send a request directly:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "reports": 0,
    "share": 0.245,
    "expenditure": 3.438,
    "owner": "yes"
  }'
```

The response contains:

* `credit_probability` — the model's positive-class probability, rounded to three decimal places
* `credit_accepted` — whether the rounded probability is at least 0.5

### 6. Explore model training

```bash
python train_model.py
```

This script downloads the dataset, trains and evaluates logistic regression models, and saves a feature encoder and model together in `model_C1.bin`.

**Note:** The Flask service loads the separate files `model1.bin` and `dv.bin`. Running the training script does not update the API's model automatically. The saved tuple currently comes from the final cross-validation fold rather than a final refit on the full training data.

## Project Status

* The repository contains training, evaluation, local inference and API code.
* Evaluation scripts calculate metrics, but no benchmark results are reported here.
* The training and serving scripts use different model artifacts and input examples; these should be aligned for a reproducible end-to-end workflow.
* The Dockerfile contains an initial container configuration. It does not explicitly copy the Flask service or define its startup command.
* Local execution and container deployment have not been verified in this documentation update.

## Use Cases

* Practicing binary classification with tabular data
* Understanding classification thresholds and metric trade-offs
* Comparing regularization settings with cross-validation
* Serving saved machine learning models through an API

This is an educational exercise, not a validated credit decision system.

## Learning Outcomes

Through this project I practiced:

* Preparing numerical and categorical features
* Encoding records with DictVectorizer
* Training logistic regression models
* Evaluating predictions with ROC AUC
* Exploring precision, recall and F1 across thresholds
* Applying cross-validation
* Saving and loading models with Pickle
* Creating a Flask prediction endpoint
* Sending JSON requests to an ML service

## Acknowledgments

This project is based on the Machine Learning Zoomcamp by DataTalksClub:

https://github.com/DataTalksClub/machine-learning-zoomcamp
