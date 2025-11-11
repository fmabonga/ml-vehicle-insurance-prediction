# Vehicle Insurance Premium Prediction using Machine Learning

A machine-learning project for predicting vehicle insurance premiums using a publicly available dataset from Kaggle.  
The goal is to help customers estimate their vehicle insurance premiums so they can negotiate better with insurers and avoid being over-charged. This project was completed as part of the DataTalks Machine Learning Zoomcamp. 

---

## Table of Contents
- [Motivation](#motivation)  
- [Dataset](#dataset)  
- [Project Structure](#project-structure)  
- [Methodology / Workflow](#methodology--workflow)  
- [How to Run](#how-to-run)  
  - [Running Locally](#running-locally)  
  - [Running with Docker](#running-with-docker)  
- [Usage / Examples](#usage--examples)  
- [Model Performance](#model-performance)  
- [Future Work](#future-work)  
- [License](#license)  
- [Contact](#contact)

---

## Motivation

Vehicle insurance premiums vary widely based on many factors such as vehicle characteristics, driver demographics, location, and prior claims. Customers often lack a clear reference point for what they *should* be paying, which can lead to overcharging or difficulty negotiating with insurers.

This project aims to build a predictive model that allows customers to input vehicle and driver information and receive an estimated premium. This empowers users to:

- Get a **baseline premium estimate**  
- Compare insurer quotes with the predicted premium  
- Negotiate more effectively  
- Detect potential overcharging

---

## Dataset

This project uses the **Car Insurance Premium Dataset** from Kaggle.

Dataset:  
[Download from Kaggle](https://www.kaggle.com/api/v1/datasets/download/govindaramsriram/car-insurance-premium-dataset)

The dataset includes:

- Vehicle age  
- Vehicle model/type  
- Driver age  
- Region  
- Previous claims  
- Annual mileage  
- Insurance premium (target variable)

---


---

## Methodology / Workflow

The Jupyter notebook walks through the complete ML pipeline:

### 1. Load Dataset
Load the Kaggle dataset into a Pandas dataframe.

### 2. Exploratory Data Analysis (EDA)
- Visualize feature distributions  
- Identify missing values  
- Explore correlations  
- Detect outliers  

### 3. Data Cleaning & Feature Engineering
- Handle missing values  
- Encode categorical variables  
- Rename columns

### 4. Train/Test Split
Split the dataset into training and validation sets.

### 5. Model Training & Selection
Evaluate several ML algorithms:  
- Decision Tree  
- Random Forest Regressor  
- Gradient Boosting Regressor  

Tune hyperparameters and compare performance.

### 6. Model Evaluation
Metrics include:  
- RMSE (Root Mean Square Error)  

### 7. Save Final Model
Persist the best model using `pickle`.

### 8. Build Prediction API
A lightweight Flask API loads the model and serves predictions through `/predict`.

### 9. Dockerization
The API is containerized so it can be deployed anywhere with Docker.

---

## How to Run

### Running Locally

```bash
docker build -t insurance-premium-predictor .

docker run -d -p 5000:5000 --name premium-api insurance-premium-predictor
```

### Test with curl

```bash
curl -X POST http://localhost:9696/predict_premium \
  -H "Content-Type: application/json" \
  -d '{
        "driver_age": 30,
        "driver_experience": 5,
        "previous_accidents": 0,
        "annual_mileage_(x1000_km)": 15,
        "car_manufacturing_year": 2020,
        "car_age": 5
      }'
```

### Expected Output

```
{
  "insurance_premium_prediction": 493.3775939941406
}
```

### Deployed to AWS Elasticbeanstalk

http://insurance-premium-predictor-env.eba-qktrejxq.us-east-1.elasticbeanstalk.com


### Testing deplores api in AWS Elasticbeanstalk 
```
import requests
url = "http://insurance-premium-predictor-env.eba-qktrejxq.us-east-1.elasticbeanstalk.com/predict_premium"
client = {
        "driver_age": 30,
        "driver_experience": 5,
        "previous_accidents": 2,
        "annual_mileage": 15,
        "car_manufacturing_year": 2020,
        "car_age": 5
      }
requests.post(url, json=client).json()
```

### Expected Output

```
{
  "insurance_premium_prediction": 493.3775939941406
}
```