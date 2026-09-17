# Breast Cancer Detection Using Machine Learning

A machine learning project for classifying breast cancer cases as **Malignant** or **Benign** using the Breast Cancer Wisconsin (Diagnostic) dataset.

The project covers the complete machine learning workflow, including data exploration, visualization, preprocessing, model training, model comparison, hyperparameter tuning, evaluation, cross-validation, and model serialization.

---

## Project Overview

Breast cancer detection is a binary classification problem where machine learning can be used to identify patterns in diagnostic measurements.

In this project, the **Breast Cancer Wisconsin (Diagnostic)** dataset available through `scikit-learn` is used to train and evaluate multiple classification algorithms.

The final model is an **XGBoost Classifier**, which achieved an accuracy of approximately **98.25%** on the held-out test set used in the notebook.

> **Note:** This project is intended for educational and machine learning demonstration purposes. It is not a medical diagnostic system and should not be used for real-world medical decisions.

---

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset provided by `sklearn.datasets.load_breast_cancer()`.

### Dataset Statistics

| Property            | Value |
| ------------------- | ----: |
| Total Samples       |   569 |
| Predictive Features |    30 |
| Target Classes      |     2 |
| Malignant Samples   |   212 |
| Benign Samples      |   357 |
| Missing Values      |  None |

The 30 numerical features describe characteristics of cell nuclei obtained from digitized images of fine needle aspirates (FNA) of breast masses.

The features include measurements related to:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave Points
* Symmetry
* Fractal Dimension

Each group contains mean, standard error, and worst-case measurements.

---

## Machine Learning Workflow

The notebook follows an end-to-end machine learning workflow:

```text
Dataset Loading
       ↓
DataFrame Creation
       ↓
Data Exploration
       ↓
Data Visualization
       ↓
Feature / Target Separation
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Multiple Model Training
       ↓
Model Evaluation
       ↓
XGBoost Tuning
       ↓
Final Model Evaluation
       ↓
Cross-Validation
       ↓
Model Serialization
```

---

## Models Implemented

Several classification algorithms were explored during the project:

1. Support Vector Classifier (SVC)
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Decision Tree Classifier
6. Random Forest Classifier
7. AdaBoost Classifier
8. XGBoost Classifier

The project also includes experimentation with scaled data and XGBoost model tuning.

---

## Train-Test Split

The dataset is divided into training and testing sets using:

* **80% Training Data**
* **20% Testing Data**
* `random_state = 5`

The test set contains **114 samples**.

---

## Final XGBoost Model

After experimenting with different classifiers, an XGBoost model was tuned and evaluated on the test set.

The final XGBoost configuration used in the notebook includes parameters such as:

* `booster = gbtree`
* `colsample_bytree = 0.4`
* `gamma = 0.2`
* `learning_rate = 0.1`
* `max_depth = 15`
* `n_estimators = 100`
* `objective = binary:logistic`
* `random_state = 0`

---

## Model Performance

### Test Set Performance

The final XGBoost model achieved:

**Accuracy: 98.25%**

The test set contained 114 samples.

### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------: | -----: | -------: | ------: |
| 0.0          |      1.00 |   0.96 |     0.98 |      48 |
| 1.0          |      0.97 |   1.00 |     0.99 |      66 |
| **Accuracy** |           |        | **0.98** | **114** |
| Macro Avg    |      0.99 |   0.98 |     0.98 |     114 |
| Weighted Avg |      0.98 |   0.98 |     0.98 |     114 |

In the dataset, the target labels correspond to:

* `0` → Malignant
* `1` → Benign

### Confusion Matrix

```text
[[46,  2],
 [ 0, 66]]
```

This represents the predictions produced by the final XGBoost model on the test set.

---

## Cross-Validation

The final XGBoost model was also evaluated using **10-fold cross-validation**.

### Mean Cross-Validation Accuracy

**96.03%**

The individual cross-validation scores reported in the notebook were:

```text
0.9783
0.9783
0.9783
1.0000
0.9130
0.9111
1.0000
1.0000
0.9556
0.8889
```

Mean accuracy:

```text
0.9603
```

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Pickle

---

## Project Structure

```text
Ml-breast-Cancer-detection-project/
│
├── notebooks/
│   └── breast-cancer-detection.ipynb
│
├── README.md
│
└── .gitignore
```

The main machine learning implementation and analysis are currently contained in the Jupyter Notebook.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aks870/Ml-breast-Cancer-detection-project.git
```

### 2. Navigate to the Project

```bash
cd Ml-breast-Cancer-detection-project
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/breast-cancer-detection.ipynb
```

Run the notebook cells sequentially to reproduce the data analysis, model training, evaluation, and final results.

---

## Model Saving

The notebook also demonstrates saving the trained XGBoost model using Python's `pickle` module:

```python
pickle.dump(xgb_classifier_pt, open('breast_cancer_detector.pickle', 'wb'))
```

The saved model can then be loaded for prediction:

```python
breast_cancer_detector_model = pickle.load(
    open('breast_cancer_detector.pickle', 'rb')
)
```

---

## Key Learning Outcomes

This project demonstrates practical implementation of:

* Exploratory Data Analysis
* Data visualization
* Feature and target preparation
* Train-test splitting
* Feature scaling
* Binary classification
* Multiple machine learning algorithms
* Model evaluation
* Confusion matrix analysis
* Classification reports
* XGBoost
* Hyperparameter tuning
* Cross-validation
* Model serialization using Pickle

---

## Future Improvements

Possible improvements for this project include:

* Building a user-friendly prediction interface
* Adding a dedicated Python application
* Creating a reproducible training pipeline
* Adding automated model comparison
* Adding feature importance visualization
* Adding unit tests
* Adding a requirements file
* Deploying the trained model as a web application

---

## Disclaimer

This project is created for **educational and machine learning practice purposes**.

The predictions generated by this project should **not** be considered medical advice, diagnosis, or a substitute for professional medical evaluation.

---

## Author

**Rajput Ankit Singh**

GitHub: [Aks870](https://github.com/Aks870)
