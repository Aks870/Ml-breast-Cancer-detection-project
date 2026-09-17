import os
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# Load breast cancer dataset
cancer_dataset = load_breast_cancer()

X = cancer_dataset.data
y = cancer_dataset.target


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=5
)


# Final XGBoost model
xgb_classifier = XGBClassifier(
    base_score=0.5,
    booster="gbtree",
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=0.4,
    gamma=0.2,
    learning_rate=0.1,
    max_delta_step=0,
    max_depth=15,
    min_child_weight=1,
    n_estimators=100,
    n_jobs=1,
    objective="binary:logistic",
    random_state=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    seed=None,
    subsample=1,
    verbosity=1
)


# Train model
xgb_classifier.fit(X_train, y_train)


# Evaluate model
y_pred = xgb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")


# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)


# Save trained model
model_path = "models/breast_cancer_model.pkl"
joblib.dump(xgb_classifier, model_path)

print(f"Model saved successfully: {model_path}")