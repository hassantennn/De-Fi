import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, "sample_esg_data.csv"))

# Preview
print("Columns:", df.columns)
print(df.head())

# Clean data
df.dropna(inplace=True)

# Define features and label
features = ['carbon_emissions', 'board_diversity', 'revenue', 'waste_output']
label = 'esg_score_category'

X = df[features]
le = LabelEncoder()
y = le.fit_transform(df[label])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model next to other models
model_path = os.path.join(BASE_DIR, "models", "esg_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump({'model': model, 'label_encoder': le}, model_path)
