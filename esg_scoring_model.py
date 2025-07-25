import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("sample_esg_data.csv")  # Replace with your actual file name

# Preview
print("Columns:", df.columns)
print(df.head())

# Clean data
df.dropna(inplace=True)

# Define features and label
features = ['carbon_emissions', 'board_diversity', 'revenue', 'waste_output']
label = 'esg_score_category'

X = df[features]
y = LabelEncoder().fit_transform(df[label])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "esg_scoring_model.pkl")
