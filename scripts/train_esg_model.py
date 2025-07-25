import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '../data/esg_dataset.csv'
MODEL_PATH = '../models/esg_model.pkl'
PRED_PATH = '../reports/esg_predictions.csv'
IMPLOT_PATH = '../reports/esg_feature_importance.png'


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Drop missing values
    df = df.dropna()

    # Encode target variable
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['esg_score_category'])

    features = ['carbon_emissions', 'board_diversity', 'revenue', 'waste_output']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Evaluation
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    joblib.dump({'model': clf, 'label_encoder': le}, MODEL_PATH)

    # Save predictions
    pred_df = X_test.copy()
    pred_df['true'] = le.inverse_transform(y_test)
    pred_df['predicted'] = le.inverse_transform(y_pred)
    pred_df.to_csv(PRED_PATH, index=False)

    # Feature importance plot
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({'feature': features, 'importance': importances})
    plt.figure(figsize=(8,4))
    sns.barplot(x='importance', y='feature', data=imp_df.sort_values('importance', ascending=False))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(IMPLOT_PATH)
    plt.close()

if __name__ == '__main__':
    main()
