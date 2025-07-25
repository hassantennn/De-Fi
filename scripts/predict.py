import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

ESG_MODEL_PATH = '../models/esg_model.pkl'
GREEN_MODEL_DIR = '../models/greenwashing_detector'


def load_models():
    esg_bundle = joblib.load(ESG_MODEL_PATH)
    esg_model = esg_bundle['model']
    esg_le = esg_bundle['label_encoder']

    tokenizer = AutoTokenizer.from_pretrained(GREEN_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(GREEN_MODEL_DIR)
    green_pipe = pipeline('text-classification', model=model, tokenizer=tokenizer)

    return esg_model, esg_le, green_pipe


def predict(esg_metrics: dict, text: str):
    esg_model, esg_le, green_pipe = load_models()

    features = ['carbon_emissions', 'board_diversity', 'revenue', 'waste_output']
    df = pd.DataFrame([esg_metrics])[features]
    esg_pred_label = esg_model.predict(df)[0]
    esg_category = esg_le.inverse_transform([esg_pred_label])[0]

    green_result = green_pipe(text)[0]
    green_flag = green_result['label']
    score = green_result['score']

    return {
        'esg_category': esg_category,
        'greenwashing_label': green_flag,
        'greenwashing_score': score,
    }

if __name__ == '__main__':
    sample_metrics = {
        'carbon_emissions': 300,
        'board_diversity': 0.35,
        'revenue': 60,
        'waste_output': 70,
    }
    sample_text = "We donate a portion of profits to local environmental groups every year."
    print(predict(sample_metrics, sample_text))
