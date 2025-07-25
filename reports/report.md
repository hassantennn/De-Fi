# GreenChain ESG & Greenwashing Models Report

## ESG Scoring Model

- **Algorithm:** RandomForestClassifier
- **Features:** carbon_emissions, board_diversity, revenue, waste_output
- **Example accuracy:** ~90% on the sample dataset (placeholder)
- **Feature Importances:** saved to `esg_feature_importance.png`

## Greenwashing Detection Model

- **Model:** bert-base-uncased fine-tuned on labelled sentences
- **Example metrics:** F1 score ~0.85 (placeholder)
- **Model directory:** `models/greenwashing_detector`

## Integration

The `predict` function in `scripts/predict.py` loads both models to return an ESG
category and greenwashing label for provided metrics and text. A Flask API in
`api/app.py` exposes `/predict` and `/health` endpoints for easy integration with
the GreenChain system.
