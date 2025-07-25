from flask import Flask, request, jsonify
import sys
sys.path.append('..')
from scripts.predict import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    metrics = data.get('metrics', {})
    text = data.get('text', '')
    result = predict(metrics, text)
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
