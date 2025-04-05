from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib


df = pd.read_csv("mutual_funds_india.csv")

# Load saved files
vectorizer = joblib.load('vectorizer.pkl')
feature_matrix = joblib.load('feature_matrix.pkl')
similarity_scores = joblib.load('similarity_scores.pkl')

# Recommendation function
def recommend_funds(fund_name, top_n=5):
    fund_name = fund_name.strip()
    if fund_name not in df["Mutual Fund Name"].values:
        return None

    idx = df[df["Mutual Fund Name"] == fund_name].index[0]
    similar_indices = similarity_scores[idx].argsort()[-(top_n+1):-1][::-1]

    recommendations = df.iloc[similar_indices][["Mutual Fund Name", "category", "risk_type"]]
    return recommendations.to_dict(orient="records")

app = Flask(__name__)
CORS(app)



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fund_name = data.get('fund_name')
    top_n = int(data.get('top_n', 5))

    if not fund_name:
        return jsonify({"error": "Please provide a fund_name parameter."}), 400

    results = recommend_funds(fund_name, top_n)
    if results is None:
        return jsonify({"error": "Fund not found. Please try another name."}), 404

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
