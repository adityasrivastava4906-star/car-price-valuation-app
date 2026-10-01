from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

model = joblib.load("car_price_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
top_brands = joblib.load("top_brands.pkl")

CURRENT_YEAR = 2026

BRAND_LABELS = {
    "bajaj": "Bajaj",
    "city": "Honda City",
    "corolla": "Toyota Corolla",
    "royal": "Royal Enfield",
    "honda": "Honda",
    "verna": "Hyundai Verna",
    "fortuner": "Toyota Fortuner",
    "hero": "Hero",
    "brio": "Honda Brio",
    "innova": "Toyota Innova",
    "other": "Other",
}

df = pd.read_csv("car_data_augmented.csv")
TIER_Q1 = float(df["Selling_Price"].quantile(0.33))
TIER_Q2 = float(df["Selling_Price"].quantile(0.66))
YEAR_MIN = int(df["Year"].min())
YEAR_MAX = int(df["Year"].max())

reg_metrics = pd.read_csv("regression_results_real.csv", index_col=0)
class_metrics = pd.read_csv("classification_results_real.csv", index_col=0)


def classify_tier(price_lakhs):
    if price_lakhs < TIER_Q1:
        return "Budget"
    if price_lakhs < TIER_Q2:
        return "Mid-Range"
    return "Premium"


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def api_config():
    brands = [
        {"value": b, "label": BRAND_LABELS.get(b, b.title())}
        for b in top_brands
    ]
    if "other" not in [b["value"] for b in brands]:
        brands.append({"value": "other", "label": "Other"})

    years = list(range(YEAR_MAX, YEAR_MIN - 1, -1))

    best_reg = reg_metrics.sort_values("Test_R2", ascending=False).iloc[0]
    best_reg_name = reg_metrics.sort_values("Test_R2", ascending=False).index[0]
    best_class = class_metrics.sort_values("Macro F1", ascending=False).iloc[0]
    best_class_name = class_metrics.sort_values("Macro F1", ascending=False).index[0]

    return jsonify({
        "currentYear": CURRENT_YEAR,
        "brands": brands,
        "years": years,
        "fuelTypes": ["Petrol", "Diesel"],
        "transmissions": ["Manual", "Automatic"],
        "sellerTypes": ["Dealer", "Individual"],
        "tierCutoffs": {"budgetMax": round(TIER_Q1, 2), "midMax": round(TIER_Q2, 2)},
        "metrics": {
            "bestRegressor": best_reg_name,
            "testR2": round(float(best_reg["Test_R2"]), 4),
            "cvR2": round(float(best_reg["CV_R2_mean"]), 4),
            "testMae": round(float(best_reg["Test_MAE"]), 3),
            "bestClassifier": best_class_name,
            "classAccuracy": round(float(best_class["Accuracy"]), 4),
            "macroF1": round(float(best_class["Macro F1"]), 4),
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        row = {col: 0 for col in feature_cols}
        row["car_age"] = CURRENT_YEAR - int(data["year"])
        row["Present_Price"] = float(data["presentPrice"])
        row["Kms_Driven"] = int(data["kmsDriven"])
        row["Owner"] = 0

        brand = data["brand"].lower()
        if brand not in top_brands:
            brand = "other"
        brand_col = f"brand_{brand}"
        if brand_col in row:
            row[brand_col] = 1

        fuel_col = f"Fuel_Type_{data['fuel']}"
        if fuel_col in row:
            row[fuel_col] = 1

        seller_col = f"Seller_Type_{data['seller']}"
        if seller_col in row:
            row[seller_col] = 1

        trans_col = f"Transmission_{data['transmission']}"
        if trans_col in row:
            row[trans_col] = 1

        X = pd.DataFrame([row])[feature_cols]
        prediction = float(model.predict(X)[0])
        prediction = max(0.1, prediction)

        present = float(data["presentPrice"])
        age = max(1, CURRENT_YEAR - int(data["year"]))
        retention = round((prediction / present) * 100, 1) if present > 0 else 0
        deprec = round(((1 - (prediction / present)) / age) * 100, 1) if present > 0 else 0

        return jsonify({
            "predicted_price_lakhs": round(prediction, 2),
            "tier": classify_tier(prediction),
            "retention_pct": retention,
            "depreciation_pct_per_year": deprec,
            "car_age": age,
        })

    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
