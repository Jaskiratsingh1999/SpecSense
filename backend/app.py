from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from phone_examples import example_phones
import pandas as pd  # Make sure this is at the top!
import os  # Required for Render port support


app = Flask(__name__)
CORS(app)

# Load trained ML parts
scaler = joblib.load("model/scaler.pkl")
base_models = joblib.load("model/base_models.pkl")
stacked_model = joblib.load("model/stacked_model.pkl")

FEATURE_NAMES = ["battery_power", "px_height", "px_width", "ram"]

price_map = {
    0: "Low Budget 💸",
    1: "Medium Tier 📱",
    2: "High-End 🔥",
    3: "Premium / Flagship 👑"
}


def find_upgrade_suggestion(original_features, current_tier):
    suggestion = None
    upgrades = [
        ("ram", 500, 8000),
        ("battery_power", 200, 6000),
        ("px_height", 200, 2400),
        ("px_width", 100, 1440)
    ]

    for feature, step, max_val in upgrades:
        temp = original_features.copy()
        while temp[feature] < max_val:
            temp[feature] += step
            input_df = pd.DataFrame([temp], columns=FEATURE_NAMES)
            scaled_input = scaler.transform(input_df)

            meta_features = np.column_stack([model.predict(scaled_input) for model in base_models])
            prediction = stacked_model.predict(meta_features)
            new_tier = round(float(prediction[0]))

            if new_tier > current_tier:
                suggestion = f"Try increasing {feature.replace('_', ' ')} to {temp[feature]} to move up to {price_map[new_tier]}"
                return suggestion
    return None


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json
        print("🛠️ Received input:", input_data)

        features = [input_data.get(feat, 0) for feat in FEATURE_NAMES]
        print("🔍 Final features:", features)

      

        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        scaled_input = scaler.transform(input_df)


        meta_features = np.column_stack([model.predict(scaled_input) for model in base_models])
        final_prediction = stacked_model.predict(meta_features)

        predicted_index = round(float(final_prediction[0]))
        human_label = price_map.get(predicted_index, "Unknown")

        example_list = example_phones.get(predicted_index, [])
        example = example_list[0] if example_list else {}

        original_dict = dict(zip(FEATURE_NAMES, features))
        suggestion = find_upgrade_suggestion(original_dict, predicted_index)

        return jsonify({
            "predicted_price_range": human_label,
            "example_phone": example,
            "upgrade_tip": suggestion,
            "market_examples": example_list[:3]  # Send top 3 examples (you can add more in phone_examples.py)
})


    except Exception as e:
        print("❌ Error occurred:", str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
