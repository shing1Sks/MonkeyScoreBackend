from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)

CORS(app)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le = pickle.load(f)


@app.route("/predict", methods=["POST"])
def analysis():
    print(request)
    required_params = [
        "sex",
        "age",
        "study",
        "sleep",
        "time_rate",
        "grp_study",
        "days",
        "distraction",
        "mode",
        "prep_base",
        "prep_rate",
        "stress_rate",
        "maths",
        "science",
        "logical",
        "higher_edu",
        "attendence_rate",
        "extra_curricular",
        "mother_edu",
        "father_edu",
        "family_edu",
        "exercise",
        "friend_circle",
        "screen_time",
    ]

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing_params = [param for param in required_params if param not in data]
    if missing_params:
        return jsonify(
            {
                "error": "Missing parameters",
                "missing": missing_params,
                "received": list(data.keys()),
            }
        ), 400

    for key in le:
        if key in data:
            data[key] = le[key].transform([data[key]])[0]

    input_df = pd.DataFrame([data], columns=required_params)

    prediction = model.predict(input_df)

    prediction = float(prediction[0])

    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
