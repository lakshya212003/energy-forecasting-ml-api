from flask import Flask, request, render_template_string
import joblib
import numpy as np
import math

app = Flask(__name__)
model = joblib.load("energy_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Energy Forecasting</title>
    <style>
        body {
            font-family: Arial;
            background-color: #f4f6f9;
            text-align: center;
            padding: 40px;
        }
        .card {
            background: white;
            padding: 30px;
            border-radius: 10px;
            width: 400px;
            margin: auto;
            box-shadow: 0px 5px 20px rgba(0,0,0,0.1);
        }
        input {
            width: 90%;
            padding: 8px;
            margin: 8px 0;
        }
        button {
            background: #007BFF;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        h1 {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>⚡ Energy Forecast</h1>
        <form method="post">
            <input name="lag_1" placeholder="Previous Hour Energy" required><br>
            <input name="lag_24" placeholder="Yesterday Same Hour" required><br>
            <input name="lag_48" placeholder="2 Days Ago Same Hour" required><br>
            <input name="rolling_mean_24" placeholder="Last 24h Average" required><br>
            <input name="rolling_std_24" placeholder="Last 24h Std Dev" required><br>
            <input name="hour" placeholder="Hour (0-23)" required><br>
            <input name="day_of_week" placeholder="Day of Week (0=Mon)" required><br>
            <button type="submit">Predict</button>
        </form>
        {% if prediction %}
            <h2>Predicted Energy: {{ prediction }}</h2>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        lag_1 = float(request.form["lag_1"])
        lag_24 = float(request.form["lag_24"])
        lag_48 = float(request.form["lag_48"])
        rolling_mean_24 = float(request.form["rolling_mean_24"])
        rolling_std_24 = float(request.form["rolling_std_24"])
        hour = int(request.form["hour"])
        day_of_week = int(request.form["day_of_week"])

        sin_hour = math.sin(2 * math.pi * hour / 24)
        cos_hour = math.cos(2 * math.pi * hour / 24)

        features = np.array([[lag_1, lag_24, lag_48,
                              rolling_mean_24, rolling_std_24,
                              sin_hour, cos_hour, day_of_week]])

        prediction = round(float(model.predict(features)[0]), 4)

    return render_template_string(HTML, prediction=prediction)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)