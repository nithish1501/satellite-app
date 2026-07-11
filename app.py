from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# CSV File Path
df = pd.read_csv(r"C:\Users\anish\Downloads\satellite_error_data.csv")

# Encode satellite_id
le = LabelEncoder()
df["satellite_id"] = le.fit_transform(df["satellite_id"])

# Create Target Column
df["status"] = (df["clock_error_ns"] > 0).astype(int)

# Features
X = df[[
    "satellite_id",
    "hour",
    "hours_since_upload",
    "clock_error_ns",
    "ephemeris_error_m"
]]

# Target
y = df["status"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    satellite = request.form["satellite"]
    hour = int(request.form["hour"])
    upload = int(request.form["upload"])
    clock = float(request.form["clock"])
    ephemeris = float(request.form["ephemeris"])

    satellite = le.transform([satellite])[0]

    prediction = model.predict([[satellite, hour, upload, clock, ephemeris]])

    if prediction[0] == 1:
        result = "⚠ Satellite Error Predicted"
    else:
        result = "✅ Satellite Working Normally"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)