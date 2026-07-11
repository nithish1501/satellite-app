from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = "secret123"

# Login details
USERNAME = "admin"
PASSWORD = "1234"

# CSV File Path
df = pd.read_csv(r"C:\Users\anish\data\satellite_error_data.csv")

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


# Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))

        return render_template(
            "login.html",
            error="Wrong username or password"
        )

    return render_template("login.html")


# Home Page
@app.route("/")
def home():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    satellite = request.form["satellite"]
    hour = int(request.form["hour"])
    upload = int(request.form["upload"])
    clock = float(request.form["clock"])
    ephemeris = float(request.form["ephemeris"])

    satellite = le.transform([satellite])[0]

    prediction = model.predict(
        [[satellite, hour, upload, clock, ephemeris]]
    )

    if prediction[0] == 1:
        result = "⚠ Satellite Error Predicted"
    else:
        result = "✅ Satellite Working Normally"

    return render_template(
        "index.html",
        prediction=result
    )


# Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)