from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
import os



app = Flask(__name__)

# Train model
X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([20,35,50,65,80])

model = LinearRegression()
model.fit(X,y)

@app.route("/", methods=["GET","POST"])
def home():

    result = None
    hours = None

    if request.method == "POST":

        hours = float(request.form["hours"])

        result = model.predict([[hours]])[0]

        # limit marks
        result = max(0, min(100, result))

        result = round(result, 2)

    return render_template("index.html", result=result, hours=hours)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)