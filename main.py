import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([20, 35, 50, 65, 80,95,95,95])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "student_model.pkl")
hours = float(input("Enter study hours: "))

prediction = model.predict([[hours]])

marks = prediction[0]

# Apply limits
if marks > 100:
    marks = 99

if marks < 0:
    marks = 0

print("Predicted marks:", marks)