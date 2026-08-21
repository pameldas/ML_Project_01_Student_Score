import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Create the path to our CSV file
file_path = project_folder / "data" / "student_scores.csv"

# Read the CSV file
data = pd.read_csv(file_path)

# Display the dataset
print(data)
print("\nFIRST FIVE ROWS:")
print(data.head())

print("\nDATASET SIZE:")
print(data.shape)

print("\nCOLUMN NAMES:")
print(data.columns)

print("\nDATA TYPES:")
print(data.dtypes)

print("\nSTATISTICAL SUMMARY:")
print(data.describe())

import matplotlib.pyplot as plt

plt.scatter(data["Hours"], data["Score"])

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plot_path = project_folder / "results" / "regression_plot.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# Separate feature and target

X = data[["Hours"]]
y = data["Score"]

print("\nFEATURE X:")
print(X)

print("\nTARGET y:")
print(y)

print("\nX shape:", X.shape)
print("y shape:", y.shape)
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING FEATURES:")
print(X_train)

print("\nTEST FEATURES:")
print(X_test)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Display what the model learned
print("\nMODEL COEFFICIENT:")
print(model.coef_)

print("\nMODEL INTERCEPT:")
print(model.intercept_)
# Make predictions using unseen test data
y_pred = model.predict(X_test)

print("\nACTUAL TEST SCORES:")
print(y_test.values)

print("\nPREDICTED TEST SCORES:")
print(y_pred)
# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
# Plot the data and regression line
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print("\nTRAIN R²:", train_r2)
print("TEST R²:", test_r2)
plt.scatter(X, y)

plt.plot(
    X,
    model.predict(X)
)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Hours Studied vs Exam Score")

Add Linear Regression implementation
