# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

##add data

# Step 1: Load the Dataset
file_path = 'D:/AI-Driven Construction and Development/New folder/ICT_Subdimension_Dataset new.csv'
data = pd.read_csv(file_path)

# Step 2: Inspect the Dataset
print("Dataset Preview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 3: Preprocessing the Data
# Assuming columns 'Feature1', 'Feature2', ..., 'Target' exist in the dataset
features = ['Household Internet Access (%)', 'e-Government (%)',
            'Public Sector e-procurement (%)']
target = 'Traffic Monitoring (%)'  # Example target column

# Check if the selected features and target exist in the dataset
if any(col not in data.columns for col in features + [target]):
    raise ValueError("One & more specified feature & target columns aren't in the dataset.")

# Drop rows with missing values
data = data.dropna(subset=features + [target])

# Splitting the data into features (X) and target (y)
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Simple AI Model
# Using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 7: Visualize Predictions
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# Step 8: Save the Model (optional)
joblib.dump(model, 'linear_regression_model.pkl')
print("Model saved as 'linear_regression_model.pkl'")
