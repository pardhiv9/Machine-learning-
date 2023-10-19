# Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Generate Synthetic Data
np.random.seed(0)
num_samples = 1000

feature1 = np.random.rand(num_samples) * 100
feature2 = np.random.rand(num_samples) * 200
feature3 = np.random.rand(num_samples) * 300

# Generate the target variable (house prices)
target = 500 * feature1 + 300 * feature2 + 100 * feature3 + np.random.normal(0, 100, num_samples)

# Step 3: Split the Data into Training and Testing Sets
X = np.column_stack((feature1, feature2, feature3))
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# Step 4: Create and Train the Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 7: Use the Model for Predictions
# Define values for new data
feature1_value = 70
feature2_value = 150
feature3_value = 250

# Use the model to predict the price for the new data
new_data = np.array([[feature1_value, feature2_value, feature3_value]])
predicted_price = model.predict(new_data)
print(f'Predicted Price: {predicted_price[0]}')
