# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Generate Synthetic Data
np.random.seed(0)

# Generate synthetic dates (for demonstration purposes)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

# Generate synthetic sales data
sales = np.sin(np.arange(100) / 10) + np.random.normal(0, 0.2, 100)

# Create a DataFrame with synthetic data
data = pd.DataFrame({'Date': dates, 'Sales': sales})

# Step 3: Preprocess Data
# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Step 4: Split Data and Train the Model
# Define features and target variable
X = data.index.to_julian_date().values.reshape(-1, 1)  # Use Julian date as a feature
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 7: Visualize Predictions
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='True Sales')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Sales', linestyle='--')
plt.title('Future Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
