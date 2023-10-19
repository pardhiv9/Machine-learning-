# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create a synthetic car dataset (replace this with your actual data)
data = {
    'Feature1': [3.0, 2.5, 2.7, 3.2, 4.0, 3.5, 2.8, 3.8, 2.5, 3.0],
    'Feature2': [150, 120, 140, 160, 180, 170, 130, 190, 110, 155],
    'Feature3': [2010, 2012, 2011, 2014, 2018, 2016, 2013, 2019, 2010, 2015],
    'Price': [15000, 13000, 14500, 16500, 19500, 18500, 14000, 20000, 12000, 16000]
}

# Create a DataFrame from the synthetic data
df = pd.DataFrame(data)

# Select the features (independent variables) and the target (dependent variable)
X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

# Example of predicting a car price
new_car_features = np.array([[3.2, 155, 2016]])  # Replace with actual feature values
predicted_price = model.predict(new_car_features)
print("Predicted Price:", predicted_price[0])
