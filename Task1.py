import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    df = pd.read_csv("C:/Users/Hema M/Downloads/house_price_regression_dataset.csv")
    print(df.columns)
except FileNotFoundError:
    print("Error: File not found.")
    exit()

FEATURE_COLUMNS = ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms']
TARGET_COLUMN = 'Price' 

X = df[FEATURE_COLUMNS] 
y = df[TARGET_COLUMN] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R-squared (R²): {r2:.4f}")

new_house_data = np.array([[2500, 4, 3]])
predicted_price = model.predict(new_house_data)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
