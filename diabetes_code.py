
# Step 1: Import Libraries and Load Data
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression().fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Fix the plotting issue by selecting one feature for X_test (e.g., "bmi")
X_test_bmi = X_test['bmi'].reset_index(drop=True)
# Ensure X_test_bmi and y_test have the same indices for proper alignment
y_test = pd.Series(y_test).reset_index(drop=True)

# Plot the data and regression line
plt.scatter(X_test_bmi, y_test, color='blue', label='Test Data')
plt.plot(X_test_bmi, y_pred, color='red', label='Regression Line')
plt.xlabel('BMI')
plt.ylabel('Target')
plt.legend()
plt.title('Linear Regression Example')
st.pyplot(plt)


# Step 2: Build Streamlit App Interface
st.set_page_config(page_title='Diabetes Progression Predictor')
st.title('Diabetes Progression Predictor')
st.write('Adjust the features below and the AI model will predict diabetes progression score.')

# User inputs
input_data = {}
for feature in X.columns:
    input_data[feature] = st.slider(f'{feature}', float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

# Prediction
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]
st.subheader(f'Predicted Diabetes Progression Score: {prediction:.2f}')
