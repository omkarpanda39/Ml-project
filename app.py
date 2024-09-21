import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load dataset
data1 = pd.read_csv('insurance.csv')
data1 = data1[data1['charges'] <= 15000]

# Model 1: Using only age to predict charges
X1 = data1[['age']]
y = data1['charges']

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X1_train, y_train)

y_pred1 = model1.predict(X1_test)

mse1 = mean_squared_error(y_test, y_pred1)
r2_1 = r2_score(y_test, y_pred1)

st.title("Charges Prediction App")

st.write(f"Model 1 (Age only) - Mean Squared Error: {mse1:.2f}")
st.write(f"Model 1 (Age only) - R^2 Score: {r2_1:.2f}")

# Display test set and predicted values for Model 1
results_model1 = pd.DataFrame({'Age': X1_test['age'], 'Actual Charges': y_test, 'Predicted Charges': y_pred1})
st.write("Model 1 - Testing Data (Age and Charges):")
st.dataframe(results_model1)

# Model 2: Using age and BMI to predict charges
X2 = data1[['age', 'bmi']]
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X2_train, y_train)

y_pred2 = model2.predict(X2_test)

mse2 = mean_squared_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)

st.write(f"\nModel 2 (Age and BMI) - Mean Squared Error: {mse2:.2f}")
st.write(f"Model 2 (Age and BMI) - R^2 Score: {r2_2:.2f}")

# Display test set and predicted values for Model 2
results_model2 = pd.DataFrame({'Age': X2_test['age'], 'BMI': X2_test['bmi'], 'Actual Charges': y_test, 'Predicted Charges': y_pred2})
st.write("Model 2 - Testing Data (Age, BMI, and Charges):")
st.dataframe(results_model2)

# Plotting Model 1: Age vs Charges
plt.figure(figsize=(10, 5))
plt.scatter(X1_test, y_test, color='blue', label='Actual data (Model 1)')
plt.scatter(X1_test, y_pred1, color='red', label='Predicted data (Model 1)')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Model 1: Age vs Charges')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Plotting Model 2: Age and BMI vs Charges (only Age is shown for plotting)
plt.figure(figsize=(10, 5))
plt.scatter(X2_test['age'], y_test, color='blue', label='Actual data (Model 2)')
plt.scatter(X2_test['age'], y_pred2, color='green', label='Predicted data (Model 2)', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Model 2: Age, BMI vs Charges')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Function to make predictions based on user input
def predict_charges(age_input, bmi_input=None):
    age_data = pd.DataFrame({'age': [age_input]})
    predicted_charges_model1 = model1.predict(age_data)[0]
    
    st.write(f"Predicted Charges (Model 1 - Age only): ${predicted_charges_model1:.2f}")
    
    if bmi_input is not None:
        age_bmi_data = pd.DataFrame({'age': [age_input], 'bmi': [bmi_input]})
        predicted_charges_model2 = model2.predict(age_bmi_data)[0]
        
        st.write(f"Predicted Charges (Model 2 - Age and BMI): ${predicted_charges_model2:.2f}")
    else:
        st.warning("For Model 2, please provide both age and BMI.")

# User input for predictions
age_input = st.number_input("Enter age:", min_value=0, max_value=120, value=25)
bmi_input = st.number_input("Enter BMI (optional for Model 1):", min_value=0.0, max_value=50.0, value=22.0)

if st.button("Predict Charges"):
    predict_charges(age_input, bmi_input)


