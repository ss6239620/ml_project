import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load and preprocess data
df = pd.read_csv('https://raw.githubusercontent.com/msaricaumbc/DS_data/master/ds602/midterm/employee_departure_dataset.csv', encoding='latin-1')
df['Salary'] = pd.to_numeric(df['Salary'].str[:2])
df['PreviousSalary'] = pd.to_numeric(df['PreviousSalary'].str[:2])
df['SalaryRise'] = df['Salary'] - df['PreviousSalary']

# Define feature columns
Features = ['Gender', 'Distance', 'YearsWorked', 'PreviousSalary', 'Salary', 'SelfReview', 'SupervisorReview', 'DepartmentCode', 'SalaryRise']
X = df[Features]
y = df['Left']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)

# Define pipelines
num_features = ['Gender', 'YearsWorked', 'PreviousSalary', 'Salary', 'SelfReview', 'SupervisorReview', 'DepartmentCode', 'SalaryRise']
cat_features = ['Distance']

num_pipeline = Pipeline([
    ('impute_missing', SimpleImputer(strategy='median')),
    ('standardize_num', StandardScaler())
])

cat_pipeline = Pipeline([
    ('impute_missing_cats', SimpleImputer(strategy='most_frequent')),
    ('create_dummies_cats', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

processing_pipeline = ColumnTransformer(transformers=[
    ('num pipeline', num_pipeline, num_features),
    ('cat pipeline', cat_pipeline, cat_features)
])

# Decision Tree Classifier Pipeline
dtree_pipeline = Pipeline([
    ('data_preprocessing', processing_pipeline),
    ('classifier', DecisionTreeClassifier(criterion='gini', random_state=1))
])
dtree_pipeline.fit(X_train, y_train)

# Linear Regression Pipeline
linreg_pipeline = Pipeline([
    ('data_processing', processing_pipeline),
    ('lm', LinearRegression())
])
linreg_pipeline.fit(X_train, y_train)

# Streamlit app
st.title("Employee Departure Prediction")
st.write("Choose a model and enter the employee's information to predict whether they will stay or leave.")

# Model selection
model_choice = st.selectbox("Choose a model:", ["Decision Tree Classifier", "Linear Regression"])

# Input fields
gender = st.selectbox("Gender:", [0, 1])  # Assuming 0 for Male and 1 for Female
distance = st.number_input("Distance to work (km):", min_value=0)
years_worked = st.number_input("Years Worked:", min_value=0)
previous_salary = st.number_input("Previous Salary:", min_value=0)
salary = st.number_input("Current Salary:", min_value=0)
self_review = st.slider("Self Review (1-5):", 1, 5)
supervisor_review = st.slider("Supervisor Review (1-5):", 1, 5)
department_code = st.selectbox("Department Code:", range(1, 6))
salary_rise = salary - previous_salary

# Prepare input data as a DataFrame
input_data = pd.DataFrame([{
    'Gender': gender,
    'Distance': distance,
    'YearsWorked': years_worked,
    'PreviousSalary': previous_salary,
    'Salary': salary,
    'SelfReview': self_review,
    'SupervisorReview': supervisor_review,
    'DepartmentCode': department_code,
    'SalaryRise': salary_rise
}])

# Prediction button
if st.button("Predict"):
    if model_choice == "Decision Tree Classifier":
        prediction = dtree_pipeline.predict(input_data)[0]
    else:
        linreg_pred = linreg_pipeline.predict(input_data)[0]
        prediction = 1 if linreg_pred > 0.5 else 0
    
    if prediction == 1:
        st.write("Prediction: The employee is likely to leave.")
    else:
        st.write("Prediction: The employee is likely to stay.")
