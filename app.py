import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Load and preprocess data, train models (cached)
@st.cache(allow_output_mutation=True)
def train_models():
    df = pd.read_csv('data.csv')

    df.drop('Loan_ID', axis=1, inplace=True)

    target = 'Loan_Status'

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target]

    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols + [target]] = imputer_cat.fit_transform(df[cat_cols + [target]])

    label_encoders = {}
    for col in cat_cols + [target]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df['Income_Credit_History'] = df['ApplicantIncome'] * df['Credit_History']

    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    to_scale = num_cols + ['Income_Credit_History']
    scaler = StandardScaler()
    X_train[to_scale] = scaler.fit_transform(X_train[to_scale])
    X_test[to_scale] = scaler.transform(X_test[to_scale])

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    for m in models.values():
        m.fit(X_train, y_train)

    return models, scaler, label_encoders, to_scale, X_train.columns.tolist()

st.sidebar.header('Applicant Information')

def user_input_features(label_encoders):

    # Categorical inputs
    gender = st.sidebar.selectbox('Gender', label_encoders['Gender'].classes_)
    married = st.sidebar.selectbox('Married', label_encoders['Married'].classes_)
    dependents = st.sidebar.selectbox('Dependents', label_encoders['Dependents'].classes_)
    education = st.sidebar.selectbox('Education', label_encoders['Education'].classes_)
    self_emp = st.sidebar.selectbox('Self Employed', label_encoders['Self_Employed'].classes_)
    prop_area = st.sidebar.selectbox('Property Area', label_encoders['Property_Area'].classes_)
    credit_hist = st.sidebar.selectbox('Credit History', [0.0, 1.0])

    # Numerical inputs
    app_income = st.sidebar.number_input('Applicant Income', min_value=0)
    coapp_income = st.sidebar.number_input('Coapplicant Income', min_value=0)
    loan_amt = st.sidebar.number_input('Loan Amount', min_value=0)
    loan_term = st.sidebar.number_input('Loan Amount Term (days)', min_value=0)

    data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_emp,
        'Property_Area': prop_area,
        'Credit_History': credit_hist,
        'ApplicantIncome': app_income,
        'CoapplicantIncome': coapp_income,
        'LoanAmount': loan_amt,
        'Loan_Amount_Term': loan_term
    }
    return pd.DataFrame([data])

st.title('Loan Approval Prediction')
models, scaler, label_encoders, to_scale, feature_cols = train_models()
input_df = user_input_features(label_encoders)

for col, le in label_encoders.items():
    if col in input_df:
        input_df[col] = le.transform(input_df[col])

input_df['Income_Credit_History'] = (
    input_df['ApplicantIncome'] * input_df['Credit_History']
)

input_df[to_scale] = scaler.transform(input_df[to_scale])

input_df = input_df[feature_cols]

model_name = st.sidebar.selectbox('Select Model', list(models.keys()))
model = models[model_name]

prediction = model.predict(input_df)[0]
pred_proba = model.predict_proba(input_df)[0, 1]

st.subheader('Prediction')
status_map = {0: 'Not Approved', 1: 'Approved'}
st.write(status_map[prediction])

st.subheader('Prediction Probability of Approval')
st.write(f"Approved: {pred_proba:.2f}, Not Approved: {1-pred_proba:.2f}")


