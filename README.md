# 🏦 Loan Approval Prediction App

This interactive web application predicts loan approval status based on applicant information. It uses multiple machine learning models trained on historical loan application data. Built with **Streamlit**, it allows users to input data and see predictions from models like Decision Trees, SVM, Naive Bayes, and more.

## 🚀 Features
- User-friendly sidebar form to input applicant details  
- Support for multiple machine learning models (Decision Tree, SVM, Random Forest, etc.)  
- Real-time prediction and probability score for loan approval  
- Feature engineering with derived attributes (e.g., `Income_Credit_History`)  
- Handles missing data, label encoding, and feature scaling automatically  

## 🛠️ Tech Stack
- **Frontend/UI**: Streamlit  
- **Backend/ML**: scikit-learn  
- **Data Handling**: pandas, numpy  
- **Models**: Decision Tree, Naive Bayes, SVM, Random Forest, Gradient Boosting  

## 🧠 Model Training Details
- Missing values imputed using median (for numeric) and most frequent (for categorical)  
- Label encoding applied to all categorical features  
- StandardScaler applied to numeric features  
- Feature `Income_Credit_History` created as a product of ApplicantIncome and Credit_History  
- 80/20 train-test split with stratified sampling  
- Models trained and cached for fast reload  

## 📦 Setup Instructions
1. Place your dataset as `data.csv` in the root directory.  
2. Install dependencies:  
```bash
pip install -r requirements.txt
```
3. Run teh app:
```bash
streamlit run app.py
```
