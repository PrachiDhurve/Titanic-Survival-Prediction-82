# 🚢 Titanic Survival Prediction

This project predicts whether a passenger survived the Titanic disaster using machine learning.  
It includes a **Jupyter notebook** for data exploration/modeling and a deployed **Streamlit web app** for real-time predictions.

---

## 📂 Project Files

- **`app.py`** → Streamlit web application  
- **`best_model.pkl`** → Trained ML model saved for deployment  
- **`requirements.txt`** → Dependencies needed to run the project  
- **`Titanic_Survival_Predict.ipynb`** → Jupyter notebook with data analysis, feature engineering, and model development  

---

## 📊 Problem Statement
The goal is to build a model that predicts passenger survival on the Titanic using demographic and ticket data.  
This is a classic **binary classification problem**.

---

## 📑 Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  
- Rows: 891 passengers  
- Key features:  
  - Pclass (ticket class)  
  - Sex  
  - Age  
  - SibSp (siblings/spouses)  
  - Parch (parents/children)  
  - Fare  
  - Embarked (port of embarkation)  
- Target: **Survived** (0 = No, 1 = Yes)  

---

## 🔧 Feature Engineering
- Filled missing Age & Embarked values  
- Extracted `Title` from Name  
- Created new features: `FamilySize`, `IsAlone`, . `FamilyGroup`, `FarePerFamilyMember`, `FarePerTicket`, `TicketGroup`
- Binned Age, Fare and FarePerPerson into categories  
- Encoded categorical variables  

---

## 🤖 Model Development
- Models trained: Logistic Regression, Random Forest, Gradient Boosting, XGBoost/LightGBM  
- Evaluation metrics: Accuracy, ROC-AUC, Confusion Matrix  
- Best model saved as **`best_model.pkl`**  

---

## 🌐 Streamlit Web App
The web app allows users to enter passenger details and get survival predictions.

### Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py

Live demo:
# add streamlit app link

Project Workflow:

Data Exploration & Visualization in Jupyter notebook
Feature Engineering and preprocessing
Model Training & Evaluation with multiple algorithms
Best Model Export as best_model.pkl
Web Deployment with Streamlit