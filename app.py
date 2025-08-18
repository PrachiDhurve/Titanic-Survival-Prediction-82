import streamlit as st 
import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import joblib


st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction App")
st.write("This app predicts if a passenger would survive on the Titanic.")

#Step1: Add input widgets (The user only enters original Titanic features.) ((raw features + optional UI fields))

# Optional (for display/record; NOT used by the model)
passenger_id = st.text_input("Passenger ID (optional)")
name = st.text_input("Name (optional)")

# Original Titanic features
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Silings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# TicketGroup in features; with a single input, assume 1 (unique ticket)
ticket = st.text_input("Ticket (optional; used to estimate group size)", value="SINGLE")

#Step:2 Implementing feature engineering pipeline in app 
def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # FamilySize, IsAlone, FamilyGroup
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilyGroup'] = pd.cut(df['FamilySize'], bins=[0, 1, 4, 11], labels=['Alone', 'Small', 'Large'])

    
    # TicketGroup (single record → treat as 1)
    df['TicketGroup'] = 1

    # FarePerTicket, FarePerFamilyMember
    df['FarePerTicket'] = df['Fare'] / df['TicketGroup']
    df['FarePerFamilyMember'] = df['Fare'] / df['FamilySize'].replace(0, 1)

    # Age/Fare bins (fixed edges, stable for single-row usage)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,50,80], labels = ['Child', 'Teen', 'Adult', 'Senior'], include_lowest=True)
    df['FareBin'] = pd.cut(df['Fare'], bins=[0, 7.9104, 14.4542, 31.0, 512.3292], labels=['Low', 'Medium', 'High', 'Very High'], include_lowest=True)
    df['FarePerPersonBin'] = pd.cut(df['FarePerTicket'], bins=[0, 7.7625, 8.85, 24.2882, 221.7792], labels=['Low', 'Medium', 'High', 'Very High'], include_lowest=True)

    # numeric encoding 
    df['FamilyGroup_num'] = df['FamilyGroup'].map({'Alone': 0, 'Small': 1, 'Large': 2}).fillna(0).astype(int)
    df['AgeBin_num'] = df['AgeBin'].map({'Child': 0, 'Teen': 1, 'Adult': 2, 'Senior': 3}).fillna(2).astype(int)
    df['FareBin_num'] = df['FareBin'].map({'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}).fillna(1).astype(int)
    df['FarePerPersonBin_num'] = df['FarePerPersonBin'].map({'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}).fillna(1).astype(int)

    # Sex-num and Embarked_num (you had both one-hot and numeric)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

    df['Title'] = 0
    df['Title_num'] = 0

    # drop unused
    # Drop unused and original categorical columns
    for col in ['PassengerId','Name','Ticket','Cabin','FamilyGroup','AgeBin','FareBin','FarePerPersonBin','Title']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Ensure all columns are numeric
    df = df.astype(float)

    return df

#Step3: Load and integrate best_model.pkl
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")
model = load_model()

#step4: Display survival probability predictions
if st.button("Predict Srvival"):
    raw = pd.DataFrame([{
        "PassengerId": passenger_id,
        "Name": name,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "SINGLE",
        "Fare": fare,
        "Cabin": "",
        "Embarked": embarked
    }])

    #Feature engineering (match training pipeline)
    X = engineer_features(raw)

    # Align with model’s expected features
    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        X = X.reindex(columns=expected_cols, fill_value=0)

    # Run prediction
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Survival Probability: **{proba:.2%}**")
    st.write(f"Predicted Class: {'Survived' if pred==1 else 'Did Not Survive'}")

    # Add model interpretation/explanation
    st.subheader("Features used for prediction")
    st.dataframe(X)

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        st.bar_chart(importances.sort_values(ascending=False).head(10))

    st.info("Note: PassengerId/Name are UI-only. Model uses engineered features like AgeBin, FareBin, FamilySize, etc.")
