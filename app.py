import streamlit as st 
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction App")
st.write("This app predicts if a passenger would survive on the Titanic.")

#Step1: Add input widgets (The user only enters original Titanic features.) ((raw features + optional UI fields))

# Optional (for display/record; NOT used by the model)
passenger_id = st.text_input("Passenger ID (optional)")
name = st.text_input("Name (optional)")

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

    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':4, 'Rev':5,
                 'Col':6, 'Mlle':1, 'Major':6, 'Ms':1, 'Lady':7, 'Sir':7,
                 'Capt':6, 'Countess':7, 'Jonkheer':7, 'Don':7, 'Mme':2, 'Dona':7}
    df['Title_num'] = df['Title'].map(title_map).fillna(0).astype(int)

    # FamilySize, IsAlone, FamilyGroup
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilyGroup'] = pd.cut(df['FamilySize'], bins=[0, 1, 4, 11], labels=['Alone', 'Small', 'Large'])
    df['FamilyGroup_num'] = df['FamilyGroup'].map({'Alone':0,'Small':1,'Large':2}).fillna(0).astype(int)

    # TicketGroup (single record → treat as 1)
    df['TicketGroup'] = 1
    df['FarePerTicket'] = df['Fare'] / df['TicketGroup']
    df['FarePerFamilyMember'] = df['Fare'] / df['FamilySize'].replace(0, 1)

    # AgeBin & Fare bins
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,50,80], labels=['Child','Teen','Adult','Senior'], include_lowest=True)
    df['AgeBin_num'] = df['AgeBin'].map({'Child':0,'Teen':1,'Adult':2,'Senior':3}).fillna(2).astype(int)

    df['FareBin'] = pd.cut(df['Fare'], bins=[0,7.9104,14.4542,31.0,512.3292], labels=['Low','Medium','High','Very High'], include_lowest=True)
    df['FareBin_num'] = df['FareBin'].map({'Low':0,'Medium':1,'High':2,'Very High':3}).fillna(1).astype(int)

    df['FarePerPersonBin'] = pd.cut(df['FarePerTicket'], bins=[0,7.7625,8.85,24.2882,221.7792], labels=['Low','Medium','High','Very High'])
    df['FarePerPersonBin_num'] = df['FarePerPersonBin'].map({'Low':0,'Medium':1,'High':2,'Very High':3}).fillna(1).astype(int)

    # One-hot encode Sex & Embarked
    df = pd.get_dummies(df, columns=['Sex','Embarked'])

    # Drop unused columns
    for col in ['PassengerId','Name','Ticket','Cabin','FamilyGroup','AgeBin','FareBin','FarePerPersonBin','Title']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Ensure all columns are numeric
    df = df.astype(float)

    return df

#Step3: Load and integrate best_model.pkl
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

# Step 4: Threshold Slider
threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.01)

#step5: Predict and Explain
if st.button("Predict & Explain"):
    raw = pd.DataFrame([{
        "PassengerId": passenger_id,
        "Name": name,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
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

    # Prediction
    proba = model.predict_proba(X)[0][1]
    pred_class = 1 if proba >= threshold else 0

    st.subheader("Prediction Result")
    st.write(f"Survival Probability: **{proba:.2%}**")
    st.write(f"Predicted Class: {'Survived' if pred_class==1 else 'Did Not Survive'}")

    # Show features used
    st.subheader("Features used for prediction")
    st.dataframe(X)

    # Top 10 feature importance
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        st.bar_chart(importances.sort_values(ascending=False).head(10))

    st.info("Note: PassengerId/Name are UI-only. Model uses engineered features like AgeBin, FareBin, FamilySize, etc.")