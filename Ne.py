#%% md
# Diabetes Risk Screening Tool

#%%
from ucimlrepo import fetch_ucirepo

# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# metadata
print(cdc_diabetes_health_indicators.metadata)

# variable information
print(cdc_diabetes_health_indicators.variables)


#%%
print(X.info())
#%%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if hasattr(y_train, "values"):
  y_train = y_train.values.ravel()
if hasattr(y_test, "values"):
  y_test = y_test.values.ravel()

model_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(random_state=42),
        ),
    ]
)

model_pipeline.fit(X_train, y_train)


#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# التنبؤ على بيانات الاختبار
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1] # احتمال الإصابة

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
#%%
import joblib

joblib.dump(model_pipeline, "diabetes_model.pkl")
print("Done!")
#%%
import joblib
import pandas as pd
import streamlit as st

# 1. تحميل الموديل المحفوظ
model = joblib.load("diabetes_model.pkl")

# 2. عنوان التطبيق وتنبيه إخلاء المسؤولية الطبية (مطلوب في الشيت)
st.title("🩺 Diabetes Risk Screening Tool")
st.warning(
    "⚠️ Note: This tool is a screening aid only and is NOT a medical"
    " diagnosis."
)

# 3. واجهة إدخال البيانات للمستخدم (بأهم الأسئلة في الداتا)
st.header("Enter Your Health Indicators:")

high_bp = st.selectbox(
    "High Blood Pressure?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
high_chol = st.selectbox(
    "High Cholesterol?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
chol_check = st.selectbox(
    "Cholesterol Check in past 5 years?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0)
smoker = st.selectbox(
    "Smoked at least 100 cigarettes in life?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
stroke = st.selectbox(
    "Ever had a stroke?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
heart_disease = st.selectbox(
    "Coronary heart disease or myocardial infarction?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
phys_activity = st.selectbox(
    "Physical activity in past 30 days?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
fruits = st.selectbox(
    "Consume Fruit 1 or more times per day?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
veggies = st.selectbox(
    "Consume Vegetables 1 or more times per day?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
hvy_alcohol = st.selectbox(
    "Heavy alcohol consumption?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
any_healthcare = st.selectbox(
    "Have any kind of health care coverage?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
no_doc_cost = st.selectbox(
    "Could not see a doctor because of cost?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
gen_hlth = st.slider("General Health (1 = Excellent, 5 = Poor)", 1, 5, 3)
ment_hlth = st.slider("Days of poor mental health in past 30 days", 0, 30, 0)
phys_hlth = st.slider("Days of poor physical health in past 30 days", 0, 30, 0)
diff_walk = st.selectbox(
    "Difficulty walking or climbing stairs?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
sex = st.selectbox(
    "Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female"
)
age = st.slider("Age Category (1 = 18-24 ... 13 = 80+)", 1, 13, 5)
education = st.slider("Education Level (1 to 6)", 1, 6, 4)
income = st.slider("Income Scale (1 to 8)", 1, 8, 5)

# 4. تجميع المدخلات بنفس ترتيب أعمدة التدريب X بالضبط
input_data = pd.DataFrame(
    [[
        high_bp,
        high_chol,
        chol_check,
        bmi,
        smoker,
        stroke,
        heart_disease,
        phys_activity,
        fruits,
        veggies,
        hvy_alcohol,
        any_healthcare,
        no_doc_cost,
        gen_hlth,
        ment_hlth,
        phys_hlth,
        diff_walk,
        sex,
        age,
        education,
        income,
    ]],
    columns=[
        "HighBP",
        "HighChol",
        "CholCheck",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income",
    ],
)

# 5. زر التنبؤ وعرض النتيجة
if st.button("Check Risk"):
  prediction = model.predict(input_data)[0]
  probability = model.predict_proba(input_data)[0][1]

  if prediction == 1:
    st.error(
        f"High Risk of Diabetes/Prediabetes (Probability: {probability:.1%})"
    )
  else:
    st.success(f"Low Risk (Probability: {probability:.1%})")
#%%
