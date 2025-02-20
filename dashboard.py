import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\PAVANI\OneDrive\Desktop\IBM Project\dataset\health prescription data.csv')

df = load_data()

# Title
st.title("Health Prescription Classifier Dashboard")

# Display dataset info
st.subheader("Dataset Overview")
st.write(df.head())

# Preprocess Data
X = df['TEXT']
y = df['DIAGNOSIS']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Handle class imbalance
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)

# Filter small classes
class_counts = y.value_counts()
small_classes = class_counts[class_counts < 2].index
y_filtered = y[~y.isin(small_classes)]
X_filtered = X_tfidf[~y.isin(small_classes)]

X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# Cross Validation Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store model performance
all_reports = []
all_confusion_matrices = []

# Train and Evaluate Model
st.subheader("Model Training and Evaluation")

for train_idx, val_idx in cv.split(X_resampled, y_resampled):
    X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
    y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]

    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    report = classification_report(y_val, y_pred, output_dict=True)
    all_reports.append(report)

    cm = confusion_matrix(y_val, y_pred)
    all_confusion_matrices.append(cm)

# Average Metrics
average_report = {
    'precision_macro_avg': np.mean([report['macro avg']['precision'] for report in all_reports]),
    'recall_macro_avg': np.mean([report['macro avg']['recall'] for report in all_reports]),
    'f1-score_macro_avg': np.mean([report['macro avg']['f1-score'] for report in all_reports]),
    'precision_weighted_avg': np.mean([report['weighted avg']['precision'] for report in all_reports]),
    'recall_weighted_avg': np.mean([report['weighted avg']['recall'] for report in all_reports]),
    'f1-score_weighted_avg': np.mean([report['weighted avg']['f1-score'] for report in all_reports]),
}

st.write("### Average Classification Report")
st.dataframe(pd.DataFrame(average_report, index=[0]))

# Average Confusion Matrix
average_cm = np.mean(all_confusion_matrices, axis=0)

st.write("### Average Confusion Matrix")
plt.figure(figsize=(12, 8))
sns.heatmap(average_cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(plt)

# Prediction on New Data
st.subheader("Make Predictions")
input_text = st.text_area("Enter Medical Text for Diagnosis Prediction:")
if st.button("Predict"):
    if input_text:
        input_tfidf = vectorizer.transform([input_text])
        prediction = model.predict(input_tfidf)
        st.success(f"Predicted Diagnosis: {prediction[0]}")
    else:
        st.error("Please enter valid medical text.")