# app.py
# Streamlit app version of your Email Phishing Detection code
# Logic preserved from original Colab code; UI switched to Streamlit.

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, Any, Tuple
import csv
from io import StringIO
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

class EmailPhishingDetector:
    """
    Email Phishing Detection System with robust dataset loading
    """
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.training_stats = {}
        self.dataset_info = {}

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset with robust error handling"""
        try:
            st.info(f"Loading dataset from: {file_path}")
            try:
                df = pd.read_csv(file_path)
                st.success("Dataset loaded successfully with standard method!")
                return df
            except pd.errors.ParserError as e:
                st.warning(f"CSV parsing error: {str(e)}")
                st.info("Trying robust parsing...")
                try:
                    df = pd.read_csv(
                        file_path,
                        quoting=3,  # QUOTE_NONE
                        on_bad_lines='skip',
                        engine='python',
                        encoding='utf-8',
                        low_memory=False
                    )
                    st.success("Dataset loaded with robust parsing!")
                    return df
                except Exception as e2:
                    st.warning(f"Robust parsing failed: {str(e2)}")
                    st.info("Trying manual reconstruction...")
                    return self._reconstruct_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    def _reconstruct_csv(self, file_path: str) -> pd.DataFrame:
        """Manually reconstruct CSV file"""
        st.info("Reconstructing CSV manually...")
        rows = []
        headers = None
        skipped_rows = 0

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            first_line = file.readline().strip()
            headers = [col.strip() for col in first_line.split(',')]
            st.write(f"Detected headers: {headers}")

            for line_num, line in enumerate(file, 2):
                try:
                    row = [col.strip().strip('"') for col in line.strip().split(',')]
                    if len(row) >= len(headers):
                        if len(row) > len(headers):
                            extra = ' '.join(row[len(headers)-1:])
                            row = row[:len(headers)-1] + [extra]
                        rows.append(row[:len(headers)])
                    elif len(row) == len(headers) - 1:
                        row.append('')
                        rows.append(row)
                    else:
                        skipped_rows += 1
                        if skipped_rows <= 5:
                            st.warning(f"Skipping malformed row {line_num}: wrong column count")
                except Exception as e:
                    skipped_rows += 1
                    if skipped_rows <= 5:
                        st.warning(f"Skipping row {line_num}: {str(e)}")

        if not rows:
            raise Exception("No valid rows found in CSV")

        df = pd.DataFrame(rows, columns=headers)
        st.success(f"CSV reconstructed! Shape: {df.shape}, Skipped: {skipped_rows} rows")
        return df

    def analyze_dataset(self, df):
        """Analyze the loaded dataset"""
        st.header("Dataset Analysis")
        st.write(f"Shape: {df.shape}")
        st.write("Columns:", list(df.columns))

        df.columns = df.columns.str.strip()

        text_column = None
        label_column = None

        text_patterns = ['email text', 'text', 'email', 'message', 'content', 'body']
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(pattern in col_lower for pattern in text_patterns):
                text_column = col
                break

        label_patterns = ['email type', 'type', 'label', 'class', 'target']
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(pattern in col_lower for pattern in label_patterns):
                label_column = col
                break

        if text_column is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample_length = df[col].astype(str).str.len().mean()
                    if sample_length > 50:
                        text_column = col
                        break

        if label_column is None:
            label_column = df.columns[-1]

        st.write(f"Text column detected: `{text_column}`")
        st.write(f"Label column detected: `{label_column}`")

        if text_column and label_column:
            st.subheader("Class distribution")
            st.write(df[label_column].value_counts())
            st.subheader("Sample rows")
            st.write(df[[text_column, label_column]].head(3))

        self.dataset_info = {
            'text_column': text_column,
            'label_column': label_column,
            'shape': df.shape,
            'columns': list(df.columns)
        }

        return text_column, label_column

    def extract_email_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive features from email text"""
        if pd.isna(text) or not isinstance(text, str):
            text = ""

        raw_text = str(text)
        text = raw_text.lower()

        features = {}
        features['email_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0

        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', raw_text.lower())
        features['url_count'] = len(urls)

        suspicious_domains = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'short.link']
        features['has_suspicious_url'] = int(any(domain in text for domain in suspicious_domains))

        phishing_keywords = [
            'urgent', 'immediate', 'verify', 'suspend', 'suspended', 'click here', 'act now',
            'confirm', 'update', 'expire', 'expires', 'expiring', 'limited time', 'winner',
            'congratulations', 'free', 'prize', 'lottery', 'inheritance', 'security alert',
            'account locked', 'unauthorized', 'fraud', 'refund', 'claim now', 'act fast',
            'final notice', 'last chance', 'opportunity', 'selected', 'chosen', 'million',
            'thousand', 'dollars', 'pounds', 'euros', 'payment', 'transaction', 'billing'
        ]
        features['suspicious_word_count'] = sum(text.count(word) for word in phishing_keywords)

        financial_terms = [
            'bank', 'account', 'credit', 'debit', 'payment', 'money', 'transfer', 'paypal',
            'visa', 'mastercard', 'transaction', 'billing', 'invoice', 'charge', 'fee'
        ]
        features['financial_terms'] = sum(text.count(term) for term in financial_terms)

        urgency_terms = ['asap', 'immediately', 'urgent', 'hurry', 'quick', 'fast', 'now', 'today']
        features['urgency_count'] = sum(text.count(term) for term in urgency_terms)

        features['caps_ratio'] = len(re.findall(r'[A-Z]', raw_text)) / max(len(raw_text), 1)
        features['exclamation_count'] = raw_text.count('!')
        features['question_count'] = raw_text.count('?')

        features['email_mentions'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', raw_text))
        features['phone_mentions'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', raw_text))

        features['has_click_here'] = int('click here' in text or 'click now' in text)
        features['has_verify_account'] = int('verify' in text and 'account' in text)
        features['has_update_info'] = int(('update' in text and 'information' in text) or ('update' in text and 'details' in text))

        return features

    def preprocess_data(self, df, text_column, label_column):
        """Preprocess dataset for training"""
        st.info("Preprocessing data...")
        df[text_column] = df[text_column].fillna('').astype(str)
        feature_dicts = df[text_column].apply(self.extract_email_features)
        features = pd.DataFrame(list(feature_dicts))

        target_series = df[label_column]
        unique_values = target_series.unique()
        st.write(f"Unique target values: {unique_values}")

        if len(unique_values) == 2:
            if 'phishing' in str(unique_values).lower():
                phishing_label = [val for val in unique_values if 'phishing' in str(val).lower()][0]
                target = (target_series == phishing_label).astype(int)
            elif 'safe' in str(unique_values).lower():
                safe_label = [val for val in unique_values if 'safe' in str(val).lower()][0]
                target = (target_series != safe_label).astype(int)
            else:
                target = (target_series == unique_values[1]).astype(int)
        else:
            raise ValueError(f"Expected binary classification, got {len(unique_values)} classes: {unique_values}")

        st.success(f"Features extracted: {features.shape}")
        st.write("Target distribution:", target.value_counts().to_dict())
        return features, target

    def train_model(self, features, target):
        """Train multiple models and select the best"""
        st.header("Training Models")
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )

        st.write(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")

        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=2000,
                C=1.0
            )
        }

        best_model = None
        best_score = 0
        results = {}

        for name, model in models.items():
            st.subheader(f"Training {name} ...")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')

            results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_test': y_test
            }

            st.write(f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
            st.write(f"CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            if f1 > best_score:
                best_score = f1
                best_model = name

        self.model = results[best_model]['model']
        self.is_trained = True
        self.training_stats = results[best_model]

        st.success(f"Best model: {best_model} with F1: {best_score:.3f}")
        self.display_results(results, best_model)
        return results

    def display_results(self, results, best_model):
        """Display training results using matplotlib + seaborn"""
        st.subheader("Model Evaluation")
        perf_df = pd.DataFrame(results).T
        st.write(perf_df[['accuracy', 'precision', 'recall', 'f1_score']].round(3))

        y_test = results[best_model]['y_test']
        y_pred = results[best_model]['y_pred']

        fig = plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Safe', 'Phishing'],
                   yticklabels=['Safe', 'Phishing'])
        plt.title(f'Confusion Matrix\n{best_model}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plt.subplot(1, 3, 2)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [results[best_model][metric] for metric in metrics]
        bars = plt.bar(metrics, values)
        plt.title('Performance Metrics')
        plt.ylim(0, 1)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        plt.subplot(1, 3, 3)
        model_names = list(results.keys())
        f1_scores = [results[name]['f1_score'] for name in model_names]
        bars = plt.bar(model_names, f1_scores)
        plt.title('Model Comparison (F1-Score)')
        plt.ylim(0, 1)
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)

    def predict_email(self, email_text):
        """Predict if email is phishing"""
        if not self.is_trained:
            return {"error": "Model not trained yet!"}

        features = self.extract_email_features(email_text)
        feature_df = pd.DataFrame([features])
        prediction = self.model.predict(feature_df)[0]
        probabilities = self.model.predict_proba(feature_df)[0]
        confidence = max(probabilities)

        result = {
            'prediction': 'Phishing Email' if prediction == 1 else 'Safe Email',
            'probability_safe': probabilities[0],
            'probability_phishing': probabilities[1],
            'confidence': confidence,
            'risk_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low',
            'features': features
        }
        return result

# ---------- Streamlit App Layout ----------
st.title("📧 Phishing Email Detection")
st.markdown("Upload dataset, train model, and test emails. Logic preserved from original code.")

detector = EmailPhishingDetector()

# Sidebar: upload dataset or use existing file path
st.sidebar.header("Dataset & Model")
uploaded_file = st.sidebar.file_uploader("Upload Phishing_Email.csv (optional)", type=["csv"])
use_saved_model = st.sidebar.checkbox("Load saved model (phishing_model.pkl) if available", value=True)

# Try loading saved model if checkbox True and file exists
model_loaded_from_file = False
if use_saved_model:
    try:
        model = joblib.load("phishing_model.pkl")
        detector.model = model
        detector.is_trained = True
        st.sidebar.success("Loaded phishing_model.pkl")
        model_loaded_from_file = True
    except Exception:
        st.sidebar.info("phishing_model.pkl not found or failed to load.")

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

# If no upload, see if local CSV exists
if df is None:
    try:
        # try to load local file
        df = pd.read_csv("Phishing_Email.csv")
        st.info("Loaded local Phishing_Email.csv")
    except Exception:
        df = None

if df is not None:
    text_col, label_col = detector.analyze_dataset(df)
else:
    st.warning("No dataset available. Upload a CSV or add Phishing_Email.csv to the repo.")

# Training section
st.header("Train Model")
if df is not None:
    if st.button("Preprocess & Train"):
        try:
            features, target = detector.preprocess_data(df, text_col, label_col)
            with st.spinner("Training models..."):
                results = detector.train_model(features, target)
            # save model
            joblib.dump(detector.model, "phishing_model.pkl")
            st.success("Model trained and saved to phishing_model.pkl")
        except Exception as e:
            st.error(f"Training failed: {e}")
else:
    st.info("Provide dataset to enable training.")

# Prediction UI
st.header("Predict Email")
example_selector = st.selectbox("Try Examples", [
    "Select an example...",
    "🚨 Phishing Example 1",
    "🚨 Phishing Example 2",
    "🚨 Phishing Example 3",
    "✅ Safe Example 1",
    "✅ Safe Example 2",
    "✅ Safe Example 3"
])

examples = {
    '🚨 Phishing Example 1': 'URGENT ACTION REQUIRED! Your bank account has been temporarily suspended due to suspicious activity. To reactivate your account immediately, please click here and verify your login credentials: http://bit.ly/bank-verify-urgent. Failure to act within 24 hours will result in permanent account closure. This is a final notice.',
    '🚨 Phishing Example 2': 'Congratulations! You have been selected as the winner of our $500,000 international email lottery! To claim your prize money, please reply with your full name, address, phone number, and bank account details. This is a limited time offer that expires in 48 hours. Click here to claim: http://lottery-winner.co/claim',
    '🚨 Phishing Example 3': 'SECURITY ALERT: We have detected unauthorized access to your PayPal account from an unknown device. Your account has been temporarily limited for your protection. Please verify your identity immediately by clicking the link below: http://paypal-security.net/verify. If you do not verify within 6 hours, your account will be permanently suspended.',
    '✅ Safe Example 1': 'Hi Sarah, I hope you are doing well. I wanted to follow up on our meeting yesterday about the quarterly marketing campaign. Could you please send me the budget breakdown we discussed? I need to review it before presenting to the board next week. Thanks for your time and looking forward to your response. Best regards, John',
    '✅ Safe Example 2': 'Dear Customer, Thank you for your recent order #ORD-789123. Your purchase of the wireless headphones has been confirmed and will be shipped within 2-3 business days. You will receive a tracking number via email once your order has been dispatched. If you have any questions, please contact our customer service team.',
    '✅ Safe Example 3': 'Hello team, I hope everyone had a great weekend. Just a reminder that we have our monthly team meeting scheduled for tomorrow at 10 AM in the conference room. We will be discussing the progress on current projects and planning for next quarter. Please bring your project status reports. See you all tomorrow!'
}

email_text = st.text_area("Enter email content to analyze", value="")

if example_selector != "Select an example...":
    email_text = examples.get(example_selector, email_text)

if st.button("Analyze Email"):
    if not detector.is_trained:
        st.warning("Model is not trained. Upload dataset and click 'Preprocess & Train', or load phishing_model.pkl.")
    elif not email_text.strip():
        st.warning("Please enter email text to analyze.")
    else:
        st.info("Analyzing...")
        result = detector.predict_email(email_text)
        if 'error' in result:
            st.error(result['error'])
        else:
            is_phishing = result['prediction'] == 'Phishing Email'
            st.markdown(f"### Result: {'🚨 Phishing Email' if is_phishing else '✅ Safe Email'}")
            st.write(f"Confidence: {result['confidence']:.2%}")
            st.write(f"Risk level: {result['risk_level']}")
            st.write(f"Phishing probability: {result['probability_phishing']:.2%}")
            st.write(f"Safe probability: {result['probability_safe']:.2%}")

            st.subheader("Feature analysis")
            key_features = ['suspicious_word_count', 'url_count', 'exclamation_count',
                            'financial_terms', 'urgency_count', 'has_click_here']
            feats = result['features']
            feat_df = pd.DataFrame.from_dict(feats, orient='index', columns=['value'])
            st.table(feat_df.loc[key_features])

st.markdown("---")
st.markdown("**Note:** This app preserves your original training & feature logic. The UI here is Streamlit-based for easy deployment.")
