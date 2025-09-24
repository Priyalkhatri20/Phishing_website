# 📧 Phishing Email Detection System

A machine learning-based system to detect **phishing emails** using text-based feature extraction and classifiers (Random Forest & Logistic Regression).  
This project is implemented in **Google Colab** with interactive UI support using `ipywidgets`.

---

## 🚀 Features
- Robust **dataset loading** (handles malformed CSVs)
- Automatic **text & label column detection**
- Advanced **email feature extraction**:
  - Email length, word count, sentence count
  - Suspicious keywords (e.g., *urgent, click here, verify*)
  - URL & domain checks
  - Financial & urgency terms
  - Capitalization, punctuation, and suspicious patterns
- **Model training** with:
  - Random Forest
  - Logistic Regression
- **Evaluation metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Cross-validation
  - Confusion Matrix & Performance Charts
- **Interactive UI** to test custom emails

---

## 📂 Project Structure
