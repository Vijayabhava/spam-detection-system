# 🛡️ SpamShield AI — SMS Spam Detection

A machine learning project for detecting SMS spam by comparing three classifiers on an imbalanced dataset.

 📌 Overview

SpamShield AI classifies SMS messages into two categories:

- **Ham:** Legitimate messages
- **Spam:** Unwanted or potentially deceptive messages

The project compares **Naive Bayes, Logistic Regression, and Linear SVM** using five evaluation metrics to study model performance, class imbalance, and classification errors.

 🎯 Research Objective

To investigate whether the choice of evaluation metric changes which machine learning model performs best for SMS spam detection.

 📊 Dataset

- **Dataset:** SMS Spam Collection
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Total messages:** 5,572
- **Ham:** 4,825 (86.59%)
- **Spam:** 747 (13.41%)

Since the dataset is imbalanced, accuracy alone may not adequately represent spam detection performance.

 🧠 Models

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (Linear SVM)

 📏 Evaluation Metrics

- Accuracy
- Minority-Class Recall (Spam Recall)
- Macro-F1
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)

 🏆 Best Result

**Linear SVM** achieved the best results across all five evaluation metrics in the recorded experiment.

| Metric | Score |
|---|---:|
| Accuracy | 98.92% |
| Spam Recall | 94.63% |
| Macro-F1 | 97.65% |
| Balanced Accuracy | 97.11% |
| MCC | 0.9531 |

 🔍 Error Analysis

The Linear SVM confusion matrix recorded:

- **True Negatives:** 962
- **False Positives:** 4
- **False Negatives:** 8
- **True Positives:** 141

Error analysis helps identify legitimate messages incorrectly flagged as spam and spam messages missed by the classifier.

 🖥️ Interactive Prototype

The Streamlit application provides:

- Project overview and dataset statistics
- Live SMS classification using three models
- Model performance comparison
- Dataset exploration
- False-positive and false-negative analysis

 ⚙️ Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF
- Streamlit
- Joblib
- Matplotlib

 🚀 Run Locally

Clone the repository:

    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd <YOUR_PROJECT_FOLDER>

Create and activate a virtual environment on Windows:

    python -m venv .venv
    .venv\Scripts\activate

Install dependencies:

    python -m pip install -r requirements.txt

Run the application:

    streamlit run app.py

Ensure the required dataset and trained model files are available at the paths expected by the application. Update the commands if your repository uses different script names.

🧪 Hypothesis Verdict

**The alternative hypothesis was not supported by the observed results.**

All five metrics produced the same ranking:

1. Linear SVM
2. Logistic Regression
3. Naive Bayes

The choice of metric did not change the top-ranked model in this experiment. This conclusion is limited to the evaluated dataset, classifiers, and experimental setup.

 🔮 Future Scope

- Evaluate additional datasets and real-world SMS messages
- Investigate decision-threshold tuning
- Improve model explainability
- Explore multilingual spam detection
- Test robustness against evolving spam patterns

 🎓 Project Information

**Project:** SpamShield AI — SMS Spam Detection  
**Program:** M.Tech CSE (AI & ML)  
**Institution:** Birla Institute of Technology, Mesra

**Presented By:** Aditya Banerjee (MT/AI/10006/26) and Sumit Sharma (MT/AI/10031/26)

---

**SpamShield AI — Detect spam. Evaluate fairly. Understand the errors.**
