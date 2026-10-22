
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = BASE_DIR / "data" / "sms_spam.csv"
METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
ERROR_FILE = RESULTS_DIR / "linear_svm_error_analysis.csv"

# -------------------- STYLING --------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #142338);
    color: #f1f5f9;
}
[data-testid="stMetric"] {
    background: #1e293b;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #334155;
}
h1, h2, h3 {
    color: #67e8f9;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ SpamShield AI")
st.caption("SMS Spam Detection | Imbalanced Classification Lab")

# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_models():
    model_files = {
        "Naive Bayes": "naive_bayes.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "Linear SVM": "linear_svm.joblib"
    }

    loaded = {}
    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            try:
                loaded[name] = joblib.load(path)
            except Exception as exc:
                st.error(f"Could not load {name}: {exc}")

    return loaded


models = load_models()


def get_classes(model):
    """Get class labels from a fitted pipeline or estimator."""
    if hasattr(model, "classes_"):
        return [str(x).lower() for x in model.classes_]

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return [str(x).lower() for x in step.classes_]

    return []


def predict_spam_probability(model, message):
    """Return spam probability if supported."""
    if not hasattr(model, "predict_proba"):
        return None

    classes = get_classes(model)
    if "spam" not in classes:
        return None

    probabilities = model.predict_proba([message])[0]
    return float(probabilities[classes.index("spam")])


def get_explanation_components(model):
    """Find TF-IDF vectorizer and final classifier in a pipeline."""
    if not hasattr(model, "named_steps"):
        return None, None

    vectorizer = None
    for step in model.named_steps.values():
        if (
            hasattr(step, "get_feature_names_out")
            and hasattr(step, "transform")
        ):
            vectorizer = step

    classifier = list(model.named_steps.values())[-1]
    return vectorizer, classifier


def get_linear_coefficients(classifier):
    """Get coefficients from logistic regression or calibrated linear SVM."""
    if hasattr(classifier, "coef_"):
        return np.asarray(classifier.coef_)[0]

    calibrated_models = getattr(
        classifier, "calibrated_classifiers_", []
    )

    coefficients = []
    for calibrated_model in calibrated_models:
        estimator = getattr(calibrated_model, "estimator", None)
        if estimator is None:
            estimator = getattr(
                calibrated_model, "base_estimator", None
            )

        if estimator is not None and hasattr(estimator, "coef_"):
            coefficients.append(np.asarray(estimator.coef_)[0])

    if coefficients:
        # Mean coefficient vector across fitted calibrated estimators.
        return np.mean(coefficients, axis=0)

    return None


def explain_message(model, message, top_n=8):
    """
    Show influential active TF-IDF features.
    Positive contributions support spam; negative support ham.
    """
    vectorizer, classifier = get_explanation_components(model)

    if vectorizer is None:
        return None, "TF-IDF vectorizer not found."

    feature_names = vectorizer.get_feature_names_out()
    transformed = vectorizer.transform([message])
    values = transformed.toarray()[0]

    if hasattr(classifier, "feature_log_prob_"):
        classes = [str(x).lower() for x in classifier.classes_]
        if "spam" not in classes or "ham" not in classes:
            return None, "Could not identify ham and spam classes."

        spam_idx = classes.index("spam")
        ham_idx = classes.index("ham")
        weights = (
            classifier.feature_log_prob_[spam_idx]
            - classifier.feature_log_prob_[ham_idx]
        )
        method = "Naive Bayes log-likelihood contributions"

    else:
        weights = get_linear_coefficients(classifier)

        if weights is None:
            return None, "This model does not expose compatible feature weights."

        estimator_classes = getattr(classifier, "classes_", None)

        if estimator_classes is None:
            calibrated = getattr(
                classifier, "calibrated_classifiers_", []
            )
            if calibrated:
                estimator_classes = getattr(
                    calibrated[0].estimator, "classes_", None
                )

        if estimator_classes is not None:
            estimator_classes = [
                str(x).lower() for x in estimator_classes
            ]
            if len(estimator_classes) == 2:
                if estimator_classes[1] != "spam":
                    weights = -weights

        method = "Linear-model feature contributions"

    contributions = values * weights

    active = [
        (feature_names[i], float(contributions[i]))
        for i in np.flatnonzero(values)
    ]

    positive = sorted(
        [item for item in active if item[1] > 0],
        key=lambda item: item[1],
        reverse=True
    )[:top_n]

    negative = sorted(
        [item for item in active if item[1] < 0],
        key=lambda item: item[1]
    )[:top_n]

    explanation = {
        "method": method,
        "spam_support": pd.DataFrame(
            positive, columns=["Word / phrase", "Contribution"]
        ),
        "ham_support": pd.DataFrame(
            negative, columns=["Word / phrase", "Contribution"]
        )
    }

    return explanation, None


# -------------------- SIDEBAR --------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live SMS Detector",
        "Model Evaluation",
        "Error Analysis",
        "Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("M.Tech AI/ML | Assignment 18")

# -------------------- LOAD RESULTS --------------------

if METRICS_FILE.exists():
    results = pd.read_csv(METRICS_FILE)
else:
    results = pd.DataFrame()


# -------------------- OVERVIEW --------------------

if page == "Overview":
    st.header("📌 Project Overview")

    if DATA_FILE.exists():
        try:
            df_overview = pd.read_csv(DATA_FILE)
            total = len(df_overview)
            spam_count = (
                df_overview["label"].astype(str).str.lower() == "spam"
            ).sum()
            spam_percent = spam_count / total * 100 if total else 0
        except Exception:
            total, spam_count, spam_percent = 5572, 747, 13.41
    else:
        total, spam_count, spam_percent = 5572, 747, 13.41

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Size", f"{total:,}")
    c2.metric("Spam Messages", f"{spam_count:,}")
    c3.metric("Spam Percentage", f"{spam_percent:.2f}%")

    st.info(
        "This project investigates model performance on an imbalanced "
        "SMS classification dataset using multiple evaluation metrics."
    )

    st.subheader("Available Features")
    st.markdown("""
    - **Live SMS Detector:** Predict using three trained classifiers.
    - **Explainability:** Inspect words supporting a prediction where
      compatible model weights are available.
    - **Model Agreement:** Check whether the models agree on the label.
    - **Model Evaluation:** Compare saved evaluation metrics.
    - **Error Analysis:** Inspect false positives and false negatives
      from the saved test-set analysis.
    """)

    if not results.empty:
        st.subheader("Evaluation Snapshot")
        st.dataframe(results.round(4), use_container_width=True)
    else:
        st.warning(
            "Evaluation metrics file not found. Save model_metrics.csv "
            "from the experiment notebook to the results folder."
        )


# -------------------- LIVE SMS DETECTOR --------------------

elif page == "Live SMS Detector":
    st.header("📩 Live SMS Detector")
    st.write(
        "Compare all trained classifiers and inspect the words "
        "that support their predictions."
    )

    message = st.text_area(
        "Enter SMS message",
        placeholder="Congratulations! You have won a free prize...",
        height=140
    )

    analyze = st.button("🔍 Analyze SMS", type="primary")

    if analyze:
        if not message.strip():
            st.warning("Please enter a message first.")

        elif not models:
            st.error(
                "No trained models were loaded. Check the models folder."
            )

        else:
            st.subheader("Model Predictions")
            prediction_rows = []

            for name, model in models.items():
                try:
                    prediction = str(model.predict([message])[0]).lower()
                    probability = predict_spam_probability(model, message)

                    prediction_rows.append({
                        "Model": name,
                        "Prediction": prediction,
                        "Spam probability": probability
                    })
                except Exception as exc:
                    st.error(f"{name} prediction failed: {exc}")

            if prediction_rows:
                prediction_df = pd.DataFrame(prediction_rows)
                st.dataframe(
                    prediction_df,
                    use_container_width=True,
                    hide_index=True
                )

                labels = prediction_df["Prediction"].dropna()
                agreement = labels.nunique() == 1

                if agreement:
                    st.success(
                        f"All {len(labels)} available models agree: "
                        f"**{labels.iloc[0].upper()}**"
                    )
                else:
                    st.warning(
                        "The models disagree. Review their predictions "
                        "before making a decision."
                    )

                st.caption(
                    "Spam probabilities are model estimates, not guarantees."
                )

                st.divider()
                st.subheader("🧠 Explainable Prediction")

                selected_model_name = st.selectbox(
                    "Choose model to explain",
                    list(models.keys()),
                    key="explanation_model"
                )

                selected_model = models[selected_model_name]
                explanation, explanation_error = explain_message(
                    selected_model, message
                )

                if explanation_error:
                    st.info(explanation_error)
                else:
                    st.caption(
                        f"Explanation method: {explanation['method']}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Words supporting SPAM**")
                        if explanation["spam_support"].empty:
                            st.write("No positive contributing words found.")
                        else:
                            st.dataframe(
                                explanation["spam_support"].round(4),
                                use_container_width=True,
                                hide_index=True
                            )

                    with col2:
                        st.markdown("**Words supporting HAM**")
                        if explanation["ham_support"].empty:
                            st.write("No negative contributing words found.")
                        else:
                            st.dataframe(
                                explanation["ham_support"].round(4),
                                use_container_width=True,
                                hide_index=True
                            )

                    st.caption(
                        "These are model-based feature contributions, "
                        "not causal explanations. TF-IDF features and "
                        "learned weights influence the displayed scores."
                    )


# -------------------- MODEL EVALUATION --------------------

elif page == "Model Evaluation":
    st.header("📊 Model Evaluation")

    if results.empty:
        st.warning(
            "Run the experiment notebook and save "
            "results/model_metrics.csv first."
        )

    elif "Model" not in results.columns:
        st.error("The metrics CSV must contain a 'Model' column.")

    else:
        st.dataframe(results.round(4), use_container_width=True)

        if "Accuracy" in results.columns:
            st.subheader("Accuracy")
            cols = st.columns(max(1, len(results)))

            for col, (_, row) in zip(cols, results.iterrows()):
                col.metric(
                    str(row["Model"]),
                    f"{float(row['Accuracy']):.2%}"
                )

        metric_columns = [
            "Accuracy",
            "Minority-Class Recall",
            "Macro-F1",
            "Balanced Accuracy",
            "MCC"
        ]
        available = [
            metric for metric in metric_columns
            if metric in results.columns
        ]

        if available:
            long_df = results.melt(
                id_vars="Model",
                value_vars=available,
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                long_df,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                range_y=[0, 1],
                title="Comparison Across Evaluation Metrics"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "Minority-Class Recall" in results.columns:
            best = results.loc[
                results["Minority-Class Recall"].idxmax()
            ]
            st.success(
                f"Highest minority-class recall in saved results: "
                f"{best['Model']} "
                f"({best['Minority-Class Recall']:.2%})"
            )

        st.caption(
            "These charts use the saved metrics file. They do not "
            "automatically retrain or retune models."
        )


# -------------------- ERROR ANALYSIS --------------------

elif page == "Error Analysis":
    st.header("🔎 Error Analysis")
    st.write(
        "Review false positives and false negatives from the saved "
        "Linear SVM test-set analysis."
    )

    if ERROR_FILE.exists():
        errors = pd.read_csv(ERROR_FILE)

        required_columns = {
            "message", "actual", "predicted", "error_type"
        }

        if not required_columns.issubset(errors.columns):
            st.error(
                "The error-analysis CSV is missing required columns: "
                + ", ".join(sorted(required_columns - set(errors.columns)))
            )
        else:
            fp_count = (
                errors["error_type"].astype(str) == "False Positive"
            ).sum()
            fn_count = (
                errors["error_type"].astype(str) == "False Negative"
            ).sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("False Positives", int(fp_count))
            c2.metric("False Negatives", int(fn_count))
            c3.metric("Total Errors", int(len(errors)))

            st.markdown("""
            - **False Positive:** A ham message incorrectly predicted as spam.
            - **False Negative:** A spam message incorrectly predicted as ham.
            """)

            if "error_type" in errors.columns:
                error_counts = (
                    errors["error_type"].value_counts()
                    .rename_axis("Error Type")
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    error_counts,
                    x="Error Type",
                    y="Count",
                    color="Error Type",
                    title="Misclassification Counts"
                )
                st.plotly_chart(fig, use_container_width=True)

            selected_type = st.selectbox(
                "Filter errors",
                ["All", "False Positive", "False Negative"]
            )

            filtered = errors.copy()
            if selected_type != "All":
                filtered = filtered[
                    filtered["error_type"] == selected_type
                ]

            search_text = st.text_input(
                "Search message text",
                placeholder="Type a word or phrase..."
            )

            if search_text.strip():
                filtered = filtered[
                    filtered["message"].astype(str).str.contains(
                        search_text,
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download displayed errors as CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="spamshield_error_analysis.csv",
                mime="text/csv"
            )

            st.info(
                "These counts describe the saved analysis file. "
                "They are not recomputed by this page."
            )

    else:
        st.warning(
            "Error-analysis file not found. Run the notebook cell that "
            "creates results/linear_svm_error_analysis.csv first."
        )


# -------------------- DATASET EXPLORER --------------------

elif page == "Dataset Explorer":
    st.header("🗂️ Dataset Explorer")

    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)

            if {"label", "message"}.issubset(df.columns):
                df["label"] = df["label"].astype(str).str.lower().str.strip()

                counts = (
                    df["label"].value_counts()
                    .rename_axis("Class")
                    .reset_index(name="Messages")
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Class Distribution")
                    fig = px.pie(
                        counts,
                        names="Class",
                        values="Messages",
                        hole=0.55
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Dataset Samples")
                    selected = st.selectbox(
                        "Choose class",
                        sorted(df["label"].dropna().unique())
                    )
                    st.dataframe(
                        df[df["label"] == selected].head(10),
                        use_container_width=True,
                        hide_index=True
                    )

                st.caption(f"Total records: {len(df):,}")

                st.subheader("Search Dataset")
                search = st.text_input(
                    "Search messages",
                    placeholder="Enter a word or phrase..."
                )

                if search.strip():
                    matches = df[
                        df["message"].astype(str).str.contains(
                            search,
                            case=False,
                            na=False,
                            regex=False
                        )
                    ]
                    st.write(f"Matches found: {len(matches)}")
                    st.dataframe(
                        matches.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

            else:
                st.error(
                    "Expected 'label' and 'message' columns in sms_spam.csv."
                )

        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")

    else:
        st.error(f"Dataset not found: {DATA_FILE}")


# -------------------- FOOTER --------------------

st.markdown("---")
st.caption(
    "SpamShield AI | Educational classification experiment | "
    "Predictions may be incorrect."
)
