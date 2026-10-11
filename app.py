
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #142338);
    color: #f1f5f9;
}
[data-testid="stMetric"] {
    background: #1e293b;
    padding: 18px;
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

models_dir = BASE_DIR / "models"
results_file = BASE_DIR / "results" / "model_metrics.csv"
data_file = BASE_DIR / "data" / "sms_spam.csv"

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in {
        "Naive Bayes": "naive_bayes.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "Linear SVM": "linear_svm.joblib"
    }.items():
        path = models_dir / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded

models = load_models()


# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live SMS Detector",
        "Model Evaluation",
        "Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("M.Tech AI/ML | Assignment 18")

# Load saved evaluation results
if results_file.exists():
    results = pd.read_csv(results_file)
else:
    results = pd.DataFrame()

if page == "Overview":
    st.header("Project Overview")
    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Size", "5,572")
    c2.metric("Spam Messages", "747")
    c3.metric("Spam Percentage", "13.41%")

    st.info(
        "This project investigates how evaluation metrics affect "
        "the ranking of classifiers on an imbalanced SMS dataset."
    )

    if not results.empty:
        st.subheader("Evaluation Snapshot")
        st.dataframe(results.round(4), use_container_width=True)

        metrics = [
            "Accuracy", "Minority-Class Recall", "Macro-F1",
            "Balanced Accuracy", "MCC"
        ]
        available = [m for m in metrics if m in results.columns]

        if available:
            metric = st.selectbox("Select evaluation metric", available)
            chart = px.bar(
                results, x="Model", y=metric, color="Model",
                title=f"Model Comparison — {metric}",
                range_y=[0, 1]
            )
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning(
            "Evaluation file not found. Run the notebook cell that saves "
            "results_df to results/model_metrics.csv."
        )

elif page == "Live SMS Detector":
    st.header("📩 Live SMS Spam Detector")
    st.write("Enter a message and compare all trained classifiers.")

    message = st.text_area(
        "SMS message",
        placeholder="Congratulations! You have won a free prize...",
        height=130
    )

    if st.button("🔍 Analyze SMS", type="primary"):
        if not message.strip():
            st.warning("Please enter an SMS message.")
        elif not models:
            st.error(
                "No trained models found. Save the models from your notebook "
                "into the project's models folder first."
            )
        else:
            cols = st.columns(len(models))

            for col, (name, model) in zip(cols, models.items()):
                prediction = model.predict([message])[0]
                is_spam = str(prediction).lower() == "spam"

                with col:
                    st.subheader(name)
                    if is_spam:
                        st.error("🚨 SPAM")
                    else:
                        st.success("✅ HAM")
                    st.caption(f"Predicted label: {prediction}")

                    # Show confidence only if model supports probabilities.

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba([message])[0]

                        # Get class labels from the final classifier in the pipeline
                        classes = list(model.named_steps["classifier"].classes_)
                        classes = [str(label).lower() for label in classes]

                        if "spam" in classes:
                            spam_index = classes.index("spam")
                            spam_probability = probabilities[spam_index]

                            st.metric(
                                "Spam probability",
                                f"{spam_probability * 100:.1f}%"
                            )
                        else:
                            st.warning("Spam class not found in model classes.")
                    else:
                        st.caption("Probability score not available.")


elif page == "Model Evaluation":
    st.header("📊 Model Evaluation")

    if results.empty:
        st.warning("Run the notebook and save model_metrics.csv first.")
    else:
        st.dataframe(results.round(4), use_container_width=True)
        st.subheader("Model Accuracy")

        accuracy_row = results.set_index("Model")["Accuracy"]

        cols = st.columns(len(accuracy_row))

        for col, (name, score) in zip(cols, accuracy_row.items()):
            col.metric(name, f"{score:.2%}")

        metric_columns = [
            "Accuracy", "Minority-Class Recall", "Macro-F1",
            "Balanced Accuracy", "MCC"
        ]
        available = [m for m in metric_columns if m in results.columns]

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
                title="Comparison Across All Evaluation Metrics"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "Minority-Class Recall" in results.columns:
            best = results.loc[
                results["Minority-Class Recall"].idxmax()
            ]
            st.success(
                f"Best minority-class recall: {best['Model']} "
                f"({best['Minority-Class Recall']:.2%})"
            )

elif page == "Dataset Explorer":
    st.header("🗂️ Dataset Explorer")

    if data_file.exists():
        df = pd.read_csv(data_file)

        if {"label", "message"}.issubset(df.columns):
            counts = df["label"].value_counts().rename_axis(
                "Class"
            ).reset_index(name="Messages")

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
                    use_container_width=True
                )

            st.caption(f"Total records: {len(df):,}")
        else:
            st.error(
                "Expected columns 'label' and 'message' in sms_spam.csv."
            )
    else:
        st.error(f"Dataset not found: {data_file}")

st.markdown("---")
st.caption("SpamShield AI | Educational classification experiment")
