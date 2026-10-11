
import pandas as pd
from pathlib import Path

# Dataset path
input_path = Path("data/spam.csv")
output_path = Path("data/sms_spam.csv")

# Check whether the dataset exists
if not input_path.exists():
    raise FileNotFoundError(
        "spam.csv not found inside the data folder!"
    )

# Load the dataset
df = pd.read_csv(input_path, encoding="latin-1")

print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())

# Select the required columns
if {"v1", "v2"}.issubset(df.columns):
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]

elif {"label", "message"}.issubset(df.columns):
    df = df[["label", "message"]].copy()

else:
    raise ValueError(
        "Expected columns v1/v2 or label/message."
    )

# Remove missing values
df = df.dropna(subset=["label", "message"]).copy()

# Standardize labels and message text
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()

# Keep valid classes only
df = df[df["label"].isin(["ham", "spam"])]

# Remove empty messages
df = df[df["message"] != ""]

# Save the cleaned dataset
df.to_csv(output_path, index=False, encoding="utf-8")

# Display dataset information
print("\nDataset preparation completed!")
print("Cleaned dataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nClass percentages:")
print(
    (df["label"].value_counts(normalize=True) * 100).round(2)
)

print("\nSample messages:")
print(df.head())

print(f"\nSaved successfully to: {output_path}")
