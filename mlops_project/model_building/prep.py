import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("mlops_project/data/tourism.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
df.drop(columns=["CustomerID"], inplace=True)

# Split features and target

# NOTE: Categorical columns (like 'TypeofContact' or 'ProductPitched') are intentionally left as raw strings.
# The training pipeline will one-hot-encode them, and the Streamlit app also expects
# raw values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) ProdTaken ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
