# titanic_pipeline_optimized.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
df = pd.read_csv("train.csv")

# --- Feature Engineering ---
# Family Size and Alone
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = 1
df.loc[df["FamilySize"] > 1, "IsAlone"] = 0

# Extract Title from Name
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(
    ["Lady", "Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "Other"
)
df["Title"] = df["Title"].replace(["Mlle","Ms"], "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")
df["Title"] = LabelEncoder().fit_transform(df["Title"])

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male":0, "female":1})
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

# Drop unnecessary columns
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"], axis=1)

# --- Features & Target ---
X = df.drop("Survived", axis=1)
y = df["Survived"]

# --- Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Data preprocessing and split done.")

# --- Logistic Regression ---
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

# --- Random Forest with Hyperparameter Tuning ---
rf = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [5, 6, 7],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# --- Evaluation ---
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

print("\n--- Random Forest (Tuned) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Best Hyperparameters:", grid_search.best_params_)

# Cross-validation scores
cv_score_rf = cross_val_score(best_rf, X_scaled, y, cv=5).mean()
cv_score_lr = cross_val_score(log_reg, X_scaled, y, cv=5).mean()
print("Random Forest CV Score:", round(cv_score_rf,4))
print("Logistic Regression CV Score:", round(cv_score_lr,4))

# --- KMeans Clustering ---
kmeans = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nAverage survival per cluster:")
print(df.groupby("Cluster")["Survived"].mean())

# --- Save KMeans Scatter Plot ---
plt.figure(figsize=(8,5))
plt.scatter(df["Age"], df["Fare"], c=df["Cluster"], cmap="viridis", alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("KMeans Clustering (2 clusters)")
plt.savefig("kmeans_plot.png")
plt.close()

# --- Save Random Forest Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("\nPipeline complete! Plots saved as 'kmeans_plot.png' and 'confusion_matrix.png'.")
