# Titanic ML Pipeline

This repository contains a **complete machine learning pipeline** applied to the Titanic dataset. The project demonstrates **data preprocessing, feature engineering, supervised classification, hyperparameter tuning, and unsupervised clustering**, along with **visualizations** for better insights.

---

## 📝 Project Overview

The goal is to predict survival of passengers aboard the Titanic using structured features from the dataset. This pipeline covers:  

- Data preprocessing and cleaning  
- Feature engineering (Family size, Titles, IsAlone)  
- Encoding categorical variables  
- Scaling numerical features  
- Training supervised models: Logistic Regression and Random Forest  
- Hyperparameter tuning to improve model performance  
- Cross-validation for robust evaluation  
- KMeans clustering for unsupervised insights  
- Visualization of results (scatter plots & confusion matrices)

---

## ⚡ Key Features / Enhancements

1. **Feature Engineering:**
   FamilySize = SibSp + Parch + 1  
   IsAlone flag for single passengers  
   Title extracted from Name (Mr, Mrs, Miss, Other)  

2. **Hyperparameter Tuning for Random Forest:**
   - Used GridSearchCV to optimize n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features.  
   - Accuracy improved to 82% after tuning.  

3. **Visualization:**
   - KMeans clustering scatter plot (Age vs Fare)  
   - Random Forest confusion matrix heatmap  

---

## 🛠 Technologies Used

- Python 3.12
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn  

---

## 📊 Model Performance

| Model                    | Accuracy | CV Score |
|---------------------------|----------|----------|
| Logistic Regression       | 0.81     | 0.7946   |
| Random Forest (Tuned)     | 0.82     | 0.826    |

Average survival per cluster using KMeans:  

| Cluster | Avg Survived |
|---------|--------------|
| 0       | 0.27         |
| 1       | 0.60         |

---

## 🚀 How to Run

**1.** Clone the repo:


git clone https://github.com/SALIMAKHALAM/Titanic-ML-Pipeline.git

cd Titanic-ML-Pipeline

**2.** Install dependencies:

pip install -r requirements.txt

**3.** Place your train.csv in the project folder.

Run the pipeline:
python titanic_pipeline_optimized.py

**4.** Check generated plots:

kmeans_plot.png

confusion_matrix.png

**Improvements & Insights:**

Hyperparameter tuning and feature engineering improved the Random Forest accuracy increased to 82%
Feature engineering (Title, FamilySize, IsAlone) helped models better capture survival patterns.
KMeans clustering highlights two distinct passenger groups with different survival rates.

