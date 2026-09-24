# Telecom_Churn_Case_Study

📊 **Project: Telecom Customer Churn Prediction Using PCA & Machine Learning**

🔍 **Objective:**
Built a predictive model to identify potential customer churn in a telecom company, enabling proactive retention strategies.

🧠 **Problem Statement:**
Customer churn is a major challenge in the telecom industry, directly impacting profitability. The goal of this project was to leverage machine learning to accurately predict whether a customer is likely to leave the service.

🛠 **Tools & Technologies Used:**

* **Python Libraries:** NumPy, Pandas, Scikit-learn, Statsmodels
* **IDE:** Jupyter Notebook
* **Machine Learning Algorithms:**

  * Logistic Regression
  * Random Forest Classifier
  * XGBoost Classifier
* **Dimensionality Reduction:** Principal Component Analysis (PCA)
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

📈 **Approach:**

1. **Data Exploration & Preprocessing:**

   * Handled missing values, outliers, and categorical encoding
   * Performed feature scaling using StandardScaler
   * Conducted exploratory data analysis (EDA) for insights into churn patterns

2. **Dimensionality Reduction with PCA:**

   * Reduced multicollinearity
   * Improved model interpretability and performance

3. **Model Building:**

   * Trained and tuned multiple models: Logistic Regression, Random Forest, and XGBoost
   * Used `GridSearchCV` for hyperparameter tuning
   * Compared model performances using classification reports and ROC curves

4. **Model Interpretation:**

   * Assessed feature importance in tree-based models
   * Interpreted logistic regression coefficients using Statsmodels for statistical significance

🏆 **Outcome:**
Achieved an optimal balance between precision and recall, enabling the identification of at-risk customers with high confidence. XGBoost emerged as the best-performing model in terms of ROC-AUC score.

🔗 **Key Learnings:**

* Importance of data preprocessing and feature engineering
* How PCA enhances model performance by reducing noise
* Comparative analysis of linear vs. ensemble models for classification
* Model interpretability using statistical techniques

---

📌 **Looking Ahead:**
This project highlighted how machine learning can bring real business value in the telecom domain. I'm excited to explore deeper techniques like survival analysis and customer lifetime value modeling in the future.


This repo is created as a part of Kaggle competition, use this with caution. 
