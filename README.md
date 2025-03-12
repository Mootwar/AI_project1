# Project 1: Bias and Fairness in Predictive Modeling

**Dr. Sherine Antoun**  
**CSCI 396 – Artificial Intelligence**  
**Due Date: March 12, 2025**

---

## 1. Objective

The goal of this project is to develop a predictive machine learning model, analyze its biases, and propose mitigation strategies to improve fairness. Students will work with real-world datasets, apply data preprocessing, exploratory data analysis (EDA), and fairness assessment techniques while maintaining ethical AI considerations.

---

## 2. Datasets

Students may choose from the following datasets:

- **Adult Income Dataset** (Predict whether an individual earns above/below \$50K per year)  
  [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)

- **COMPAS Recidivism Dataset** (Predict whether a defendant is likely to reoffend)  
  [https://github.com/propublica/compas-analysis](https://github.com/propublica/compas-analysis)

- **Customer Churn Dataset** (Predict whether a customer will leave a company)  
  [https://www.kaggle.com/datasets/ajay1735/telecom-customer-churn](https://www.kaggle.com/datasets/ajay1735/telecom-customer-churn)

---

## 3. Task Breakdown

### 3.1. Dataset Selection
- Choose one of the datasets listed above.  
- Provide a brief justification for your choice, including potential ethical concerns related to bias in the dataset.

### 3.2. Data Preprocessing
- Handle missing data, outliers, and inconsistencies.  
- Normalize and standardize numeric features where necessary.  
- Justify feature selection and discuss ethical implications of omitting/including certain features.

### 3.3. Exploratory Data Analysis (EDA)
- Generate summary statistics and visualizations.  
- Identify patterns, correlations, and relationships between key variables.  
- Investigate potential biases in the dataset.

### 3.4. Model Training and Evaluation
- Train a classification model (e.g., logistic regression, decision tree, random forest, or neural network).  
- Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.  
- Analyze how well the model generalizes to unseen data.

### 3.5. Bias and Fairness Analysis
- Assess model fairness across different demographic groups (e.g., gender, race, age).  
- Compute fairness metrics such as demographic parity, equal opportunity, and disparate impact.  
- Use Explainable AI (XAI) tools to understand feature importance and model decision-making.

### 3.6. Bias Mitigation Strategies
- Propose and implement at least one bias mitigation strategy (e.g., reweighting data, adversarial debiasing, post-processing adjustments).  
- Compare model performance before and after applying fairness interventions.  
- Reflect on trade-offs between predictive accuracy and fairness.

### 3.7. Ethical Considerations and Reflection
- Discuss the societal implications of biased AI models.  
- Propose recommendations for responsible AI development in similar applications.

---

## 4. Final Deliverables

Students must submit:

- **Codebase**: Well-commented Python or R code implementing the entire pipeline.  
- **Project Report**: A structured document detailing methodology, results, fairness analysis, and ethical considerations (minimum 5 pages).  
- **Presentation**: A short slide deck (5-7 slides) summarizing findings and recommendations.

---

## 5. Evaluation Criteria

Grading will be based on:

- **Correctness (30%)**: Proper implementation of machine learning models and fairness analysis.  
- **Completeness (20%)**: Coverage of all required components.  
- **Clarity (20%)**: Well-documented code and well-written report.  
- **Insight (20%)**: Thoughtful discussion of bias, fairness, and ethical implications.  
- **Presentation (10%)**: Clear and engaging delivery of findings.

---