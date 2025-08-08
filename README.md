# 👔 IBM HR Attrition Analysis

This project analyzes and predicts **employee attrition** using the IBM HR Analytics dataset.  
It applies **Exploratory Data Analysis (EDA)**, **Feature Engineering**, and **Machine Learning** to uncover attrition patterns, identify key factors influencing turnover, and build predictive models.

Attrition prediction enables HR teams to implement **proactive retention strategies**, reducing costs and improving workforce stability.

---

## 📌 Project Objectives
1. **Identify Key Factors Influencing Attrition** — age, job satisfaction, overtime, income, tenure, etc.  
2. **Predict Employee Turnover** — using classification models.  
3. **Visualize Attrition Trends** — dashboards, charts, and statistical plots.  
4. **Enhance Retention Strategies** — actionable HR recommendations.  

---

## 📂 Dataset
- **Source:** [IBM HR Analytics Employee Attrition & Performance Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
- **Size:** 1,470 records × 35 features  
- **Type:** Categorical + numerical attributes  
- **Target:** `Attrition` (Yes/No)  

---

## 🛠 Methodology

### **Workflow**

1. **Data Preprocessing** — Handle missing values, encode categorical features, normalize numerical variables.  
2. **Exploratory Data Analysis** — Attrition breakdowns by demographics, role, income, work-life balance, etc.  
3. **Feature Engineering** — Derived features for tenure, overtime ratios, promotion history.  
4. **Model Training** — Random Forest, Gradient Boosting, XGBoost.  
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, and k-fold cross-validation.  

---

## 📊 Results

### **Model Performance**
| Model                      | Accuracy | 
|----------------------------|----------|
| **Random Forest**          | **85.32%** |
| Gradient Boosting          | 85%      |
| XGBoost                    | 85%      | 
| XGBoost (10-Fold CV Avg)   | **87%**  | 

**Best Model:** XGBoost with 10-Fold Cross-Validation — *Avg. Accuracy: 87%*

---

### **Comparison with Related Work**
| Study                                      | Dataset                     | Best Model       | Accuracy |
|--------------------------------------------|------------------------------|------------------|----------|
| N. Bhartiya et al., 2019                    | Proprietary HR dataset       | Random Forest    | 85%      |
| H. Alqahtani et al., 2024                   | Mixed HR datasets            | Gradient Boosting| 86%      |
| **Our Study (IBM Dataset)**                 | IBM HR Analytics dataset     | **XGBoost**      | **87%**  |

---

## 🔍 Key Insights
- **Work-Life Balance** — Poor balance strongly linked to higher attrition.  
- **Job Satisfaction** — Low satisfaction significantly increases turnover risk.  
- **Monthly Income** — Employees in lower pay brackets leave more often.  
- **Age** — Younger employees have higher attrition rates.  
- **Roles** — Sales Representatives face the highest attrition probability.  


---

## 💡 Recommendations
1. **Improve Work-Life Balance** — Flexible hours, remote work, wellness programs.  
2. **Boost Job Satisfaction** — Conduct regular feedback surveys, address concerns promptly.  
3. **Revise Compensation Structures** — Align with market benchmarks, provide incentives.  
4. **Career Growth Initiatives** — Mentorship programs, skill development, promotion pathways.  
5. **Target High-Risk Segments** — Focus retention strategies on roles and demographics flagged by the model.  

---
