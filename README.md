# 🏬 Mall Customer Spending Prediction using Linear Regression

## 📘 Project Overview

This project explores how demographic and income-related factors influence customer spending behavior in a mall setting.  
Using the **Mall Customers Dataset**, a **Linear Regression model** is trained to predict each customer’s **Spending Score (1–100)** based on:
- Gender  
- Age  
- Annual Income (k$)

The purpose of this project is to understand whether a simple linear model can effectively explain human purchasing patterns.

---

## 🎯 Objective

To analyze how well a Linear Regression model can predict spending behavior and to evaluate **why linear models may fall short in modeling complex human decisions**.

---

## 🧩 Dataset Information

**Dataset Name:** Mall_Customers.csv  
**Attributes:**
- `CustomerID` — Unique identifier for each customer  
- `Gender` — Male or Female  
- `Age` — Age of the customer  
- `Annual Income (k$)` — Annual income of the customer in thousands  
- `Spending Score (1–100)` — Score assigned by the mall based on customer behavior and spending nature

**Target Variable:** `Spending Score (1–100)`

---

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Libraries Used:**
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Dropped the `CustomerID` column.  
   - Encoded the `Gender` column using LabelEncoder.  
   - Split the dataset into features (`X`) and target (`y`).

2. **Model Training**
   - Used `train_test_split` (80/20 split).  
   - Applied **Linear Regression** from scikit-learn.  
   - Fitted the model using training data.

3. **Model Evaluation**
   - Evaluated performance using:
     - Mean Squared Error (MSE)  
     - R² Score (Coefficient of Determination)

4. **Visualization**
   - Scatter plot of **Actual vs Predicted Spending Scores**.  
   - Bar chart of **Feature Coefficients** to analyze influence of each variable.

---

## 📊 Results

- **R² Score:** ~0.25  
- **Mean Squared Error:** (Refer to console output)

➡️ The R² value of 0.25 indicates that only **25% of the variance** in spending scores can be explained by the model.  
This suggests that **linear regression is not sufficient to capture the complexity of human spending behavior.**

---

## 💡 Insights & Discussion

1. **Non-Linearity of Behavior** — Human spending patterns are non-linear and influenced by emotional and social factors not captured in numeric data.  
2. **Limited Features** — Only demographic and income variables are used; missing psychological or lifestyle data reduces predictive power.  
3. **Interactions Ignored** — Linear models assume independent effects of each variable and fail to capture feature interactions (e.g., how age and income combine).  
4. **Behavioral Complexity** — A single equation cannot describe all customer types; segmentation or advanced models are more suitable.

---

## 🧾 Hypothesis

> Linear Regression cannot fully explain human spending behavior because decision-making involves non-linear, psychological, and contextual factors that go beyond measurable demographic variables.

---

## 🚀 Future Work

To improve model accuracy and behavioral insight:
- Use **non-linear models** (Random Forests, Decision Trees, XGBoost, Neural Networks).  
- Perform **Customer Segmentation** using K-Means or Hierarchical Clustering.  
- Add **behavioral features** such as shopping frequency, product preferences, or satisfaction scores.  
- Explore **Polynomial Regression** or interaction terms for better curve fitting.

---

## 🧩 How to Run the Project

### 1. Clone this Repository
```bash
git clone https://github.com/<feisaladen>/customer-spending-analysis.git
cd customer-spending-analysis
