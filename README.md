# DataVisualization_Iris
"Data visualization and EDA on the Iris dataset using Python—includes charts, correlations, and a simple ML model with ~96% accuracy."
# 📊 Data Visualization & EDA – Iris Dataset

## 🔹 Project Overview
This project performs **Exploratory Data Analysis (EDA)** and **data visualization** on the famous **Iris dataset** using Python.  
It demonstrates how raw data can be transformed into clear visual insights, highlights important patterns, and even builds a simple predictive model.

---

## ✅ Features
- Data loading, inspection, and cleaning
- Descriptive statistics & correlation analysis
- Histograms, boxplots, and pairplots
- Outlier detection (IQR method)
- Interactive 3D scatter plots
- Feature engineering (petal area)
- Species-wise comparison visualizations
- Logistic Regression model (~96% accuracy)
- Saved cleaned dataset (`enhanced_iris.csv`)

---

## 📖 Data Story – Why This Matters
- **Petal size** separates species more effectively than sepal size.  
- *Setosa* has very small petals → easy to classify visually.  
- *Versicolor* and *Virginica* overlap but differ when using petal measurements.  
- Petal length & width have a **0.96 correlation**—one almost predicts the other.  
- A simple logistic regression reaches **>95% accuracy**, proving good EDA helps in quick, reliable predictions.

---

## 📂 Dataset
- **Source:** [Iris dataset – Kaggle](https://www.kaggle.com/datasets/uciml/iris) or built-in via `seaborn.load_dataset("iris")`.
- **Rows:** 150 | **Columns:** 5 (4 numeric features + species).

---

## 💻 Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

---

## 🚀 How to Run
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `iris.csv` (or use the built-in seaborn dataset).
3. Paste the notebook code.
4. Run all cells → visualizations and analysis will appear.

---

## 📊 Sample Visuals
*(Add screenshots of your plots here)*

```markdown
![Heatmap](heatmap.png)
![3D Scatter](scatter3d.png)
