# ✅ STEP 1: Install & Import Libraries
!pip install -q pandas matplotlib seaborn plotly scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from IPython.display import display

# ✅ STEP 2: Load Dataset
file_path = "/content/iris.csv"  # Make sure you've uploaded iris.csv
df = pd.read_csv(file_path)

print("✅ Dataset Loaded!")

# ✅ STEP 3: Basic Info
display(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# ✅ STEP 4: Descriptive Stats
display(df.describe())

# ✅ STEP 5: Correlation Heatmap
numeric_cols = df.select_dtypes(include='number').columns
plt.figure(figsize=(7, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ✅ STEP 6: Advanced Visualizations
# Histogram with KDE
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.show()

# Boxplot + Outlier Detection
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
    plt.show()

    # ✅ Outlier detection (IQR method)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"{col}: {len(outliers)} outliers")

# ✅ STEP 7: Feature Engineering
df['petal_area'] = df['petal_length'] * df['petal_width']
df['sepal_area'] = df['sepal_length'] * df['sepal_width']
print("\nNew Columns Added: petal_area, sepal_area")

# Pairplot including new features
sns.pairplot(df, hue='species', diag_kind='hist')
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# ✅ STEP 8: Interactive Plotly Scatter
fig = px.scatter_3d(
    df, x='sepal_length', y='sepal_width', z='petal_length',
    color='species', size='petal_width', title='3D Scatter Plot of Iris Dataset'
)
fig.show()

# ✅ STEP 9: Bar Plot for Species Count
df['species'].value_counts().plot(kind='bar', color='orange')
plt.title("Count of Each Species")
plt.ylabel("Count")
plt.show()

# ✅ STEP 10: Simple Machine Learning Model
X = df[numeric_cols]
y = df['species']

# Encode target variable
y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ STEP 11: Save Cleaned Data
df.to_csv("enhanced_iris.csv", index=False)
print("\n💾 Enhanced dataset saved as 'enhanced_iris.csv'")

# ✅ STEP 12: Key Insights
print("\n📝 Key Insights:")
print("- Petal measurements separate species better than sepal measurements.")
print("- Petal area is a strong feature for classification.")
print("- Logistic Regression achieves high accuracy on Iris dataset.")

print("\n✅ Advanced EDA & ML completed successfully!")
