# Task2.py

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the cleaned dataset
df = pd.read_csv("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-1//cleaned_titanic.csv")

# Display first few rows
print(df.head())

# Display basic information
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Histogram for Age
plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-2//histogram_age.png")
plt.show()

# Histogram for Fare
plt.figure(figsize=(8, 5))
plt.hist(df['Fare'], bins=30, color='salmon', edgecolor='black')
plt.title('Histogram of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-2//histogram_fare.png")
plt.show()

# Boxplot of Age
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Age'], color='lightgreen')
plt.title('Boxplot of Age')
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-2//boxplot_age.png")
plt.show()

# Boxplot of Fare
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Fare'], color='lightblue')
plt.title('Boxplot of Fare')
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-2//boxplot_fare.png")
plt.show()

# Correlation Matrix
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-2//correlation_matrix.png")
plt.show()

# Pairplot for selected features
selected_columns = ['Survived', 'Pclass', 'Age', 'Fare']
sns.pairplot(df[selected_columns])
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-2//pairplot.png")
plt.show()

# Example plot with Plotly
fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Age vs Fare (Colored by Survival)')
fig.write_html("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-2//age_fare_scatter.html")
fig.show()
# Saving new CSV File
df.to_csv("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-2//eda_titanic.csv", index=False)

