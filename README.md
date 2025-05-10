Importing Libraries and Uploading Data
# === Upload CSV File ===
from google.colab import files
import io

uploaded = files.upload()

# === Import Libraries ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Load Dataset ===
filename = list(uploaded.keys())[0]  # Automatically get uploaded filename
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# === Display Head ===
print("Dataset Head:")
print(df.head())
Use code with caution
Import necessary libraries:

google.colab.files: Provides functions to interact with files in Google Colab.
io: Handles input/output operations.
pandas: Used for data manipulation and analysis. It provides data structures like DataFrames.
matplotlib.pyplot: Used for creating visualizations (plots and charts).
seaborn: Builds on top of matplotlib to create more statistically informative and visually appealing plots.
sklearn: Provides tools for machine learning, including model selection, training, and evaluation.
Upload the CSV file:

This code utilizes Google Colab's file upload feature to allow the user to upload their CSV data file. The uploaded variable stores the uploaded file.
Load the dataset:

pandas reads the CSV file into a DataFrame called df. DataFrames are table-like structures that are excellent for organizing and working with data in Python.
Display the head of the dataset:

df.head() shows the first few rows of the DataFrame, giving you a quick preview of the data.
Data Preparation
# === Select and Clean Data ===
selected_features = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                     'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition']

# Drop rows with missing values in selected columns only
df = df[selected_features].dropna()

# One-hot encode categorical variable
df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)
Use code with caution
Select Features:

selected_features is a list containing the names of the columns (features) that will be used for the model.
Handle Missing Data:

dropna() is used to remove rows with any missing values in the selected features.
One-Hot Encoding:

pd.get_dummies() converts the 'Weather_Condition' column (which is likely categorical) into numerical form using one-hot encoding. This is essential for many machine learning algorithms that work best with numerical data.
Model Training and Evaluation
# === Split Features and Target ===
X = df.drop('Severity', axis=1)
y = df['Severity']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predict and Evaluate ===
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Confusion Matrix Plot ===
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
Use code with caution
Split Data:

X contains the features (independent variables), and y contains the target (dependent variable, 'Severity').
train_test_split() divides the data into training and testing sets. The model will be trained on the training set and evaluated on the testing set.
Train the Model:

A RandomForestClassifier model is created, which is an ensemble learning method known for its accuracy and robustness.
model.fit() trains the model using the training data (X_train and y_train).
Make Predictions:

model.predict() uses the trained model to predict the target variable ('Severity') for the test data (X_test). The predictions are stored in y_pred.
Evaluate the Model:

classification_report() provides metrics like precision, recall, F1-score, and support to evaluate the model's performance.
A confusion matrix is created and visualized using seaborn.heatmap(). It helps to understand the types of errors the model is making.
In summary, this code takes a CSV file as input, preprocesses the data, trains a Random Forest model to predict 'Severity', and then evaluates the model's performance using various metrics and visualizations
