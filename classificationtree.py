import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Employee_Salary_Dataset.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())
print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Prepare features and target for classification
# Target: Gender (Male/Female)
# Features: Experience_Years, Age, Salary

X = df[['Experience_Years', 'Age', 'Salary']]  # Features
y = df['Gender']  # Target variable

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target classes: {y.unique()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # Limit depth to prevent overfitting
    min_samples_split=2,
    min_samples_leaf=1
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nClassification Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=X.columns,
          class_names=clf.classes_,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Classifier - Gender Prediction")
plt.tight_layout()
plt.savefig('classification_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Make predictions on new data (example)
print("\nExample Predictions:")
sample_data = pd.DataFrame({
    'Experience_Years': [3, 10, 1],
    'Age': [25, 35, 20],
    'Salary': [100000, 500000, 15000]
})

predictions = clf.predict(sample_data)
print("Sample predictions for Gender:")
for i, pred in enumerate(predictions):
    print(f"Experience: {sample_data.iloc[i]['Experience_Years']} years, "
          f"Age: {sample_data.iloc[i]['Age']}, "
          f"Salary: {sample_data.iloc[i]['Salary']} -> Predicted Gender: {pred}")

# Save the model (optional)
import joblib
joblib.dump(clf, 'gender_classifier_model.pkl')
print("\nModel saved as 'gender_classifier_model.pkl'")
