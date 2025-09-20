import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Employee_Salary_Dataset.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())
print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Prepare features and target for regression
# Target: Salary
# Features: Experience_Years, Age, Gender (encoded)

# Encode Gender as numerical values
df_encoded = df.copy()
df_encoded['Gender_encoded'] = df_encoded['Gender'].map({'Male': 1, 'Female': 0})

X = df_encoded[['Experience_Years', 'Age', 'Gender_encoded']]  # Features
y = df_encoded['Salary']  # Target variable

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Salary range: ${y.min():,} - ${y.max():,}")
print(f"Average salary: ${y.mean():,.2f}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train the decision tree regressor
reg = DecisionTreeRegressor(
    random_state=42,
    max_depth=5,  # Limit depth to prevent overfitting
    min_samples_split=2,
    min_samples_leaf=1
)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nRegression Metrics:")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize the decision trees
plt.figure(figsize=(20, 10))
plot_tree(reg, 
          feature_names=X.columns,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Regressor - Salary Prediction")
plt.tight_layout()
plt.savefig('regression_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Make predictions on new data (example)
print("\nExample Predictions:")
sample_data = pd.DataFrame({
    'Experience_Years': [3, 10, 1, 20],
    'Age': [25, 35, 20, 45],
    'Gender_encoded': [1, 0, 1, 0]  # 1=Male, 0=Female
})

predictions = reg.predict(sample_data)
print("Sample predictions for Salary:")
for i, pred in enumerate(predictions):
    gender = "Male" if sample_data.iloc[i]['Gender_encoded'] == 1 else "Female"
    print(f"Experience: {sample_data.iloc[i]['Experience_Years']} years, "
          f"Age: {sample_data.iloc[i]['Age']}, "
          f"Gender: {gender} -> Predicted Salary: ${pred:,.2f}")

# Create a comparison plot
plt.figure(figsize=(12, 8))

# Actual vs Predicted scatter plot
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.grid(True, alpha=0.3)

# Residuals plot
plt.subplot(2, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True, alpha=0.3)

# Feature importance bar plot
plt.subplot(2, 2, 3)
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)

# Salary distribution
plt.subplot(2, 2, 4)
plt.hist(y, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Salary Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the model (optional)
import joblib
joblib.dump(reg, 'salary_regressor_model.pkl')
print("\nModel saved as 'salary_regressor_model.pkl'")

# Additional analysis: Salary by gender
print("\nSalary Analysis by Gender:")
salary_by_gender = df.groupby('Gender')['Salary'].agg(['count', 'mean', 'median', 'std'])
print(salary_by_gender)

# Salary by experience ranges
print("\nSalary Analysis by Experience:")
df['Experience_Range'] = pd.cut(df['Experience_Years'], 
                               bins=[0, 2, 5, 10, 20, 100], 
                               labels=['0-2', '3-5', '6-10', '11-20', '20+'])
salary_by_exp = df.groupby('Experience_Range')['Salary'].agg(['count', 'mean', 'median'])
print(salary_by_exp)
