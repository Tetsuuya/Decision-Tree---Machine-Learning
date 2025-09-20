# Decision Tree Analysis - Employee Salary Dataset

This repository contains decision tree implementations for analyzing the Employee Salary Dataset, including both classification and regression models.

## 📁 Project Structure

```
multiple/
├── classificationtree.py          # Gender classification using decision tree
├── regressiontree.py             # Salary prediction using decision tree
├── Employee_Salary_Dataset.csv   # Dataset with employee information
├── myenv/                        # Virtual environment
├── Readme.md                     # This file
└── Generated Files:
    ├── classification_tree.png   # Classification tree visualization
    ├── regression_tree.png       # Regression tree visualization
    ├── regression_analysis.png   # Regression analysis plots
    ├── gender_classifier_model.pkl    # Saved classification model
    └── salary_regressor_model.pkl     # Saved regression model
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd multiple
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate

# On macOS/Linux:
source myenv/bin/activate
```

### 3. Install Required Packages
```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib joblib

# Or install from requirements.txt (if you create one):
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import pandas, numpy, sklearn, matplotlib; print('All packages installed successfully!')"
```

## 📊 Dataset Information

The `Employee_Salary_Dataset.csv` contains the following columns:
- **ID**: Employee identifier
- **Experience_Years**: Years of work experience
- **Age**: Employee age
- **Gender**: Male/Female
- **Salary**: Annual salary in currency units

## 🔧 Usage

### Classification Tree (Gender Prediction)
```bash
python classificationtree.py
```

**What it does:**
- Predicts employee gender based on Experience_Years, Age, and Salary
- Generates classification accuracy, confusion matrix, and feature importance
- Creates visualization of the decision tree
- Saves the trained model as `gender_classifier_model.pkl`

### Regression Tree (Salary Prediction)
```bash
python regressiontree.py
```

**What it does:**
- Predicts employee salary based on Experience_Years, Age, and Gender
- Generates regression metrics (MSE, RMSE, MAE, R²)
- Creates multiple visualizations including tree structure and analysis plots
- Saves the trained model as `salary_regressor_model.pkl`

## 📈 Expected Outputs

### Classification Tree Outputs:
- Classification accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Feature importance ranking
- Decision tree visualization (`classification_tree.png`)
- Example predictions

### Regression Tree Outputs:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²) score
- Feature importance ranking
- Decision tree visualization (`regression_tree.png`)
- Analysis plots (`regression_analysis.png`)
- Salary analysis by gender and experience ranges

## 🛠️ Requirements

### Python Packages:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
joblib>=1.0.0
```

### Python Version:
- Python 3.7 or higher

## 📝 Code Examples

### Using the Classification Model:
```python
import joblib
import pandas as pd

# Load the trained model
clf = joblib.load('gender_classifier_model.pkl')

# Make predictions
new_data = pd.DataFrame({
    'Experience_Years': [5, 10, 2],
    'Age': [30, 40, 25],
    'Salary': [100000, 500000, 30000]
})

predictions = clf.predict(new_data)
print(predictions)  # ['Male', 'Female', 'Male']
```

### Using the Regression Model:
```python
import joblib
import pandas as pd

# Load the trained model
reg = joblib.load('salary_regressor_model.pkl')

# Make predictions
new_data = pd.DataFrame({
    'Experience_Years': [5, 10, 2],
    'Age': [30, 40, 25],
    'Gender_encoded': [1, 0, 1]  # 1=Male, 0=Female
})

predictions = reg.predict(new_data)
print(predictions)  # [salary1, salary2, salary3]
```

## 🔍 Model Performance

### Classification Model:
- **Target**: Gender (Male/Female)
- **Features**: Experience_Years, Age, Salary
- **Evaluation**: Accuracy, Precision, Recall, F1-score

### Regression Model:
- **Target**: Salary
- **Features**: Experience_Years, Age, Gender_encoded
- **Evaluation**: MSE, RMSE, MAE, R²

## 🎯 Key Features

1. **Data Preprocessing**: Automatic train-test split (80/20)
2. **Model Tuning**: Optimized hyperparameters to prevent overfitting
3. **Comprehensive Evaluation**: Multiple metrics and visualizations
4. **Feature Importance**: Shows which features are most predictive
5. **Visualization**: Decision tree plots and analysis charts
6. **Model Persistence**: Saves trained models for future use
7. **Example Predictions**: Demonstrates model usage

## 🚨 Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Make sure virtual environment is activated and packages are installed
2. **FileNotFoundError**: Ensure `Employee_Salary_Dataset.csv` is in the same directory
3. **ImportError**: Check if all required packages are installed correctly

### Solutions:
```bash
# Reinstall packages
pip install --upgrade pandas numpy scikit-learn matplotlib joblib

# Check Python version
python --version

# Verify file location
ls -la Employee_Salary_Dataset.csv
```

## 📞 Support

If you encounter any issues:
1. Check that all dependencies are installed
2. Verify the dataset file is present
3. Ensure Python version compatibility
4. Check file permissions

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Analyzing! 🎉**
