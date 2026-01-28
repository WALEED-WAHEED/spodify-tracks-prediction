# Canadian Job Market Analysis - Labor Market Trends Prediction

A comprehensive machine learning model that analyzes national job posting trends in Canada to understand labour market demand across occupations, industries, and regions. This model identifies patterns, skills needs, and hiring pressure to generate insights useful for employers, policymakers, and labour market analysts.

## 📋 Overview

This project implements advanced machine learning models to analyze Canadian job postings data from the Government of Canada's National Job Bank. The analysis focuses on:

- **Job Demand Prediction**: Predicting vacancy counts across different occupations and regions
- **Salary Trend Analysis**: Understanding compensation patterns across industries
- **Regional Analysis**: Identifying labor market trends by province/territory
- **Industry Insights**: Analyzing demand patterns across NAICS sectors
- **Occupational Analysis**: Understanding trends by NOC (National Occupational Classification) codes

## 📊 Dataset

- **Source**: Government of Canada Open Data Portal
- **Title**: Job Postings Advertised on Canada's National Job Bank Website
- **Structure**: Monthly CSV files with detailed job posting information
- **Variables**: 
  - Job title, NOC code, NAICS code
  - Location (Province/Territory, City)
  - Vacancies, salary, benefits
  - Job requirements, employment terms
- **Rationale**: Large, credible dataset updated monthly, rich in occupational and regional detail

## 📁 Project Structure

```
spodify-tracks-prediction/
│
├── linear_regression_modeling.py  # Main model implementation
├── example_usage.py               # Usage examples and demonstrations
├── job-bank-open-data-all-job-postings-en-december2025.csv  # Job postings dataset
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── linear_regression_results.png  # Generated visualization (after running)
└── feature_importance.png         # Feature importance plot (after running)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this project**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Model

#### Option 1: Quick Start (Full Pipeline)

Run the complete modeling pipeline:

```bash
python linear_regression_modeling.py
```

This will:
- Load and preprocess the job postings data
- Split into training and test sets
- Scale features and create polynomial features
- Train multiple models and select the best one
- Validate with cross-validation
- Generate predictions
- Create visualizations

#### Option 2: Using the Example Script

See various usage examples:

```bash
python example_usage.py
```

## 💻 Usage Guide

### Basic Usage

```python
from linear_regression_modeling import JobMarketAnalysis

# Initialize the model
model = JobMarketAnalysis(
    csv_path='job-bank-open-data-all-job-postings-en-december2025.csv'
)

# Run the complete pipeline
metrics = model.run_full_pipeline(
    target_column='Vacancy Count',
    test_size=0.2,
    random_state=42,
    use_scaled=True,
    cv_folds=5
)

# View results
print(f"Test R² Score: {metrics['test_r2']:.4f}")
print(f"Test RMSE: {metrics['test_rmse']:.4f}")
```

### Step-by-Step Usage

For more control over the process:

```python
from linear_regression_modeling import JobMarketAnalysis

# Initialize
model = JobMarketAnalysis(
    csv_path='job-bank-open-data-all-job-postings-en-december2025.csv'
)

# Step 1: Load data
model.load_data()

# Step 2: Preprocess
X, y = model.preprocess_data(target_column='Vacancy Count')

# Step 3: Split data
model.split_data(test_size=0.2, random_state=42)

# Step 4: Scale features
model.scale_features()

# Step 5: Train model
model.train_best_model(use_scaled=True)

# Step 6: Validate
metrics = model.validate_model(use_scaled=True, cv_folds=5)

# Step 7: Make predictions
predictions = model.make_predictions(use_scaled=True)

# Step 8: Visualize results
model.visualize_results()
```

### Predicting Salary Trends

To analyze salary trends instead of vacancy counts:

```python
model = JobMarketAnalysis(
    csv_path='job-bank-open-data-all-job-postings-en-december2025.csv'
)

metrics = model.run_full_pipeline(
    target_column='Salary_Annual',  # Predict annual salary
    test_size=0.2,
    random_state=42,
    use_scaled=True,
    cv_folds=5
)
```

## 📊 Model Features

### Features Used

The model uses the following features:

1. **Salary Information**: Annual salary estimates, salary ranges
2. **Location**: Province/Territory (encoded)
3. **Employment Details**: Employment type, employment term
4. **Education**: Education level required (LOS)
5. **Experience**: Experience level required
6. **Industry**: NAICS sector codes
7. **Occupation**: NOC major group codes
8. **Work Hours**: Minimum and maximum hours per week

### Model Performance

The model provides:
- **R² Score**: Coefficient of determination (how well the model fits)
- **RMSE**: Root Mean Squared Error (prediction error)
- **MAE**: Mean Absolute Error (average prediction error)
- **Cross-Validation**: K-fold cross-validation for robust evaluation

### Visualizations

The model generates:
- Feature importance/coefficient plots
- Actual vs Predicted scatter plots
- Residual plots
- Error distribution charts
- Model performance metrics summary

## 📦 Dependencies

All required packages are listed in `requirements.txt`:

- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- scipy >= 1.10.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

## 🔧 Configuration Options

### Model Parameters

- `target_column`: Name of the target variable (default: 'Vacancy Count')
  - Options: 'Vacancy Count', 'Salary_Annual', or other numeric columns
- `test_size`: Proportion of data for testing (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)
- `use_scaled`: Whether to use scaled features (default: True)
- `cv_folds`: Number of cross-validation folds (default: 5)

### Feature Engineering

The model automatically:
- Converts hourly salaries to annual equivalents
- Extracts NOC major groups and NAICS sectors
- Encodes categorical variables (provinces, employment types, etc.)
- Creates polynomial interaction features
- Selects the most important features

## 📈 Understanding the Results

### Metrics Explained

- **R² Score**: 
  - Range: -∞ to 1.0
  - 1.0 = Perfect predictions
  - 0.0 = Model performs as well as predicting the mean
  - Negative = Model performs worse than predicting the mean

- **RMSE**: 
  - Lower is better
  - Measured in the same units as the target variable
  - For vacancy count: number of vacancies
  - For salary: dollars

- **MAE**: 
  - Lower is better
  - Average absolute difference between predictions and actual values
  - Easier to interpret than RMSE

### Interpreting Feature Importance

The model shows feature coefficients/importance indicating:
- **Positive coefficient**: Feature increases the target (e.g., more vacancies, higher salary)
- **Negative coefficient**: Feature decreases the target
- **Larger absolute value**: Stronger influence on the target

## 🎯 Use Cases

This model is useful for:

1. **Employers**: 
   - Understanding competitive salary ranges
   - Identifying regions with high demand
   - Planning recruitment strategies

2. **Job Seekers**:
   - Understanding market demand for their skills
   - Identifying high-demand regions
   - Salary expectations by location/industry

3. **Policymakers**:
   - Identifying labor market trends
   - Understanding regional disparities
   - Planning workforce development programs

4. **Labor Market Analysts**:
   - Analyzing hiring patterns
   - Understanding skills demand
   - Regional economic analysis

## 🛠️ Troubleshooting

### Common Issues

1. **FileNotFoundError: job-bank-open-data-all-job-postings-en-december2025.csv**
   - Ensure the CSV file is in the same directory as the script
   - Check the file path in the `csv_path` parameter

2. **Encoding errors**
   - The script tries multiple encodings (UTF-16, UTF-8, Latin-1)
   - If issues persist, check the file encoding manually

3. **Missing dependencies**
   - Run: `pip install -r requirements.txt`
   - Ensure you're using the correct Python version

4. **Memory errors with large datasets**
   - Reduce dataset size or use data sampling
   - Close other applications to free memory

5. **Import errors**
   - Activate your virtual environment
   - Reinstall packages: `pip install --upgrade -r requirements.txt`

## 📝 Notes

- The model automatically handles missing values and outliers
- Categorical variables are automatically encoded
- Salary conversion assumes 40 hours/week, 52 weeks/year for hourly rates
- The model selects the best performing algorithm automatically
- Feature engineering creates interaction terms to capture complex relationships

## 🔍 Data Preprocessing

The model performs extensive preprocessing:

1. **Salary Conversion**: Converts hourly to annual salaries for consistency
2. **Feature Extraction**: Extracts NOC major groups and NAICS sectors
3. **Encoding**: Encodes categorical variables (provinces, employment types)
4. **Missing Value Handling**: Fills numeric missing values with medians
5. **Outlier Handling**: Caps or removes extreme values using IQR method
6. **Feature Scaling**: Normalizes features to mean=0, std=1
7. **Polynomial Features**: Creates interaction terms
8. **Feature Selection**: Selects top K most important features

## 📧 Support

For questions or issues, please refer to the code documentation or contact the development team.

## 📄 License

This project is provided as-is for client use.

---

**Version**: 2.0  
**Last Updated**: January 2026  
**Dataset**: Government of Canada Open Data Portal - Job Bank Postings
