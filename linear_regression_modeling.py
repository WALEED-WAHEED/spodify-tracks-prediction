"""
Labor Market Analysis and Modeling for Canadian Job Postings

This script analyzes national job posting trends in Canada to understand 
labour market demand across occupations, industries, and regions. It identifies 
patterns, skills needs, and hiring pressure to generate insights useful for 
employers, policymakers, and labour market analysts.

Dataset: Job Postings Advertised on Canada's National Job Bank Website
Source: Government of Canada Open Data Portal

Features analyzed:
- Job title, NOC code, NAICS code
- Location (Province/Territory, City)
- Vacancies, salary, benefits
- Job requirements, employment terms
"""

# ============================================================================
# IMPORT STATEMENTS - Loading necessary libraries
# ============================================================================

# pandas: Data manipulation and analysis - used for reading CSV files and working with DataFrames
# Think of it as Excel for Python - helps organize and manipulate tabular data
import pandas as pd

# numpy: Numerical computing library - provides mathematical operations and array handling
# Used for calculations like mean, std, sqrt, and working with arrays efficiently
import numpy as np

# sklearn.model_selection: Tools for splitting data and model evaluation
# - train_test_split: Divides data into training and testing sets (80/20 split typically)
# - cross_val_score: Evaluates model using cross-validation (testing multiple times on different splits)
# - KFold: Creates folds for cross-validation (divides data into K equal parts)
# - GridSearchCV: Automatically finds best model parameters by testing different combinations
# - RandomizedSearchCV: Faster alternative to GridSearchCV, samples random parameter combinations
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV

# sklearn.linear_model: Different types of linear regression models
# - LinearRegression: Basic linear regression (finds a line/plane that best fits data)
# - Ridge: Linear regression with L2 regularization (prevents overfitting by penalizing large coefficients)
# - Lasso: Linear regression with L1 regularization (can remove unimportant features by setting coefficients to zero)
# - ElasticNet: Combines Ridge and Lasso regularization (best of both worlds)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# sklearn.ensemble: Advanced models that combine multiple simpler models for better predictions
# - RandomForestRegressor: Uses many decision trees and averages their predictions (reduces overfitting)
# - GradientBoostingRegressor: Builds models sequentially, each new model corrects errors of previous ones
# - VotingRegressor: Combines multiple models by averaging their predictions
# - StackingRegressor: Uses a meta-model to combine predictions from multiple base models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor

# XGBoost and LightGBM: State-of-the-art gradient boosting libraries (often outperform sklearn's GradientBoosting)
# These are optional - will use if available, otherwise fall back to sklearn models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# sklearn.metrics: Functions to measure how good our model predictions are
# - mean_squared_error: Average of squared differences between predictions and actual values (penalizes large errors more)
# - mean_absolute_error: Average of absolute differences (easier to interpret, treats all errors equally)
# - r2_score: R-squared score (0-1, higher is better, shows proportion of variance explained by model)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# sklearn.preprocessing: Tools to prepare data before training models
# - StandardScaler: Normalizes features (makes them have mean=0, std=1) so all features are on same scale
# - PolynomialFeatures: Creates new features by combining existing ones (e.g., x*y interactions)
# - LabelEncoder: Converts categorical text labels (like "Ontario", "Quebec") into numbers (0, 1, 2...)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

# sklearn.feature_selection: Tools to select the most important features
# - SelectKBest: Selects the K best features based on a scoring function
# - f_regression: Statistical test (F-test) that measures how well each feature predicts the target variable
from sklearn.feature_selection import SelectKBest, f_regression

# matplotlib.pyplot: Library for creating plots and graphs (visualization)
# Used to create scatter plots, histograms, bar charts, etc. for model analysis
import matplotlib.pyplot as plt

# seaborn: Makes matplotlib plots look nicer and easier to create
# Provides better default styles and color palettes for professional-looking visualizations
import seaborn as sns

# warnings: Python's warning system
# We ignore warnings to keep output clean (warnings don't stop code from running, just clutter output)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VISUALIZATION SETTINGS - Configure plot appearance for professional output
# ============================================================================

# Set seaborn style to "whitegrid" - white background with grid lines for better readability
# This makes plots easier to read and more professional-looking
sns.set_style("whitegrid")

# Set default figure size to 14 inches wide by 10 inches tall
# Larger figures show more detail and are better for presentations/reports
plt.rcParams['figure.figsize'] = (14, 10)

# Set default font sizes for different plot elements
# This ensures all text is readable and consistent across all plots
plt.rcParams['font.size'] = 10              # General font size for all text
plt.rcParams['axes.labelsize'] = 11          # Size of axis labels (x-axis, y-axis labels)
plt.rcParams['axes.titlesize'] = 13          # Size of plot titles (main title above each subplot)
plt.rcParams['xtick.labelsize'] = 9          # Size of numbers/text on x-axis ticks
plt.rcParams['ytick.labelsize'] = 9          # Size of numbers/text on y-axis ticks
plt.rcParams['legend.fontsize'] = 10         # Size of legend text (explains what different colors/symbols mean)

# ============================================================================
# CLASS DEFINITION - JobMarketAnalysis
# ============================================================================

class JobMarketAnalysis:
    """
    Advanced Machine Learning Model for Canadian Job Market Analysis
    
    This class provides a complete analysis pipeline:
    1. Loads and preprocesses job posting data
    2. Analyzes trends across occupations, industries, and regions
    3. Trains models to predict job demand and salary trends
    4. Validates models using cross-validation
    5. Creates visualizations of labor market insights
    """
    
    def __init__(self, csv_path='job-bank-open-data-all-job-postings-en-december2025.csv', 
                 use_polynomial=True, handle_outliers=True):
        """
        Initialize the job market analysis model
        
        This method sets up all the variables (attributes) that the class will use.
        Think of it as preparing empty containers that will be filled with data as we process it.
        
        Parameters:
        -----------
        csv_path : str
            Path to the job postings CSV file
            Default: 'job-bank-open-data-all-job-postings-en-december2025.csv'
        use_polynomial : bool
            Whether to create polynomial features (interaction terms like x*y)
            True = Create polynomial features (can improve accuracy but slower and more complex)
            False = Use only original features (faster but potentially less accurate)
        handle_outliers : bool
            Whether to remove or cap extreme values in the data
            True = Handle outliers (recommended for better model performance)
            False = Keep all data as-is (may include problematic extreme values that skew results)
        """
        # Store configuration parameters
        self.csv_path = csv_path                    # Path to the data file
        self.use_polynomial = use_polynomial        # Whether to use polynomial features
        self.handle_outliers = handle_outliers     # Whether to handle outliers
        
        # ====================================================================
        # DATA STORAGE VARIABLES - Will be filled as we process data
        # ====================================================================
        # None means "empty" - we'll fill these as we load and process the data
        
        self.df = None                    # The full dataset loaded from CSV (raw data)
        self.X = None                     # Features (input variables) - what we use to make predictions
                                          # Examples: salary, province, employment type, etc.
        self.y = None                     # Target (output variable) - what we're trying to predict
                                          # Examples: vacancy count, salary, etc.
        
        # Training and testing data splits
        # We split data so we can train on one part and test on another (to see if model works on new data)
        self.X_train = None               # Features for training (used to teach the model)
        self.X_test = None                # Features for testing (used to evaluate the model)
        self.y_train = None               # Target values for training (what we want to predict)
        self.y_test = None                # Target values for testing (to compare predictions against)
        
        # ====================================================================
        # MODEL STORAGE VARIABLES
        # ====================================================================
        self.model = None                 # The trained machine learning model (will be one of: Linear, Ridge, Random Forest, etc.)
        self.best_model_name = None       # Name of the best performing model (e.g., "Random Forest")
        
        # ====================================================================
        # PREPROCESSING TOOLS - Will be fitted on training data
        # ====================================================================
        self.scaler = StandardScaler()     # Tool to normalize features (make them have mean=0, std=1)
                                          # Must fit on training data only, then transform both train and test
        self.poly_features = None         # Tool to create polynomial features (combinations like x*y)
                                          # Will be created if use_polynomial=True
        self.feature_selector = None      # Tool to select best features (removes unimportant ones)
                                          # Helps reduce complexity and improve performance
        self.label_encoders = {}          # Dictionary to store encoders for each categorical feature
                                          # Maps text labels (like "Ontario") to numbers (like 0, 1, 2...)
        
        # ====================================================================
        # FEATURE TRACKING VARIABLES
        # ====================================================================
        self.feature_names = None         # Names of final features used in model (after engineering and selection)
        self.original_feature_names = None # Names of original features before any transformations
                                          # Useful for understanding what features the model uses
        self.target_column_name = None    # Name of the target variable being predicted
                                          # Useful for tracking which target is actually being used
        
    def load_data(self):
        """
        Load and inspect the job postings dataset
        
        Returns:
        --------
        self.df : pandas.DataFrame
            The loaded dataset
        """
        print("=" * 60)
        print("LOADING JOB POSTINGS DATA")
        print("=" * 60)
        
        # ====================================================================
        # LOAD CSV FILE WITH MULTIPLE ENCODING ATTEMPTS
        # ====================================================================
        # CSV files can have different encodings (how text is stored)
        # We try multiple encodings because the file might be saved in different formats
        # - UTF-16: Unicode encoding (common for files with special characters)
        # - UTF-8: Most common Unicode encoding
        # - Latin-1: Older encoding, fallback option
        
        # Try UTF-16 first (common for files exported from Excel with special characters)
        try:
            # sep='\t' means tab-separated (tabs between columns, not commas)
            self.df = pd.read_csv(self.csv_path, sep='\t', encoding='utf-16')
        except:
            # If UTF-16 fails, try UTF-8 (most common encoding)
            try:
                self.df = pd.read_csv(self.csv_path, sep='\t', encoding='utf-8')
            except:
                # If UTF-8 fails, try Latin-1 (older encoding, works for most files)
                self.df = pd.read_csv(self.csv_path, encoding='latin-1')
        
        # ====================================================================
        # DISPLAY DATASET INFORMATION
        # ====================================================================
        # Show basic statistics about the loaded dataset
        
        # .shape returns (number_of_rows, number_of_columns)
        print(f"\nDataset shape: {self.df.shape}")
        print(f"  - Rows (job postings): {self.df.shape[0]}")      # Number of job postings
        print(f"  - Columns (features): {self.df.shape[1]}")       # Number of features/variables
        
        # Show first 20 column names (there might be many columns)
        print(f"\nColumns: {list(self.df.columns)[:20]}...")  # Show first 20 columns
        
        # Show first 5 rows - gives us a preview of what the data looks like
        # This helps us understand the structure and values in the dataset
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        # Show data types - tells us if columns are numbers, text, dates, etc.
        # Important because different types need different processing
        # .value_counts() counts how many columns of each type we have
        print(f"\nData types:")
        print(self.df.dtypes.value_counts())
        
        # Check for missing values - empty cells in our data
        # Missing values can cause problems, so we need to know how many we have
        # .isnull().sum() counts missing values per column
        # .sort_values(ascending=False) sorts by most missing first
        print(f"\nMissing values (top 20):")
        missing = self.df.isnull().sum().sort_values(ascending=False)
        print(missing.head(20))  # Show top 20 columns with most missing values
        
        return self.df
    
    def preprocess_data(self, target_column='Vacancy Count'):
        """
        Preprocess the job postings data
        
        Parameters:
        -----------
        target_column : str
            Name of the target variable (default: 'Vacancy Count')
        
        Returns:
        --------
        self.X, self.y : Features and target
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        
        # Create a copy of the original data to avoid modifying the original dataset
        # This is important - if we modify the original, we can't reload it easily
        df_processed = self.df.copy()
        
        # ====================================================================
        # FEATURE ENGINEERING - Create new features from existing ones
        # ====================================================================
        # Feature engineering is creating new features that might be more useful for prediction
        # than the original features. This helps the model find better patterns.
        
        # ====================================================================
        # SALARY CONVERSION - Convert all salary formats to annual for consistency
        # ====================================================================
        # Based on data analysis: 85% are hourly, 13% annual, 2% other (month, week, day, bi-weekly)
        # We convert everything to annual so we can compare them fairly
        # This is critical for accurate salary predictions
        if 'Salary Minimum' in df_processed.columns and 'Salary Maximum' in df_processed.columns:
            # Define a function to convert salary based on the "Salary Per" column
            def convert_to_annual(row):
                """
                Convert salary to annual equivalent based on Salary Per column
                
                Handles all formats found in data:
                - Hour: 85% of data - convert using 40 hrs/week * 52 weeks/year = 2,080 hrs/year
                - Year/Annual: 13% of data - use as-is
                - Month: ~0.6% - multiply by 12
                - Week: ~0.4% - multiply by 52
                - Day: ~0.3% - multiply by 260 (working days/year)
                - Bi-weekly: ~0.5% - multiply by 26 (bi-weekly periods/year)
                """
                # Check if salary values are missing
                if pd.isna(row['Salary Minimum']):
                    return np.nan
                
                salary_min = row['Salary Minimum']
                salary_max = row['Salary Maximum'] if not pd.isna(row['Salary Maximum']) else salary_min
                
                # Use average of min and max for better representation
                salary_avg = (salary_min + salary_max) / 2
                
                # Get salary period (how often salary is paid)
                if 'Salary Per' in row and not pd.isna(row['Salary Per']):
                    salary_per = str(row['Salary Per']).lower().strip()
                else:
                    # If Salary Per is missing, infer from value size
                    # Values < 100 are likely hourly, >= 100 are likely annual
                    if salary_avg < 100:
                        salary_per = 'hour'
                    else:
                        salary_per = 'year'
                
                # Convert based on period
                if 'hour' in salary_per:
                    # Hourly: convert to annual (40 hrs/week * 52 weeks = 2,080 hrs/year)
                    return salary_avg * 40 * 52
                elif 'year' in salary_per or 'annual' in salary_per:
                    # Already annual - use as-is
                    return salary_avg
                elif 'month' in salary_per:
                    # Monthly: multiply by 12 months/year
                    return salary_avg * 12
                elif 'week' in salary_per:
                    # Weekly: multiply by 52 weeks/year
                    return salary_avg * 52
                elif 'day' in salary_per:
                    # Daily: multiply by 260 working days/year (5 days/week * 52 weeks)
                    return salary_avg * 260
                elif 'bi-weekly' in salary_per or 'biweekly' in salary_per:
                    # Bi-weekly: multiply by 26 periods/year (52 weeks / 2)
                    return salary_avg * 26
                else:
                    # Unknown format - try to infer from value
                    if salary_avg < 100:
                        return salary_avg * 40 * 52  # Assume hourly
                    else:
                        return salary_avg  # Assume annual
            
            # Apply conversion to both min and max, then use average
            # This gives us a single annual salary estimate per job posting
            df_processed['Salary_Annual'] = df_processed.apply(convert_to_annual, axis=1)
            
            # Also create a more robust version using maximum salary
            def convert_max_to_annual(row):
                """Convert maximum salary to annual (for range analysis)"""
                if pd.isna(row['Salary Maximum']):
                    return np.nan
                salary_max = row['Salary Maximum']
                if 'Salary Per' in row and not pd.isna(row['Salary Per']):
                    salary_per = str(row['Salary Per']).lower().strip()
                else:
                    salary_per = 'hour' if salary_max < 100 else 'year'
                
                if 'hour' in salary_per:
                    return salary_max * 40 * 52
                elif 'year' in salary_per or 'annual' in salary_per:
                    return salary_max
                elif 'month' in salary_per:
                    return salary_max * 12
                elif 'week' in salary_per:
                    return salary_max * 52
                elif 'day' in salary_per:
                    return salary_max * 260
                elif 'bi-weekly' in salary_per or 'biweekly' in salary_per:
                    return salary_max * 26
                else:
                    return salary_max * 40 * 52 if salary_max < 100 else salary_max
            
            df_processed['Salary_Annual_Max'] = df_processed.apply(convert_max_to_annual, axis=1)
        
        # ====================================================================
        # SALARY RANGE FEATURE - Calculate the range between min and max salary
        # ====================================================================
        # Some jobs have a salary range (e.g., $50,000-$60,000), others have a single value
        # The range tells us how flexible the employer is with compensation
        if 'Salary Minimum' in df_processed.columns and 'Salary Maximum' in df_processed.columns:
            # Calculate the difference between maximum and minimum salary
            # Larger range = more flexibility in compensation
            df_processed['Salary_Range'] = (
                df_processed['Salary Maximum'] - df_processed['Salary Minimum']
            )
            # If Salary Maximum is missing, the range will be NaN
            # Fill NaN with 0 (meaning single-value salary, no range)
            df_processed['Salary_Range'] = df_processed['Salary_Range'].fillna(0)
        
        # ====================================================================
        # NOC CODE EXTRACTION - Extract major occupational group
        # ====================================================================
        # NOC (National Occupational Classification) codes classify jobs by occupation
        # Format: 5-digit code (e.g., 31200 = Psychologists)
        # First 2 digits represent the major group (e.g., 31 = Health occupations)
        # We extract just the major group to reduce complexity while keeping useful information
        if 'NOC21 Code' in df_processed.columns:
            # Convert to string, then take first 2 characters (major group)
            # Example: "31200" -> "31" (Health occupations major group)
            # Then convert to numeric, handling non-numeric values (like "NA", "Pr", etc.)
            df_processed['NOC_Major_Group'] = (
                df_processed['NOC21 Code'].astype(str).str[:2]
            )
            # Convert to numeric, replacing non-numeric values with NaN
            # pd.to_numeric with errors='coerce' converts invalid values to NaN
            df_processed['NOC_Major_Group'] = pd.to_numeric(
                df_processed['NOC_Major_Group'], errors='coerce'
            )
        
        # ====================================================================
        # NAICS CODE EXTRACTION - Extract industry sector
        # ====================================================================
        # NAICS (North American Industry Classification System) codes classify jobs by industry
        # Data analysis shows: NAICS is stored as TEXT (e.g., "Accommodation and food services")
        # NOT as numeric codes. We need to handle this properly.
        # We'll encode the text values directly since they're already meaningful categories
        if 'NAICS' in df_processed.columns:
            # NAICS is text format (e.g., "Accommodation and food services", "Construction")
            # We'll keep it as-is for encoding later - it's already a meaningful categorical feature
            # No need to extract numeric codes since the data uses text descriptions
            df_processed['NAICS_Sector'] = df_processed['NAICS'].astype(str)
            
            # Replace 'nan' strings with actual NaN
            df_processed['NAICS_Sector'] = df_processed['NAICS_Sector'].replace('nan', np.nan)
            
            # Also try to extract first 2 characters if some entries are numeric codes
            # This handles mixed formats (some text, some numeric)
            naics_numeric = pd.to_numeric(df_processed['NAICS'], errors='coerce')
            if naics_numeric.notna().any():
                # Some entries are numeric - extract first 2 digits
                naics_str = naics_numeric.dropna().astype(int).astype(str).str[:2]
                # Update only numeric entries
                numeric_mask = naics_numeric.notna()
                df_processed.loc[numeric_mask, 'NAICS_Sector'] = naics_str.values
        
        # ====================================================================
        # FEATURE SELECTION - Choose which columns to use for prediction
        # ====================================================================
        # Not all columns in the dataset are useful for prediction
        # We select only the features that are relevant for predicting our target variable
        # This reduces complexity and improves model performance
        
        feature_columns = []  # List to store selected feature names
        
        # ====================================================================
        # NUMERIC FEATURES - Features that are already numbers
        # ====================================================================
        # These features can be used directly in the model (no encoding needed)
        # IMPORTANT: We exclude the target variable from features to prevent data leakage!
        # Data leakage = using the target to predict itself (gives perfect but meaningless results)
        
        # ====================================================================
        # HOURS FEATURE ENGINEERING - Use average instead of min/max
        # ====================================================================
        # Data analysis shows Hours Min and Max are highly correlated (0.997)
        # Using both is redundant - we'll use average instead
        if 'Hours Minimum' in df_processed.columns and 'Hours Maximum' in df_processed.columns:
            hmin = pd.to_numeric(df_processed['Hours Minimum'], errors='coerce')
            hmax = pd.to_numeric(df_processed['Hours Maximum'], errors='coerce')
            # Calculate average hours per week
            df_processed['Hours_Average'] = (hmin + hmax) / 2
            # If only one value exists, use that
            df_processed['Hours_Average'] = df_processed['Hours_Average'].fillna(hmin).fillna(hmax)
        elif 'Hours Minimum' in df_processed.columns:
            df_processed['Hours_Average'] = pd.to_numeric(df_processed['Hours Minimum'], errors='coerce')
        elif 'Hours Maximum' in df_processed.columns:
            df_processed['Hours_Average'] = pd.to_numeric(df_processed['Hours Maximum'], errors='coerce')
        
        numeric_features = [
            'Salary_Range',       # Salary range (difference between min and max) - only if not predicting salary
            'Hours_Average',      # Average hours per week (better than min/max due to high correlation)
            'NOC_Major_Group',    # NOC major group (we extracted this above)
            # Note: NAICS_Sector is now categorical (text), not numeric
        ]
        
        # Note: Salary_Annual is NOT included here because:
        # 1. If it's the target, including it would cause data leakage (perfect but meaningless predictions)
        # 2. If it's not the target, we can optionally add it back (but be careful with Salary_Range)
        
        # Add numeric features to our list if they exist in the dataset
        # AND if they're not the target variable (critical to prevent data leakage!)
        for feat in numeric_features:
            if feat in df_processed.columns and feat != target_column:
                feature_columns.append(feat)
        
        # If target is NOT Salary_Annual, we can safely include Salary_Annual as a feature
        # This allows predicting other targets (like Vacancy Count) using salary information
        if target_column != 'Salary_Annual' and 'Salary_Annual' in df_processed.columns:
            feature_columns.append('Salary_Annual')
            print(f"Note: Including 'Salary_Annual' as a feature (target is '{target_column}')")
        
        # ====================================================================
        # CATEGORICAL FEATURES - Features that are text/categories
        # ====================================================================
        # These features need to be converted to numbers before using in the model
        # Examples: "Ontario" -> 0, "Quebec" -> 1, "British Columbia" -> 2, etc.
        categorical_features = [
            'Province/Territory',  # Location (Alberta, Ontario, Quebec, etc.) - 13 provinces/territories
            'Employment Type',     # Full-time, Part-time, etc. - 3 categories
            'Employment Term',     # Permanent, Temporary, Contract, etc. - 4 categories
            'Education LOS',      # Level of Study required - 11 categories (56.8% missing)
            'Experience Level',    # Experience required - 7 categories (56.8% missing)
            'NAICS_Sector'        # Industry sector - text format, ~26 categories (56.8% missing)
        ]
        
        # Add City as a feature if available (high cardinality, but useful for regional analysis)
        # We'll use it only if there are reasonable number of unique values
        if 'City' in df_processed.columns:
            city_unique = df_processed['City'].nunique()
            city_missing_pct = df_processed['City'].isnull().sum() / len(df_processed) * 100
            # Only include if not too many unique values (< 100) and not too many missing (< 50%)
            if city_unique < 100 and city_missing_pct < 50:
                categorical_features.append('City')
                print(f"Note: Including 'City' as feature ({city_unique} unique cities, {city_missing_pct:.1f}% missing)")
        
        # Add categorical features to our list if they exist in the dataset
        for feat in categorical_features:
            if feat in df_processed.columns:
                feature_columns.append(feat)
        
        # ====================================================================
        # VALIDATION - Check if all selected features exist in the dataset
        # ====================================================================
        # This prevents errors later if a column name is misspelled or missing
        missing_features = [f for f in feature_columns if f not in df_processed.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Remove missing features from our list (keep only features that exist)
            feature_columns = [f for f in feature_columns if f in df_processed.columns]
        
        # ====================================================================
        # SEPARATE FEATURES (X) AND TARGET (y)
        # ====================================================================
        # X = Input features (what we use to make predictions)
        # y = Target variable (what we're trying to predict)
        # This is standard machine learning notation
        
        # Check if target column exists in the dataset
        if target_column not in df_processed.columns:
            # If target doesn't exist, try to find an alternative or create a default
            print(f"Warning: '{target_column}' not found. Using 'Vacancy Count' or creating default.")
            if 'Vacancy Count' in df_processed.columns:
                # Use 'Vacancy Count' as fallback target
                target_column = 'Vacancy Count'
            else:
                # Create a default target (all ones) - not ideal but allows code to run
                # In practice, you'd want to ensure the target column exists
                df_processed[target_column] = 1
                print("Created default target variable (all ones)")
        
        # Store target column name for later reference
        self.target_column_name = target_column
        
        # Extract target variable first to check its variation
        y_temp = df_processed[target_column].copy()
        
        # ====================================================================
        # CHECK TARGET VARIABLE VARIATION
        # ====================================================================
        # If target has no variation (all same values), model can't learn anything
        # We need to check this and potentially use a different target
        
        # Remove missing values for checking
        y_temp_clean = y_temp.dropna()
        
        # Check if target has sufficient variation
        if len(y_temp_clean) > 0:
            target_std = y_temp_clean.std()
            target_unique = y_temp_clean.nunique()
            
            # If standard deviation is 0 or very small, or only 1 unique value, target has no variation
            if target_std == 0 or target_unique <= 1:
                print(f"\n⚠ Warning: Target variable '{target_column}' has no variation!")
                print(f"   - Standard deviation: {target_std}")
                print(f"   - Unique values: {target_unique}")
                print(f"   - This means all values are the same - model cannot learn patterns")
                
                # Try to use Salary_Annual as alternative target
                if 'Salary_Annual' in df_processed.columns:
                    salary_temp = df_processed['Salary_Annual'].dropna()
                    if len(salary_temp) > 0:
                        salary_std = salary_temp.std()
                        salary_unique = salary_temp.nunique()
                        
                        if salary_std > 0 and salary_unique > 1:
                            print(f"\n✅ Switching to 'Salary_Annual' as target variable")
                            print(f"   - Salary_Annual has variation: std={salary_std:.2f}, unique={salary_unique}")
                            target_column = 'Salary_Annual'
                            self.target_column_name = 'Salary_Annual'
                        else:
                            print(f"\n⚠ Salary_Annual also has no variation. Using original target.")
                else:
                    print(f"\n⚠ Salary_Annual not available. Using original target.")
                    print(f"   - Model will predict constant value (not useful for analysis)")
        
        # ====================================================================
        # CRITICAL: REMOVE TARGET FROM FEATURES (Prevent Data Leakage)
        # ====================================================================
        # Data leakage occurs when the target variable is accidentally included in features
        # This gives perfect predictions (R² = 1.0) but they're meaningless
        # Example: Predicting Salary_Annual using Salary_Annual as a feature = cheating!
        
        # Remove target column from features if it was accidentally included
        if target_column in feature_columns:
            print(f"\n⚠ CRITICAL: Removing '{target_column}' from features to prevent data leakage!")
            print(f"   - Target variable should NOT be used as a feature")
            print(f"   - This would give perfect but meaningless predictions")
            feature_columns = [f for f in feature_columns if f != target_column]
        
        # Also check for Salary_Range if target is Salary_Annual (might leak information)
        # Salary_Range is calculated from salary min/max, which directly relates to annual salary
        # Removing it for more realistic and generalizable predictions
        if target_column == 'Salary_Annual' and 'Salary_Range' in feature_columns:
            print(f"\n⚠ Removing 'Salary_Range' from features (may leak information about target)")
            print(f"   - Salary_Range is derived from salary min/max, which relates to annual salary")
            print(f"   - Removing it ensures more realistic predictions without data leakage")
            feature_columns = [f for f in feature_columns if f != 'Salary_Range']
        
        # Extract features (X) and target (y) from the processed dataframe
        # .copy() creates a copy to avoid modifying the original dataframe
        self.X = df_processed[feature_columns].copy()  # Features (input variables)
        self.y = df_processed[target_column].copy()    # Target (output variable - what we predict)
        
        # ====================================================================
        # HANDLE SKEWED TARGET DISTRIBUTION (Vacancy Count)
        # ====================================================================
        # Data analysis shows Vacancy Count is heavily skewed: 80% are 1, max is 282
        # For better model performance, we can use log transform or create binary feature
        # However, we'll keep original for interpretability and handle outliers instead
        
        # Final safety check: Ensure target is not in features
        if target_column in self.X.columns:
            raise ValueError(f"CRITICAL ERROR: Target variable '{target_column}' found in features! "
                          f"This causes data leakage. Features: {list(self.X.columns)}")
        
        # ====================================================================
        # ENCODE CATEGORICAL FEATURES - Convert text labels to numbers
        # ====================================================================
        # Machine learning models need numbers, not text
        # We convert categorical text values (like "Ontario", "Quebec") to numbers (0, 1, 2...)
        # LabelEncoder assigns a unique number to each unique category
        
        for col in categorical_features:
            if col in self.X.columns:
                # Create a LabelEncoder for this column
                # Each column gets its own encoder (so "Ontario" might be 0 in one column, 5 in another)
                le = LabelEncoder()
                
                # ====================================================================
                # IMPROVED CATEGORICAL MISSING VALUE HANDLING
                # ====================================================================
                # For categorical features with high missing rates, use mode (most common value)
                # instead of 'Unknown' for better model performance
                missing_pct = self.X[col].isnull().sum() / len(self.X) * 100
                
                if missing_pct > 50:
                    # High missing rate: use mode (most common value) for better imputation
                    mode_val = self.X[col].mode()
                    if len(mode_val) > 0:
                        fill_value = mode_val[0]
                        self.X[col] = self.X[col].fillna(fill_value)
                        print(f"  - Filled {missing_pct:.1f}% missing values in '{col}' with mode ('{fill_value}')")
                    else:
                        # If no mode (all unique), use 'Unknown'
                        self.X[col] = self.X[col].fillna('Unknown')
                else:
                    # Low missing rate: use 'Unknown' to preserve information about missingness
                    self.X[col] = self.X[col].fillna('Unknown')
                
                # Convert to string (in case some values are numbers stored as text)
                # Then fit_transform: learns the mapping AND transforms the data
                # Example: ["Ontario", "Quebec", "Ontario"] -> [0, 1, 0]
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                
                # Store the encoder so we can use it later for new data
                # This is important - we need the same mapping for test data and future predictions
                self.label_encoders[col] = le
        
        # Store original feature names before any transformations
        # We'll need these later for creating polynomial feature names and understanding results
        self.original_feature_names = list(self.X.columns)
        
        # ====================================================================
        # ENSURE NUMERIC COLUMNS ARE ACTUALLY NUMERIC
        # ====================================================================
        # Some columns that should be numeric might still be strings or objects
        # We need to convert them to numeric before scaling
        # This handles cases where data was read as strings or has mixed types
        
        for col in self.X.columns:
            # Check if column is supposed to be numeric (in our numeric_features list)
            if col in numeric_features:
                # Try to convert to numeric, replacing non-numeric values with NaN
                # errors='coerce' means invalid values become NaN instead of raising error
                self.X[col] = pd.to_numeric(self.X[col], errors='coerce')
        
        # ====================================================================
        # HANDLE MISSING VALUES - Fill empty cells with reasonable values
        # ====================================================================
        # Missing values (NaN) cause problems in machine learning models
        # We need to either remove them or fill them with reasonable values
        
        # ====================================================================
        # IMPROVED MISSING VALUE HANDLING
        # ====================================================================
        # Data analysis shows 56.8% missing Education LOS, Experience Level, NAICS
        # We need better imputation strategies
        
        # Fill numeric missing values with median (middle value)
        # Median is better than mean because it's not affected by outliers
        # Example: If most salaries are $50k-$60k but one is $500k, median stays around $55k
        # Store original values before imputation to preserve variation
        X_before_imputation = self.X.copy()
        
        for col in self.X.select_dtypes(include=[np.number]).columns:
            # Check if column has any missing values
            if self.X[col].isnull().any():
                missing_pct = self.X[col].isnull().sum() / len(self.X) * 100
                
                # If > 50% missing, use a more conservative approach
                if missing_pct > 50:
                    # For heavily missing numeric features, use median but flag it
                    # However, if most values are missing, median might not be representative
                    # Use median of non-missing values
                    median_val = self.X[col].median()
                    if pd.isna(median_val):
                        # If all are missing, use 0 as fallback
                        self.X[col] = self.X[col].fillna(0)
                    else:
                        # Fill with median
                        # Note: We don't add noise here to avoid introducing artificial variation
                        # Instead, we'll be careful with outlier capping to preserve natural variation
                        self.X[col] = self.X[col].fillna(median_val)
                        print(f"  - Filled {missing_pct:.1f}% missing values in '{col}' with median ({median_val:.2f})")
                else:
                    # For < 50% missing, use median (standard approach)
                    median_val = self.X[col].median()
                    if pd.isna(median_val):
                        self.X[col] = self.X[col].fillna(0)
                    else:
                        self.X[col] = self.X[col].fillna(median_val)
        
        # For categorical features, fill with mode (most common value) before encoding
        # This is done in the encoding section below
        
        # Remove rows with missing target values
        # We can't predict if we don't know what the actual value should be
        # ~ means "not" - so we keep rows that DON'T have missing target values
        mask = ~self.y.isnull()  # True for rows where target is not missing
        self.X = self.X[mask]    # Keep only rows where target is not missing
        self.y = self.y[mask]    # Keep only corresponding target values
        
        # Final check: Ensure target has variation after removing missing values
        # If still no variation, try to switch to Salary_Annual
        if self.y.nunique() <= 1 or self.y.std() == 0:
            print(f"\n⚠ Warning: Target variable '{self.target_column_name}' still has no variation after preprocessing!")
            print(f"   - Unique values: {self.y.nunique()}")
            print(f"   - Standard deviation: {self.y.std()}")
            
            # Try to use Salary_Annual as alternative target if available
            if 'Salary_Annual' in df_processed.columns and self.target_column_name != 'Salary_Annual':
                # Get the same rows that we kept for X
                salary_temp = df_processed.loc[mask, 'Salary_Annual'].dropna()
                if len(salary_temp) > 0 and salary_temp.nunique() > 1 and salary_temp.std() > 0:
                    print(f"\n✅ Switching to 'Salary_Annual' as target variable")
                    print(f"   - Salary_Annual has variation: std={salary_temp.std():.2f}, unique={salary_temp.nunique()}")
                    # Update both y and target column name
                    self.y = df_processed.loc[mask, 'Salary_Annual'].copy()
                    self.target_column_name = 'Salary_Annual'
                    # Also need to remove rows where Salary_Annual is missing
                    mask_salary = ~self.y.isnull()
                    self.X = self.X[mask_salary]
                    self.y = self.y[mask_salary]
                    print(f"   - Updated dataset: {len(self.X)} samples with Salary_Annual as target")
        
        print(f"\nBefore outlier handling: {len(self.X)} samples")
        
        # ====================================================================
        # OUTLIER HANDLING - Remove or cap extreme values
        # ====================================================================
        # Outliers are values that are very different from most of the data
        # They can skew the model and make it less accurate
        # We use the IQR (Interquartile Range) method to identify outliers
        
        if self.handle_outliers:
            initial_count = len(self.X)  # Remember how many samples we started with
            
            # ====================================================================
            # CALCULATE OUTLIER BOUNDARIES FOR TARGET VARIABLE
            # ====================================================================
            # IQR method: Values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR are outliers
            # Q1 = 25th percentile (25% of values are below this)
            # Q3 = 75th percentile (75% of values are below this)
            # IQR = Interquartile Range (middle 50% of data)
            
            Q1 = self.y.quantile(0.25)  # 25th percentile
            Q3 = self.y.quantile(0.75)  # 75th percentile
            IQR = Q3 - Q1                # Interquartile Range
            
            # Calculate boundaries for outliers
            # Values outside these bounds are considered extreme outliers
            lower_bound = Q1 - 1.5 * IQR  # Lower boundary
            upper_bound = Q3 + 1.5 * IQR  # Upper boundary
            
            # ====================================================================
            # CAP FEATURE OUTLIERS - Limit extreme values instead of removing
            # ====================================================================
            # For features: We "cap" outliers (set them to boundary) instead of removing
            # This preserves data while reducing the impact of extremes
            # Example: If max salary is $1,000,000 but boundary is $200,000, cap it at $200,000
            
            # Store original values before capping to check variation preservation
            X_before_capping = self.X.copy()
            
            for col in self.X.select_dtypes(include=[np.number]).columns:
                # Skip outlier capping if feature has no variation (all same values)
                # This prevents destroying features that were imputed with a single value
                unique_before = self.X[col].nunique()
                std_before = self.X[col].std()
                
                if unique_before <= 1 or std_before == 0:
                    continue  # Skip this feature - no variation to cap
                
                # Calculate IQR for this feature column
                Q1_feat = self.X[col].quantile(0.25)
                Q3_feat = self.X[col].quantile(0.75)
                IQR_feat = Q3_feat - Q1_feat
                
                # Skip if IQR is 0 (no variation)
                if IQR_feat == 0:
                    continue
                
                # Calculate boundaries
                # Use less aggressive capping (3.0*IQR instead of 1.5*IQR) to preserve more variation
                # This is especially important for features like Hours_Average that have been imputed
                lower_feat = Q1_feat - 3.0 * IQR_feat  # Very lenient lower bound
                upper_feat = Q3_feat + 3.0 * IQR_feat  # Very lenient upper bound
                
                # Ensure bounds are reasonable (not negative for positive-only features)
                if col == 'Hours_Average':
                    # Hours should be positive and reasonable (1-80 hours/week)
                    lower_feat = max(1.0, lower_feat)
                    upper_feat = min(80.0, upper_feat)
                
                # Clip values to boundaries (cap outliers)
                # Values below lower_feat become lower_feat
                # Values above upper_feat become upper_feat
                self.X[col] = self.X[col].clip(lower=lower_feat, upper=upper_feat)
                
                # Check if capping preserved variation
                unique_after = self.X[col].nunique()
                std_after = self.X[col].std()
                
                # If capping removed too much variation, use original values
                # This happens when most values were imputed with median
                if unique_after <= 1 and unique_before > 1:
                    # Restore original values - don't cap this feature
                    self.X[col] = X_before_capping[col]
                    print(f"  - Skipped outlier capping for '{col}' (would remove all variation)")
                elif std_after < std_before * 0.1:  # If std dropped by >90%
                    # Restore original - capping too aggressive
                    self.X[col] = X_before_capping[col]
                    print(f"  - Skipped outlier capping for '{col}' (too aggressive, would lose variation)")
            
            # ====================================================================
            # REMOVE TARGET OUTLIERS - Remove extreme target values completely
            # ====================================================================
            # For target variable: We're more strict - remove extreme outliers completely
            # Outliers in the target can really hurt model performance
            # Keep only rows where target is within acceptable bounds
            
            # Only remove outliers if target has variation (IQR > 0)
            if IQR > 0:
                # For Vacancy Count: use less aggressive outlier removal (keep more data)
                # Since it's heavily skewed, we want to preserve the tail distribution
                if self.target_column_name == 'Vacancy Count':
                    # Use 3*IQR instead of 1.5*IQR for Vacancy Count (less aggressive)
                    # This preserves more of the skewed distribution
                    lower_bound_vacancy = Q1 - 3.0 * IQR  # More lenient lower bound
                    upper_bound_vacancy = Q3 + 3.0 * IQR  # More lenient upper bound
                    mask = (self.y >= lower_bound_vacancy) & (self.y <= upper_bound_vacancy)
                    print(f"  - Using less aggressive outlier removal for Vacancy Count (3*IQR)")
                else:
                    # For other targets (like Salary_Annual), use standard 1.5*IQR
                    mask = (self.y >= lower_bound) & (self.y <= upper_bound)
                
                self.X = self.X[mask]
                self.y = self.y[mask]
                
                # Report how many samples were removed
                removed = initial_count - len(self.X)
                if removed > 0:
                    print(f"Outlier handling: Removed {removed} samples ({removed/initial_count*100:.2f}%)")
                else:
                    print(f"Outlier handling: No outliers removed (all values within bounds)")
            else:
                # If IQR is 0, target has no variation, so no outliers to remove
                print(f"Outlier handling: Skipped (target has no variation, IQR=0)")
        
        # ====================================================================
        # DISPLAY SUMMARY
        # ====================================================================
        print(f"\nSelected features: {self.original_feature_names}")
        print(f"\nFeature statistics:")
        print(self.X.describe())
        print(f"\nTarget variable statistics:")
        print(self.y.describe())
        
        # Display target variable information
        target_std = self.y.std()
        target_unique = self.y.nunique()
        print(f"\nTarget variable '{self.target_column_name}' analysis:")
        print(f"  - Unique values: {target_unique}")
        print(f"  - Standard deviation: {target_std:.4f}")
        if target_std == 0 or target_unique <= 1:
            print(f"  - ⚠ Warning: No variation in target - model will predict constant value")
        else:
            print(f"  - ✅ Target has variation - model can learn meaningful patterns")
        
        print(f"\nFinal dataset shape: X={self.X.shape}, y={self.y.shape}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        WHY SPLIT DATA?
        - We train the model on one part (training set - 80% typically)
        - We test the model on a different part (test set - 20% typically) that it hasn't seen
        - This tells us if the model can make good predictions on new data
        - Without splitting, we can't know if the model just memorized the training data
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset to use for testing (default: 0.2 = 20%)
            Remaining 80% is used for training
            Common values: 0.2 (20% test) or 0.3 (30% test)
        random_state : int
            Random seed for reproducibility
            Using the same number gives the same split every time
            This is important so results are consistent and reproducible
        
        Returns:
        --------
        self.X_train, self.X_test, self.y_train, self.y_test
            Training and testing splits of features and target
        """
        print("\n" + "=" * 60)
        print("SPLITTING DATA")
        print("=" * 60)
        
        # Split the data into training and testing sets
        # train_test_split automatically shuffles and splits the data
        # It returns 4 things: X_train, X_test, y_train, y_test
        # Shuffling ensures the split is random (not just first 80% vs last 20%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,              # Features to split
            self.y,              # Target to split
            test_size=test_size, # Percentage for testing (e.g., 0.2 = 20%)
            random_state=random_state  # Seed for random number generator (for reproducibility)
        )
        
        # Display the sizes of each split
        print(f"\nTraining set: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        print(f"  - Used to teach the model")
        print(f"  - Contains {self.X_train.shape[0]} samples")
        
        print(f"\nTesting set: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
        print(f"  - Used to evaluate the model")
        print(f"  - Contains {self.X_test.shape[0]} samples")
        print(f"  - Model hasn't seen this data during training")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Scale features and optionally add polynomial features
        
        WHY SCALE FEATURES?
        - Different features have different scales (e.g., salary is 30,000-100,000, 
          hours is 20-40, province codes are 0-12)
        - Models work better when all features are on similar scales
        - StandardScaler makes all features have mean=0 and std=1
        - This prevents features with larger numbers from dominating
        
        POLYNOMIAL FEATURES:
        - Creates new features by combining existing ones (e.g., salary * hours)
        - Can capture interactions between features
        - Example: A job might have high demand if it's BOTH high-salary AND full-time
        - This interaction wouldn't be captured by just salary + hours separately
        
        FEATURE SELECTION:
        - Not all features are equally important
        - We select the K best features to reduce complexity and improve performance
        """
        print("\n" + "=" * 60)
        print("SCALING FEATURES")
        print("=" * 60)
        
        # ====================================================================
        # STEP 1: SCALE FEATURES (Normalization)
        # ====================================================================
        # StandardScaler transforms features so they have:
        # - Mean = 0 (centered around zero)
        # - Standard deviation = 1 (consistent spread)
        # 
        # IMPORTANT: We fit on training data only, then transform both
        # This prevents "data leakage" - we can't use test data info during training
        
        # Ensure all columns are numeric before scaling
        # Convert any remaining non-numeric columns to numeric
        # This handles cases where some columns might still be strings or objects
        X_train_numeric = self.X_train.copy()
        X_test_numeric = self.X_test.copy()
        
        # Convert all columns to numeric, handling any non-numeric values
        for col in X_train_numeric.columns:
            # Check if column is not already numeric
            if X_train_numeric[col].dtype == 'object' or X_train_numeric[col].dtype.name == 'category':
                # Try to convert object/string columns to numeric
                # errors='coerce' means invalid values become NaN instead of raising error
                X_train_numeric[col] = pd.to_numeric(X_train_numeric[col], errors='coerce')
                X_test_numeric[col] = pd.to_numeric(X_test_numeric[col], errors='coerce')
            elif not pd.api.types.is_numeric_dtype(X_train_numeric[col]):
                # For any other non-numeric types, try to convert
                X_train_numeric[col] = pd.to_numeric(X_train_numeric[col], errors='coerce')
                X_test_numeric[col] = pd.to_numeric(X_test_numeric[col], errors='coerce')
        
        # Fill any NaN values created during conversion
        # Use median from training data for both train and test (prevents data leakage)
        for col in X_train_numeric.select_dtypes(include=[np.number]).columns:
            if X_train_numeric[col].isnull().any() or X_test_numeric[col].isnull().any():
                # Calculate median from training data only
                median_val = X_train_numeric[col].median()
                if pd.isna(median_val):
                    # If all values are NaN, fill with 0 as fallback
                    X_train_numeric[col] = X_train_numeric[col].fillna(0)
                    X_test_numeric[col] = X_test_numeric[col].fillna(0)
                else:
                    # Fill both train and test with training median (prevents data leakage)
                    X_train_numeric[col] = X_train_numeric[col].fillna(median_val)
                    X_test_numeric[col] = X_test_numeric[col].fillna(median_val)
        
        # Update self.X_train and self.X_test to be numeric versions
        # This ensures consistency throughout the rest of the pipeline
        self.X_train = X_train_numeric
        self.X_test = X_test_numeric
        
        # fit_transform: Learn the mean/std from training data AND transform it
        # This calculates: (value - mean) / std for each feature
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # transform: Apply the SAME transformation to test data
        # Uses the mean/std learned from training data (not test data!)
        # This is critical - we must use training statistics, not test statistics
        X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nFeatures scaled: Mean ≈ 0, Std ≈ 1 for all features")
        
        # ====================================================================
        # STEP 2: CREATE POLYNOMIAL FEATURES (Feature Engineering)
        # ====================================================================
        # Polynomial features capture interactions between features
        # Example: If we have salary and hours, we might create: salary * hours
        # This helps the model learn that some combinations matter
        
        if self.use_polynomial:
            print("\nAdding polynomial features (degree=2 with interactions only)...")
            
            # Create polynomial feature transformer
            # degree=2: Create features up to degree 2 (x, y, x*y)
            # include_bias=False: Don't add a constant term (already handled by model)
            # interaction_only=True: Only create interaction terms (x*y), not squares (x²)
            # This keeps the number of features manageable
            self.poly_features = PolynomialFeatures(
                degree=2,              # Maximum degree of polynomial features
                include_bias=False,    # Don't add constant term (model handles this)
                interaction_only=True  # Only interactions (x*y), not squares (x²)
            )
            
            # Fit on training data and transform both sets
            # fit_transform: Learn which combinations to create AND create them
            X_train_scaled = self.poly_features.fit_transform(X_train_scaled)
            
            # transform: Create the same combinations for test data
            # Must use same combinations learned from training data
            X_test_scaled = self.poly_features.transform(X_test_scaled)
            
            # Get names of the polynomial features (for understanding what was created)
            # Example: ['Salary_Annual', 'Hours_Minimum', 'Salary_Annual*Hours_Minimum', ...]
            poly_feature_names = self.poly_features.get_feature_names_out(self.original_feature_names)
            
            print(f"Polynomial features created: {X_train_scaled.shape[1]} features")
            print(f"  - Started with {len(self.original_feature_names)} original features")
            print(f"  - Created {X_train_scaled.shape[1] - len(self.original_feature_names)} interaction features")
        else:
            # If not using polynomial features, keep original names
            poly_feature_names = self.original_feature_names
        
        # ====================================================================
        # STEP 3: FEATURE SELECTION (Choose Best Features)
        # ====================================================================
        # With many features (especially after polynomial expansion), we want
        # to select only the most important ones. This:
        # - Reduces overfitting (model memorizing training data)
        # - Speeds up training
        # - Can improve accuracy by removing noise
        
        print("\nApplying feature selection...")
        
        # Select top K features (K = all features if < 50, otherwise 50)
        # For this dataset with 8 original features -> 36 polynomial features,
        # we can use all features since it's still manageable
        # This captures all feature interactions which is important for salary prediction
        k_best = min(50, X_train_scaled.shape[1])
        
        # If we have few features (< 50), use all of them
        # Feature selection is more important when we have hundreds of features
        if X_train_scaled.shape[1] <= 50:
            k_best = X_train_scaled.shape[1]  # Use all features
        
        # SelectKBest: Selects K features with highest scores
        # f_regression: Statistical test (F-test) that measures how well each feature
        #                predicts the target variable
        # Higher score = better predictor
        self.feature_selector = SelectKBest(
            score_func=f_regression,  # Scoring function (F-test for regression)
            k=k_best                  # Number of features to select
        )
        
        # Fit on training data: Learn which features are best
        # fit_transform: Select best features AND return only those features
        X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, self.y_train)
        
        # Transform test data: Keep only the same selected features
        # Must use same features selected from training data
        X_test_scaled = self.feature_selector.transform(X_test_scaled)
        
        # Get names of selected features
        # get_support() returns True/False for each feature (True = selected)
        selected_mask = self.feature_selector.get_support()
        self.feature_names = [
            poly_feature_names[i] 
            for i in range(len(poly_feature_names)) 
            if selected_mask[i]  # Keep only features where mask is True
        ]
        
        print(f"Selected {X_train_scaled.shape[1]} best features out of {len(poly_feature_names)}")
        
        # ====================================================================
        # STEP 4: CONVERT BACK TO DATAFRAME
        # ====================================================================
        # Convert numpy arrays back to pandas DataFrames
        # This makes it easier to work with and keeps feature names
        
        self.X_train_scaled = pd.DataFrame(
            X_train_scaled,              # Scaled data (numpy array)
            columns=self.feature_names,  # Feature names (list of strings)
            index=self.X_train.index      # Original row indices (for tracking)
        )
        
        self.X_test_scaled = pd.DataFrame(
            X_test_scaled,               # Scaled data (numpy array)
            columns=self.feature_names,  # Feature names (list of strings)
            index=self.X_test.index      # Original row indices (for tracking)
        )
        
        print("\nFeatures scaled and engineered successfully!")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def train_best_model(self, use_scaled=True):
        """
        Train multiple models and select the best one
        
        WHY TEST MULTIPLE MODELS?
        - Different models work better for different types of data
        - We test several models and pick the one that performs best
        - This is called "model selection"
        
        MODELS TESTED:
        1. Linear Regression: Simple, fast, good baseline
        2. Ridge Regression: Linear regression with regularization (prevents overfitting)
        3. Random Forest: Ensemble method using many decision trees
        4. Gradient Boosting: Sequential ensemble that learns from mistakes
        
        EVALUATION METRICS:
        - R² Score: How well model fits (0-1, higher is better)
        - RMSE: Root Mean Squared Error (lower is better, in same units as target)
        
        Parameters:
        -----------
        use_scaled : bool
            Whether to use scaled features (default: True)
            Scaled features usually work better
        
        Returns:
        --------
        self.model : Trained model object
            The best performing model
        """
        print("\n" + "=" * 60)
        print("TRAINING AND SELECTING BEST MODEL")
        print("=" * 60)
        
        # Choose which dataset to use (scaled or unscaled)
        # Must match what was used during preprocessing
        if use_scaled:
            X_train = self.X_train_scaled  # Use scaled features
            X_test = self.X_test_scaled
        else:
            X_train = self.X_train          # Use original features
            X_test = self.X_test
        
        # Storage for models and their results
        models = {}    # Dictionary to store all trained models
        results = []    # List to store performance metrics for each model
        
        # ====================================================================
        # MODEL 1: LINEAR REGRESSION
        # ====================================================================
        # Simplest model: Finds a line (or hyperplane) that best fits the data
        # Formula: y = a*x1 + b*x2 + c*x3 + ... + intercept
        # Pros: Fast, interpretable, good baseline
        # Cons: Assumes linear relationships (may miss complex patterns)
        
        print("\n1. Testing Linear Regression...")
        
        # Create the model
        model_lr = LinearRegression()
        
        # Train the model: Learn the coefficients (a, b, c, ...) from training data
        # fit() finds the best line that minimizes prediction error
        model_lr.fit(X_train, self.y_train)
        
        # Make predictions on test set
        y_pred = model_lr.predict(X_test)
        
        # Evaluate performance
        # R²: Proportion of variance explained (0-1, higher is better)
        r2 = r2_score(self.y_test, y_pred)
        # RMSE: Average prediction error in same units as target
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        # Store model and results
        models['Linear Regression'] = model_lr
        results.append({
            'model': 'Linear Regression', 
            'r2': r2, 
            'rmse': rmse, 
            'model_obj': model_lr
        })
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # ====================================================================
        # MODEL 2: RIDGE REGRESSION
        # ====================================================================
        # Linear regression with L2 regularization
        # Adds penalty for large coefficients to prevent overfitting
        # Overfitting = model memorizes training data but fails on new data
        # alpha parameter controls strength of regularization (higher = more penalty)
        
        print("\n2. Testing Ridge Regression...")
        
        # Define parameter values to test
        # GridSearchCV will try each value and pick the best
        # alpha controls regularization strength: higher = more penalty on large coefficients
        # Expanded search space for better hyperparameter tuning
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
        
        # Create base model
        ridge = Ridge()
        
        # GridSearchCV: Automatically tests different alpha values
        # cv=3: Uses 3-fold cross-validation to evaluate each alpha
        # scoring='r2': Optimizes for R² score (higher is better)
        # n_jobs=1: Use 1 CPU core (set to -1 to use all cores for faster execution)
        # verbose=0: Don't print progress messages
        grid_ridge = GridSearchCV(
            ridge,              # Base model to tune
            ridge_params,      # Parameters to test
            cv=3,              # 3-fold cross-validation
            scoring='r2',      # Metric to optimize
            n_jobs=1,          # Number of CPU cores to use
            verbose=0          # Suppress output
        )
        
        # Train and find best alpha
        grid_ridge.fit(X_train, self.y_train)
        
        # Make predictions with best model
        y_pred = grid_ridge.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        # Store best model (best_estimator_ is the model with best alpha)
        models['Ridge Regression'] = grid_ridge.best_estimator_
        results.append({
            'model': 'Ridge Regression', 
            'r2': r2, 
            'rmse': rmse, 
            'model_obj': grid_ridge.best_estimator_
        })
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}, Best alpha: {grid_ridge.best_params_['alpha']}")
        
        # ====================================================================
        # MODEL 3: RANDOM FOREST (WITH HYPERPARAMETER TUNING)
        # ====================================================================
        # Ensemble method: Combines many decision trees
        # Each tree makes a prediction, final prediction is average of all trees
        # Decision trees: Make predictions by asking yes/no questions about features
        # Pros: Handles non-linear relationships, feature importance, robust to outliers
        # Cons: Slower to train, less interpretable, can overfit with too many trees
        
        print("\n3. Testing Random Forest (with hyperparameter tuning)...")
        
        # First, train a quick baseline to get reasonable starting point
        print("   Training baseline model...")
        rf_baseline = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1  # Use all CPU cores for faster training
        )
        rf_baseline.fit(X_train, self.y_train)
        baseline_r2 = r2_score(self.y_test, rf_baseline.predict(X_test))
        print(f"   Baseline R²: {baseline_r2:.4f}")
        
        # Define optimized parameter grid for RandomizedSearchCV
        # Reduced parameter space and iterations for faster execution
        rf_params = {
            'n_estimators': [100, 150, 200],  # Reduced from [200, 300, 400, 500]
            'max_depth': [15, 20, 25],         # Reduced from [15, 20, 25, 30, None]
            'min_samples_split': [3, 5, 10],  # Reduced from [2, 3, 5, 10]
            'min_samples_leaf': [1, 2],       # Reduced from [1, 2, 4]
            'max_features': ['sqrt', 0.7]     # Reduced from ['sqrt', 'log2', 0.5, 0.7]
        }
        
        # Create base Random Forest model
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)  # Use all cores
        
        # Use RandomizedSearchCV for efficient hyperparameter tuning
        # n_iter=10: Reduced from 30 for faster execution (still finds good solutions)
        # cv=2: Reduced from 3 for faster execution (2-fold is faster, still reliable)
        # scoring='r2': Optimize for R² score
        # verbose=1: Show progress so user knows it's working
        rf_search = RandomizedSearchCV(
            rf_base,
            rf_params,
            n_iter=10,              # Reduced from 30 for faster execution
            cv=2,                   # Reduced from 3 for faster execution
            scoring='r2',           # Optimize for R² score
            n_jobs=1,               # Use 1 CPU core for search (parallelization handled by RF itself)
            random_state=42,        # For reproducibility
            verbose=1               # Show progress
        )
        
        # Train and find best hyperparameters
        print("   Tuning hyperparameters (this may take a few minutes)...")
        rf_search.fit(X_train, self.y_train)
        
        # Get best model
        rf = rf_search.best_estimator_
        
        # Evaluate
        y_pred = rf.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        models['Random Forest'] = rf
        results.append({
            'model': 'Random Forest', 
            'r2': r2, 
            'rmse': rmse, 
            'model_obj': rf
        })
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
        print(f"   Best params: {rf_search.best_params_}")
        
        # ====================================================================
        # MODEL 4: GRADIENT BOOSTING (WITH HYPERPARAMETER TUNING)
        # ====================================================================
        # Sequential ensemble: Builds trees one at a time
        # Each new tree tries to correct mistakes of previous trees
        # Pros: Often very accurate, handles complex patterns
        # Cons: Slower to train, more complex, can overfit
        
        print("\n4. Testing Gradient Boosting (with hyperparameter tuning)...")
        
        # First, train a quick baseline
        print("   Training baseline model...")
        gb_baseline = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        gb_baseline.fit(X_train, self.y_train)
        baseline_r2 = r2_score(self.y_test, gb_baseline.predict(X_test))
        print(f"   Baseline R²: {baseline_r2:.4f}")
        
        # Define optimized parameter grid for RandomizedSearchCV
        # Reduced parameter space for faster execution
        gb_params = {
            'n_estimators': [100, 150, 200],   # Reduced from [200, 300, 400]
            'learning_rate': [0.03, 0.05, 0.1],  # Reduced from [0.01, 0.03, 0.05, 0.1]
            'max_depth': [5, 6, 7],           # Reduced from [4, 5, 6, 7, 8]
            'min_samples_split': [3, 5, 10], # Reduced from [2, 3, 5, 10]
            'min_samples_leaf': [1, 2],       # Reduced from [1, 2, 4]
            'subsample': [0.8, 0.9],          # Reduced from [0.7, 0.8, 0.9, 1.0]
            'max_features': ['sqrt', 0.7]     # Reduced from ['sqrt', 'log2', 0.5, 0.7]
        }
        
        # Create base Gradient Boosting model
        gb_base = GradientBoostingRegressor(random_state=42)
        
        # Use RandomizedSearchCV for efficient hyperparameter tuning
        gb_search = RandomizedSearchCV(
            gb_base,
            gb_params,
            n_iter=10,              # Reduced from 30 for faster execution
            cv=2,                   # Reduced from 3 for faster execution
            scoring='r2',           # Optimize for R² score
            n_jobs=1,               # Use 1 CPU core
            random_state=42,        # For reproducibility
            verbose=1               # Show progress
        )
        
        # Train and find best hyperparameters
        print("   Tuning hyperparameters (this may take a few minutes)...")
        gb_search.fit(X_train, self.y_train)
        
        # Get best model
        gb = gb_search.best_estimator_
        
        # Evaluate
        y_pred = gb.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        models['Gradient Boosting'] = gb
        results.append({
            'model': 'Gradient Boosting', 
            'r2': r2, 
            'rmse': rmse, 
            'model_obj': gb
        })
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
        print(f"   Best params: {gb_search.best_params_}")
        
        # ====================================================================
        # MODEL 5: XGBOOST (IF AVAILABLE)
        # ====================================================================
        # XGBoost: Optimized gradient boosting library, often outperforms sklearn's GradientBoosting
        # Pros: Very fast, highly accurate, handles missing values well
        # Cons: Requires separate installation
        
        if XGBOOST_AVAILABLE:
            print("\n5. Testing XGBoost (with hyperparameter tuning)...")
            
            # First, train a quick baseline
            print("   Training baseline model...")
            xgb_baseline = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            xgb_baseline.fit(X_train, self.y_train)
            baseline_r2 = r2_score(self.y_test, xgb_baseline.predict(X_test))
            print(f"   Baseline R²: {baseline_r2:.4f}")
            
            # Define optimized parameter grid for XGBoost
            # Reduced parameter space for faster execution
            xgb_params = {
                'n_estimators': [100, 150, 200],  # Reduced from [200, 300, 400]
                'learning_rate': [0.03, 0.05, 0.1],  # Reduced from [0.01, 0.03, 0.05, 0.1]
                'max_depth': [5, 6, 7],           # Reduced from [4, 5, 6, 7, 8]
                'min_child_weight': [1, 2],       # Reduced from [1, 2, 3]
                'subsample': [0.8, 0.9],          # Reduced from [0.7, 0.8, 0.9]
                'colsample_bytree': [0.8, 0.9],   # Reduced from [0.7, 0.8, 0.9]
                'reg_alpha': [0, 0.1],            # Reduced from [0, 0.1, 0.5]
                'reg_lambda': [1, 1.5]            # Reduced from [1, 1.5, 2]
            }
            
            # Create base XGBoost model
            xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
            
            # Use RandomizedSearchCV for hyperparameter tuning
            xgb_search = RandomizedSearchCV(
                xgb_base,
                xgb_params,
                n_iter=10,          # Reduced from 30 for faster execution
                cv=2,               # Reduced from 3 for faster execution
                scoring='r2',
                n_jobs=1,
                random_state=42,
                verbose=1           # Show progress
            )
            
            # Train and find best hyperparameters
            print("   Tuning hyperparameters (this may take a few minutes)...")
            xgb_search.fit(X_train, self.y_train)
            
            # Get best model
            xgb_model = xgb_search.best_estimator_
            
            # Evaluate
            y_pred = xgb_model.predict(X_test)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            models['XGBoost'] = xgb_model
            results.append({
                'model': 'XGBoost', 
                'r2': r2, 
                'rmse': rmse, 
                'model_obj': xgb_model
            })
            print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
            print(f"   Best params: {xgb_search.best_params_}")
        else:
            print("\n5. XGBoost skipped (not installed)")
            print("   Install with: pip install xgboost")
        
        # ====================================================================
        # MODEL 6: LIGHTGBM (IF AVAILABLE)
        # ====================================================================
        # LightGBM: Fast gradient boosting framework, often outperforms XGBoost
        # Pros: Very fast, memory efficient, highly accurate
        # Cons: Requires separate installation
        
        if LIGHTGBM_AVAILABLE:
            print("\n6. Testing LightGBM (with hyperparameter tuning)...")
            
            # First, train a quick baseline
            print("   Training baseline model...")
            lgb_baseline = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            lgb_baseline.fit(X_train, self.y_train)
            baseline_r2 = r2_score(self.y_test, lgb_baseline.predict(X_test))
            print(f"   Baseline R²: {baseline_r2:.4f}")
            
            # Define optimized parameter grid for LightGBM
            # Reduced parameter space for faster execution
            lgb_params = {
                'n_estimators': [100, 150, 200],  # Reduced from [200, 300, 400]
                'learning_rate': [0.03, 0.05, 0.1],  # Reduced from [0.01, 0.03, 0.05, 0.1]
                'max_depth': [5, 6, 7],           # Reduced from [4, 5, 6, 7, 8, -1]
                'num_leaves': [31, 50, 70],       # Reduced from [31, 50, 70, 100]
                'min_child_samples': [20, 30],    # Reduced from [20, 30, 50]
                'subsample': [0.8, 0.9],         # Reduced from [0.7, 0.8, 0.9]
                'colsample_bytree': [0.8, 0.9],   # Reduced from [0.7, 0.8, 0.9]
                'reg_alpha': [0, 0.1],            # Reduced from [0, 0.1, 0.5]
                'reg_lambda': [0, 0.1]            # Reduced from [0, 0.1, 0.5]
            }
            
            # Create base LightGBM model
            lgb_base = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
            
            # Use RandomizedSearchCV for hyperparameter tuning
            lgb_search = RandomizedSearchCV(
                lgb_base,
                lgb_params,
                n_iter=10,          # Reduced from 30 for faster execution
                cv=2,               # Reduced from 3 for faster execution
                scoring='r2',
                n_jobs=1,
                random_state=42,
                verbose=1           # Show progress
            )
            
            # Train and find best hyperparameters
            print("   Tuning hyperparameters (this may take a few minutes)...")
            lgb_search.fit(X_train, self.y_train)
            
            # Get best model
            lgb_model = lgb_search.best_estimator_
            
            # Evaluate
            y_pred = lgb_model.predict(X_test)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            models['LightGBM'] = lgb_model
            results.append({
                'model': 'LightGBM', 
                'r2': r2, 
                'rmse': rmse, 
                'model_obj': lgb_model
            })
            print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
            print(f"   Best params: {lgb_search.best_params_}")
        else:
            print("\n6. LightGBM skipped (not installed)")
            print("   Install with: pip install lightgbm")
        
        # ====================================================================
        # MODEL 7: ENSEMBLE (VOTING REGRESSOR)
        # ====================================================================
        # Combine top 3 models by averaging their predictions
        # Often improves performance by leveraging strengths of different models
        
        print("\n7. Testing Ensemble (Voting Regressor)...")
        
        # Sort models by R² score and get top 3
        results_sorted = sorted(results, key=lambda x: x['r2'], reverse=True)
        top_3_models = results_sorted[:3]
        
        if len(top_3_models) >= 2:
            # Create list of (name, model) tuples for VotingRegressor
            estimators = [(result['model'], result['model_obj']) for result in top_3_models]
            
            # Create VotingRegressor that averages predictions
            ensemble = VotingRegressor(estimators=estimators)
            
            # Train ensemble
            ensemble.fit(X_train, self.y_train)
            
            # Evaluate
            y_pred = ensemble.predict(X_test)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            models['Ensemble (Voting)'] = ensemble
            results.append({
                'model': 'Ensemble (Voting)', 
                'r2': r2, 
                'rmse': rmse, 
                'model_obj': ensemble
            })
            print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
            print(f"   Combined models: {[m['model'] for m in top_3_models]}")
        else:
            print("   Skipped (need at least 2 models for ensemble)")
        
        # ====================================================================
        # SELECT BEST MODEL
        # ====================================================================
        # Compare all models and pick the one with highest R² score
        # R² is a good metric because it's normalized (0-1) and interpretable
        
        # Convert results to DataFrame for easy comparison
        results_df = pd.DataFrame(results)
        
        # Find model with maximum R² score
        # idxmax() returns index of row with maximum R²
        # loc[] gets that row
        best_result = results_df.loc[results_df['r2'].idxmax()]
        
        # Store the best model
        self.best_model_name = best_result['model']
        self.model = best_result['model_obj']
        
        # Display results
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"{'='*60}")
        print(f"R² Score: {best_result['r2']:.4f}")
        print(f"RMSE: {best_result['rmse']:.4f}")
        
        # ====================================================================
        # WARNING: Check for suspiciously perfect results
        # ====================================================================
        # R² > 0.99 is suspiciously high and might indicate:
        # 1. Data leakage (target in features)
        # 2. Overfitting (model memorized training data)
        # 3. Target has very little variation
        if best_result['r2'] > 0.99:
            print(f"\n⚠ WARNING: Suspiciously high R² score ({best_result['r2']:.4f})!")
            print(f"   - R² > 0.99 is unusually high for real-world data")
            print(f"   - Possible causes:")
            print(f"     1. Data leakage (target variable in features) - CHECK FEATURES!")
            print(f"     2. Overfitting (model memorized training data)")
            print(f"     3. Target has very little variation")
            print(f"   - Verify features don't include target variable")
            print(f"   - Check train/test performance gap for overfitting")
        
        # Show comparison of all models
        print(f"\nAll Models Comparison:")
        print(results_df[['model', 'r2', 'rmse']].to_string(index=False))
        
        return self.model
    
    def validate_model(self, use_scaled=True, cv_folds=5):
        """
        Validate the model using cross-validation and test set
        
        VALIDATION METHODS:
        1. Cross-Validation: Tests model multiple times on different data splits
           - More reliable than single test set
           - Shows how consistent the model is
           - Divides training data into K folds, trains on K-1, tests on 1, repeats K times
        2. Test Set Evaluation: Final check on unseen data
           - Simulates real-world performance
           - Should be similar to cross-validation results
        
        METRICS EXPLAINED:
        - R² Score: Proportion of variance explained (0-1, higher is better)
          - 1.0 = Perfect predictions
          - 0.0 = Model is as good as predicting the mean
          - Negative = Model is worse than predicting the mean
        - RMSE: Root Mean Squared Error (lower is better)
          - Average prediction error in same units as target
          - Penalizes large errors more than small ones
        - MAE: Mean Absolute Error (lower is better)
          - Average absolute difference between predictions and actual
          - Easier to interpret than RMSE
        
        Parameters:
        -----------
        use_scaled : bool
            Whether to use scaled features (should match what was used for training)
        cv_folds : int
            Number of folds for cross-validation (default: 5)
            More folds = more reliable but slower
        
        Returns:
        --------
        dict : Dictionary containing all performance metrics
        """
        print("\n" + "=" * 60)
        print("MODEL VALIDATION")
        print("=" * 60)
        
        # Choose which dataset to use (must match training)
        if use_scaled:
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
        else:
            X_train = self.X_train
            X_test = self.X_test
        
        # ====================================================================
        # CROSS-VALIDATION
        # ====================================================================
        # Cross-validation splits training data into K folds
        # For each fold:
        #   - Train on K-1 folds
        #   - Test on remaining fold
        # Average results across all folds
        # This gives a more reliable estimate of model performance
        
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        # Use fewer folds for ensemble models (they're slower to train)
        # Random Forest and Gradient Boosting take longer, so we use 3 folds instead of 5
        if 'Forest' in self.best_model_name or 'Boosting' in self.best_model_name:
            cv_folds_actual = 3  # Reduce to 3 folds for ensemble models
            print(f"Using {cv_folds_actual} folds for faster execution...")
        else:
            cv_folds_actual = cv_folds
        
        # Create KFold object: Defines how to split data
        # shuffle=True: Randomly shuffle data before splitting (ensures randomness)
        # random_state=42: For reproducibility (same seed = same splits)
        kfold = KFold(n_splits=cv_folds_actual, shuffle=True, random_state=42)
        
        # Cross-validation for RMSE
        # scoring='neg_mean_squared_error': Returns negative MSE (sklearn convention)
        # We negate it because sklearn maximizes scores, but we want to minimize error
        cv_scores = cross_val_score(
            self.model,           # Model to evaluate
            X_train,              # Training features
            self.y_train,         # Training target
            cv=kfold,             # Cross-validation strategy
            scoring='neg_mean_squared_error',  # Metric to optimize
            n_jobs=1              # Use 1 CPU core
        )
        # Convert to RMSE (take square root and negate)
        # Negative because sklearn returns negative MSE (for maximization)
        cv_rmse = np.sqrt(-cv_scores)
        
        # Cross-validation for R²
        # R² is already a maximization metric (higher is better)
        cv_r2 = cross_val_score(
            self.model, 
            X_train, 
            self.y_train, 
            cv=kfold, 
            scoring='r2',  # R² score (higher is better)
            n_jobs=1
        )
        
        # Display cross-validation results
        # Mean: Average performance across all folds
        # Std * 2: 95% confidence interval (shows consistency)
        # Lower std = more consistent model
        print(f"\nCross-Validation Results:")
        print(f"  RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        print(f"    - Lower is better")
        print(f"    - +/- shows variation across folds")
        print(f"  R² Score: {cv_r2.mean():.4f} (+/- {cv_r2.std() * 2:.4f})")
        print(f"    - Higher is better (max = 1.0)")
        print(f"    - +/- shows consistency")
        
        # Training set evaluation
        print(f"\nEvaluating on training set...")
        y_train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        print(f"\nTraining Set Performance:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  R² Score: {train_r2:.4f}")
        
        # Test set evaluation
        print(f"\nEvaluating on test set (unseen data)...")
        y_test_pred = self.model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  R² Score: {test_r2:.4f}")
        
        if train_r2 - test_r2 > 0.1:
            print(f"\n⚠ Warning: Large gap between train ({train_r2:.4f}) and test ({test_r2:.4f}) R²")
        
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred
        
        return {
            'cv_rmse': cv_rmse.mean(),
            'cv_r2': cv_r2.mean(),
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    
    def make_predictions(self, X_new=None, use_scaled=True):
        """
        Make predictions on new data
        
        This function can be used in two ways:
        1. Predict on test set (X_new=None): For evaluation
        2. Predict on new data (X_new provided): For real-world predictions
        
        IMPORTANT: When predicting on new data, we must apply the same
        preprocessing steps that were used during training:
        - Scaling (using same mean/std from training)
        - Polynomial features (if used)
        - Feature selection (keep only selected features)
        
        Parameters:
        -----------
        X_new : pd.DataFrame or None
            New data to make predictions on
            - If None: Uses test set (for evaluation)
            - If provided: Must have same columns as original features
        use_scaled : bool
            Whether to use scaled features (must match training)
        
        Returns:
        --------
        predictions : np.array
            Predicted values (e.g., vacancy counts or salaries)
        """
        print("\n" + "=" * 60)
        print("MAKING PREDICTIONS")
        print("=" * 60)
        
        if X_new is None:
            # ====================================================================
            # CASE 1: Predict on test set (for evaluation)
            # ====================================================================
            # Use the test set that was already prepared
            if use_scaled:
                X_pred = self.X_test_scaled
            else:
                X_pred = self.X_test
            
            # Use predictions that were already calculated during validation
            # This avoids recalculating (saves time)
            predictions = self.y_test_pred
            print("\nUsing test set for predictions")
            print("  - These predictions were already calculated during validation")
        else:
            # ====================================================================
            # CASE 2: Predict on new, unseen data
            # ====================================================================
            # This is what you'd use in production to predict for new job postings
            
            print(f"\nPreprocessing new data for prediction...")
            
            # Create a copy to avoid modifying the original
            X_new_processed = X_new.copy()
            
            # Apply the same preprocessing steps used during training
            if use_scaled:
                # STEP 1: Scale features using the same scaler from training
                # transform (not fit_transform!) - use existing mean/std
                # This is critical - we can't refit on new data
                # Must use the same statistics learned from training data
                X_new_scaled = self.scaler.transform(X_new_processed)
                
                # STEP 2: Apply polynomial features if they were used
                if self.use_polynomial:
                    X_new_scaled = self.poly_features.transform(X_new_scaled)
                    print("  - Applied polynomial feature transformations")
                
                # STEP 3: Apply feature selection (keep only selected features)
                # Must use the same features selected during training
                X_new_scaled = self.feature_selector.transform(X_new_scaled)
                print("  - Applied feature selection")
                
                # Convert back to DataFrame with feature names
                X_pred = pd.DataFrame(X_new_scaled, columns=self.feature_names)
            else:
                # If not using scaled features, use data as-is
                X_pred = X_new_processed
            
            # Make predictions using the trained model
            # This is where the magic happens - model uses learned patterns
            predictions = self.model.predict(X_pred)
            print(f"\nMade predictions on {len(predictions)} new samples")
        
        # ====================================================================
        # DISPLAY PREDICTION STATISTICS
        # ====================================================================
        # Show summary statistics of predictions
        print(f"\nPrediction statistics:")
        print(f"  Mean: {predictions.mean():.4f} (average predicted value)")
        print(f"  Std: {predictions.std():.4f} (variation in predictions)")
        print(f"  Min: {predictions.min():.4f} (lowest predicted value)")
        print(f"  Max: {predictions.max():.4f} (highest predicted value)")
        
        return predictions
    
    def visualize_results(self, save_plots=True):
        """
        Create visualizations for model results and labor market insights
        """
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Get feature importance
        # Handle different model types: individual models, ensembles, etc.
        if hasattr(self.model, 'feature_importances_'):
            # Direct feature importances (Random Forest, Gradient Boosting, XGBoost, LightGBM)
            importances = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
        elif hasattr(self.model, 'coef_'):
            # Coefficients (Linear Regression, Ridge, Lasso, etc.)
            importances = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False).head(15)
            importances['Importance'] = np.abs(importances['Coefficient'])
        elif isinstance(self.model, VotingRegressor):
            # For VotingRegressor, average feature importances from all base models
            all_importances = []
            # VotingRegressor stores estimators in 'estimators_' attribute (list of tuples)
            # or 'named_estimators_' (dict) depending on sklearn version
            estimators_dict = getattr(self.model, 'named_estimators_', {})
            if not estimators_dict:
                # Fallback: try 'estimators_' attribute (list of tuples)
                estimators_list = getattr(self.model, 'estimators_', [])
                if estimators_list:
                    estimators_dict = {f'model_{i}': est for i, est in enumerate(estimators_list)}
            
            for name, estimator in estimators_dict.items():
                if estimator is not None:
                    if hasattr(estimator, 'feature_importances_'):
                        all_importances.append(estimator.feature_importances_)
                    elif hasattr(estimator, 'coef_'):
                        coef = estimator.coef_
                        # Handle 1D and 2D coefficient arrays
                        if coef.ndim > 1:
                            coef = coef.flatten()
                        all_importances.append(np.abs(coef))
            
            if all_importances:
                # Average the importances from all models
                avg_importances = np.mean(all_importances, axis=0)
                # Ensure we have the right number of features
                if len(avg_importances) == len(self.feature_names):
                    importances = pd.DataFrame({
                        'Feature': self.feature_names,
                        'Importance': avg_importances
                    }).sort_values('Importance', ascending=False).head(15)
                else:
                    importances = None
            else:
                importances = None
        elif isinstance(self.model, StackingRegressor):
            # For StackingRegressor, use the meta-model's feature importances if available
            # Otherwise, average base models
            if hasattr(self.model.final_estimator_, 'feature_importances_'):
                importances = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': self.model.final_estimator_.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
            elif hasattr(self.model.final_estimator_, 'coef_'):
                importances = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Coefficient': self.model.final_estimator_.coef_
                }).sort_values('Coefficient', key=abs, ascending=False).head(15)
                importances['Importance'] = np.abs(importances['Coefficient'])
            else:
                # Fallback: average base models
                all_importances = []
                for name, estimator in self.model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        all_importances.append(estimator.feature_importances_)
                    elif hasattr(estimator, 'coef_'):
                        all_importances.append(np.abs(estimator.coef_))
                
                if all_importances:
                    avg_importances = np.mean(all_importances, axis=0)
                    importances = pd.DataFrame({
                        'Feature': self.feature_names,
                        'Importance': avg_importances
                    }).sort_values('Importance', ascending=False).head(15)
                else:
                    importances = None
        else:
            importances = None
        
        # 1. Feature importance
        ax1 = fig.add_subplot(gs[0, 0])
        if importances is not None:
            colors = ['#2E86AB' if x >= 0 else '#A23B72' 
                     for x in importances.get('Coefficient', importances['Importance'])]
            bars = ax1.barh(importances['Feature'], importances['Importance'], 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            ax1.set_xlabel('Importance / |Coefficient|', fontweight='bold')
            ax1.set_title('Top Feature Importances', fontweight='bold', fontsize=12)
            for i, (bar, val) in enumerate(zip(bars, importances['Importance'])):
                ax1.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left', 
                        fontweight='bold', fontsize=8)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_facecolor('#F8F9FA')
        
        # 2. Actual vs Predicted - Training
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(self.y_train, self.y_train_pred, alpha=0.6, s=30,
                             c=self.y_train, cmap='viridis', edgecolors='black', linewidth=0.5)
        ax2.plot([self.y_train.min(), self.y_train.max()],
                [self.y_train.min(), self.y_train.max()],
                'r--', lw=2.5, label='Perfect Prediction', zorder=3)
        ax2.set_xlabel('Actual', fontweight='bold')
        ax2.set_ylabel('Predicted', fontweight='bold')
        ax2.set_title('Training Set: Actual vs Predicted', fontweight='bold', fontsize=12)
        ax2.legend(loc='upper left', framealpha=0.9)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_facecolor('#F8F9FA')
        plt.colorbar(scatter, ax=ax2, label='Actual Value')
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        ax2.text(0.05, 0.95, f'R² = {train_r2:.4f}', transform=ax2.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontweight='bold')
        
        # 3. Actual vs Predicted - Test
        ax3 = fig.add_subplot(gs[0, 2])
        scatter = ax3.scatter(self.y_test, self.y_test_pred, alpha=0.6, s=30,
                             c=self.y_test, cmap='plasma', edgecolors='black', linewidth=0.5)
        ax3.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                'r--', lw=2.5, label='Perfect Prediction', zorder=3)
        ax3.set_xlabel('Actual', fontweight='bold')
        ax3.set_ylabel('Predicted', fontweight='bold')
        ax3.set_title('Test Set: Actual vs Predicted', fontweight='bold', fontsize=12)
        ax3.legend(loc='upper left', framealpha=0.9)
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.set_facecolor('#F8F9FA')
        plt.colorbar(scatter, ax=ax3, label='Actual Value')
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        ax3.text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=ax3.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontweight='bold')
        
        # 4-9. Additional plots (residuals, distributions, metrics)
        # Residuals - Training
        ax4 = fig.add_subplot(gs[1, 0])
        residuals_train = self.y_train - self.y_train_pred
        scatter4 = ax4.scatter(self.y_train_pred, residuals_train, alpha=0.6, s=30,
                              c=residuals_train, cmap='coolwarm', edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Zero Residual')
        ax4.set_xlabel('Predicted', fontweight='bold')
        ax4.set_ylabel('Residuals', fontweight='bold')
        ax4.set_title('Training Set: Residuals Plot', fontweight='bold', fontsize=12)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(alpha=0.3, linestyle='--')
        ax4.set_facecolor('#F8F9FA')
        plt.colorbar(scatter4, ax=ax4, label='Residual Value')
        
        # Residuals - Test
        ax5 = fig.add_subplot(gs[1, 1])
        residuals_test = self.y_test - self.y_test_pred
        scatter5 = ax5.scatter(self.y_test_pred, residuals_test, alpha=0.6, s=30,
                             c=residuals_test, cmap='coolwarm', edgecolors='black', linewidth=0.5)
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Zero Residual')
        ax5.set_xlabel('Predicted', fontweight='bold')
        ax5.set_ylabel('Residuals', fontweight='bold')
        ax5.set_title('Test Set: Residuals Plot', fontweight='bold', fontsize=12)
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(alpha=0.3, linestyle='--')
        ax5.set_facecolor('#F8F9FA')
        plt.colorbar(scatter5, ax=ax5, label='Residual Value')
        
        # Residual distribution
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(residuals_test, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax6.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Residual')
        ax6.set_xlabel('Residual Value', fontweight='bold')
        ax6.set_ylabel('Frequency', fontweight='bold')
        ax6.set_title('Test Set: Residual Distribution', fontweight='bold', fontsize=12)
        ax6.legend(loc='best', framealpha=0.9)
        ax6.grid(alpha=0.3, linestyle='--', axis='y')
        ax6.set_facecolor('#F8F9FA')
        mean_res = residuals_test.mean()
        std_res = residuals_test.std()
        ax6.text(0.05, 0.95, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}',
                transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontweight='bold')
        
        # Feature importance ranking
        ax7 = fig.add_subplot(gs[2, 0])
        if importances is not None:
            importances_sorted = importances.sort_values('Importance', ascending=True)
            colors_abs = ['#2E86AB' if x >= 0 else '#A23B72' 
                        for x in importances_sorted.get('Coefficient', importances_sorted['Importance'])]
            bars = ax7.barh(importances_sorted['Feature'], importances_sorted['Importance'],
                           color=colors_abs, alpha=0.8, edgecolor='black', linewidth=1.2)
            ax7.set_xlabel('Importance', fontweight='bold')
            ax7.set_title('Feature Importance Ranking', fontweight='bold', fontsize=12)
            for i, (bar, val) in enumerate(zip(bars, importances_sorted['Importance'])):
                ax7.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left',
                        fontweight='bold', fontsize=8)
        ax7.grid(axis='x', alpha=0.3, linestyle='--')
        ax7.set_facecolor('#F8F9FA')
        
        # Error distribution
        ax8 = fig.add_subplot(gs[2, 1])
        errors = np.abs(residuals_test)
        ax8.hist(errors, bins=50, color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax8.axvline(x=errors.mean(), color='r', linestyle='--', linewidth=2.5,
                   label=f'Mean Error: {errors.mean():.2f}')
        ax8.set_xlabel('Absolute Error', fontweight='bold')
        ax8.set_ylabel('Frequency', fontweight='bold')
        ax8.set_title('Test Set: Absolute Error Distribution', fontweight='bold', fontsize=12)
        ax8.legend(loc='best', framealpha=0.9)
        ax8.grid(alpha=0.3, linestyle='--', axis='y')
        ax8.set_facecolor('#F8F9FA')
        
        # Metrics summary
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        
        metrics_text = f"""
        MODEL PERFORMANCE METRICS
        
        Model: {self.best_model_name}
        
        Training Set:
        ──────────────────────
        R² Score:  {train_r2:.4f}
        RMSE:      {train_rmse:.4f}
        MAE:       {train_mae:.4f}
        
        Test Set:
        ──────────────────────
        R² Score:  {test_r2:.4f}
        RMSE:      {test_rmse:.4f}
        MAE:       {test_mae:.4f}
        
        Features:   {len(self.original_feature_names)} original
                    {len(self.feature_names)} engineered
        """
        
        ax9.text(0.1, 0.5, metrics_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=15),
                fontweight='bold')
        
        fig.suptitle('Canadian Job Market Analysis - Model Performance', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight', facecolor='white')
            print("\nPlot saved as 'linear_regression_results.png'")
        
        plt.show()
        
        # Separate feature importance plot
        if importances is not None:
            plt.figure(figsize=(12, 7))
            importances_display = importances.head(20).sort_values('Importance', ascending=True)
            colors_abs = ['#2E86AB' if x >= 0 else '#A23B72' 
                         for x in importances_display.get('Coefficient', importances_display['Importance'])]
            bars = plt.barh(importances_display['Feature'], importances_display['Importance'],
                           color=colors_abs, alpha=0.8, edgecolor='black', linewidth=1.5)
            plt.xlabel('Importance / |Coefficient|', fontweight='bold', fontsize=12)
            plt.ylabel('Feature', fontweight='bold', fontsize=12)
            plt.title('Feature Importance Analysis - Job Market Trends', fontweight='bold', fontsize=14, pad=20)
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.gca().set_facecolor('#F8F9FA')
            
            for i, (bar, val) in enumerate(zip(bars, importances_display['Importance'])):
                plt.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left',
                        fontweight='bold', fontsize=9)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2E86AB', alpha=0.8, label='Positive Impact'),
                Patch(facecolor='#A23B72', alpha=0.8, label='Negative Impact')
            ]
            plt.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=10)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
                print("Plot saved as 'feature_importance.png'")
            
            plt.show()
        
        print("\nVisualizations created successfully!")
    
    def run_full_pipeline(self, target_column='Vacancy Count', test_size=0.2,
                         random_state=42, use_scaled=True, cv_folds=3):
        """
        Run the complete modeling pipeline
        
        This method runs all steps of the machine learning pipeline in order:
        1. Load data from CSV
        2. Preprocess and engineer features
        3. Split into training and testing sets
        4. Scale features (if requested)
        5. Train multiple models and select the best
        6. Validate the model using cross-validation
        7. Make predictions
        8. Create visualizations
        
        This is a convenience method - you can also call each step individually
        for more control.
        
        Parameters:
        -----------
        target_column : str
            Name of the target variable (what we're predicting)
            Default: 'Vacancy Count' (predicting number of job vacancies)
            Alternative: 'Salary_Annual' (predicting annual salary)
        test_size : float
            Proportion of dataset for testing (default: 0.2 = 20%)
        random_state : int
            Random seed for reproducibility (default: 42)
        use_scaled : bool
            Whether to use scaled features (default: True, recommended)
        cv_folds : int
            Number of cross-validation folds (default: 3)
            More folds = more reliable but slower
        
        Returns:
        --------
        dict : Dictionary containing all performance metrics
        """
        print("\n" + "=" * 60)
        print("CANADIAN JOB MARKET ANALYSIS - COMPLETE PIPELINE")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Polynomial features: {self.use_polynomial}")
        print(f"  - Outlier handling: {self.handle_outliers}")
        
        # Step 1: Load data from CSV file
        self.load_data()
        
        # Step 2: Preprocess data (clean, engineer features, handle missing values)
        self.preprocess_data(target_column=target_column)
        
        # Step 3: Split data into training and testing sets
        self.split_data(test_size=test_size, random_state=random_state)
        
        # Step 4: Scale features (normalize to mean=0, std=1)
        if use_scaled:
            self.scale_features()
        
        # Step 5: Train multiple models and select the best one
        self.train_best_model(use_scaled=use_scaled)
        
        # Step 6: Validate model using cross-validation and test set
        metrics = self.validate_model(use_scaled=use_scaled, cv_folds=cv_folds)
        
        # Step 7: Make predictions on test set
        self.make_predictions(use_scaled=use_scaled)
        
        # Step 8: Create visualizations (plots and charts)
        self.visualize_results()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return metrics


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run the job market analysis
    
    This function demonstrates how to use the JobMarketAnalysis class
    to build a complete machine learning pipeline from start to finish.
    
    The pipeline includes:
    1. Loading data
    2. Preprocessing
    3. Feature engineering
    4. Model training and selection
    5. Model validation
    6. Making predictions
    7. Creating visualizations
    
    This is the entry point when running the script directly:
    python linear_regression_modeling.py
    """
    
    # ====================================================================
    # STEP 1: INITIALIZE THE MODEL
    # ====================================================================
    # Create an instance of the JobMarketAnalysis class
    # This sets up all the necessary components
    
    print("Initializing Canadian Job Market Analysis Model...")
    model = JobMarketAnalysis(
        csv_path='job-bank-open-data-all-job-postings-en-december2025.csv',  # Path to data file
        use_polynomial=True,           # Create polynomial features (improves accuracy)
        handle_outliers=True           # Remove/cap extreme values (improves robustness)
    )
    
    # ====================================================================
    # STEP 2: RUN THE COMPLETE PIPELINE
    # ====================================================================
    # This single function call runs everything:
    # - Loads and preprocesses data
    # - Splits into train/test
    # - Scales features
    # - Trains multiple models and picks best
    # - Validates the model
    # - Makes predictions
    # - Creates visualizations
    # - Returns performance metrics
    
    print("\nRunning complete machine learning pipeline...")
    
    # Start with Salary_Annual as target since Vacancy Count has no variation in this dataset
    # The preprocessing will auto-switch if needed, but starting with Salary_Annual gives better results
    # Salary_Annual has meaningful variation and provides useful insights about compensation trends
    metrics = model.run_full_pipeline(
        target_column='Salary_Annual',  # Predict annual salary (has variation, provides meaningful insights)
        test_size=0.2,                    # 20% for testing, 80% for training
        random_state=42,                  # For reproducibility
        use_scaled=True,                  # Use scaled features (recommended)
        cv_folds=3                       # 3-fold cross-validation (faster than 5)
    )
    
    # ====================================================================
    # STEP 3: DISPLAY FINAL SUMMARY
    # ====================================================================
    # Show the most important results in a clean format
    
    print("\n" + "=" * 60)
    print("FINAL MODEL SUMMARY")
    print("=" * 60)
    print(f"\nBest Model: {model.best_model_name}")
    print(f"  - This model performed best among all tested models")
    
    print(f"\nTest Set Performance (on unseen data):")
    print(f"  R² Score: {metrics['test_r2']:.4f}")
    print(f"    - Closer to 1.0 is better")
    print(f"    - Shows how well model explains variance")
    print(f"  RMSE: {metrics['test_rmse']:.4f}")
    print(f"    - Lower is better")
    print(f"    - Average prediction error in target units")
    print(f"  MAE: {metrics['test_mae']:.4f}")
    print(f"    - Lower is better")
    print(f"    - Average absolute error (easier to interpret)")
    
    print(f"\nCross-Validation Performance:")
    print(f"  R² Score: {metrics['cv_r2']:.4f}")
    print(f"    - Average across multiple validation folds")
    print(f"    - More reliable than single test set")
    print(f"  RMSE: {metrics['cv_rmse']:.4f}")
    print(f"    - Average RMSE across validation folds")
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"   Check the generated plots for detailed analysis.")


if __name__ == "__main__":
    main()
