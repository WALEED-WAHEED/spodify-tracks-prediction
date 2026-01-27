"""
Linear Regression Modeling, Model Validation, and Predictions
for Spotify Tracks Prediction

This script builds a machine learning model to predict Spotify track popularity
based on audio features and metadata.

Features used:
- Audio Features: Danceability, Energy, Valence, Tempo, Loudness, Duration_min
- Metadata features: explicit

Advanced improvements:
- Feature engineering (polynomial features, interactions)
- Outlier handling
- Multiple model types with hyperparameter tuning
- Feature selection
"""

# ============================================================================
# IMPORT STATEMENTS - Loading necessary libraries
# ============================================================================

# pandas: Used for reading CSV files and working with data tables (DataFrames)
# Think of it as Excel for Python - helps organize and manipulate data
import pandas as pd

# numpy: Used for mathematical operations and working with arrays
# Provides functions for calculations like square root, mean, etc.
import numpy as np

# sklearn.model_selection: Tools for splitting data and evaluating models
# - train_test_split: Divides data into training and testing sets
# - cross_val_score: Evaluates model using cross-validation (testing multiple times)
# - KFold: Creates folds for cross-validation
# - GridSearchCV: Automatically finds best model parameters
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

# sklearn.linear_model: Different types of linear regression models
# - LinearRegression: Basic linear regression (finds a straight line that fits data)
# - Ridge: Linear regression with L2 regularization (prevents overfitting)
# - Lasso: Linear regression with L1 regularization (can remove unimportant features)
# - ElasticNet: Combines Ridge and Lasso regularization
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# sklearn.ensemble: Advanced models that combine multiple simpler models
# - RandomForestRegressor: Uses many decision trees and averages their predictions
# - GradientBoostingRegressor: Builds models sequentially, each improving on previous errors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# sklearn.metrics: Functions to measure how good our model predictions are
# - mean_squared_error: Average of squared differences between predictions and actual values
# - mean_absolute_error: Average of absolute differences (easier to understand)
# - r2_score: R-squared score (0-1, higher is better, shows how well model fits)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# sklearn.preprocessing: Tools to prepare data before training
# - StandardScaler: Normalizes features (makes them have mean=0, std=1)
# - PolynomialFeatures: Creates new features by combining existing ones (e.g., x*y)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# sklearn.feature_selection: Tools to select the most important features
# - SelectKBest: Selects the K best features based on a scoring function
# - f_regression: Statistical test to measure feature importance
from sklearn.feature_selection import SelectKBest, f_regression

# matplotlib.pyplot: Library for creating plots and graphs
import matplotlib.pyplot as plt

# seaborn: Makes matplotlib plots look nicer and easier to create
import seaborn as sns

# warnings: Python's warning system
# We ignore warnings to keep output clean (warnings don't stop code from running)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VISUALIZATION SETTINGS - Making plots look professional
# ============================================================================

# Set the style of plots to have a white background with grid lines
# This makes plots easier to read
sns.set_style("whitegrid")

# Set default figure size to 14 inches wide by 10 inches tall
# Larger figures show more detail
plt.rcParams['figure.figsize'] = (14, 10)

# Set default font sizes for different plot elements
# This ensures text is readable
plt.rcParams['font.size'] = 10              # General font size
plt.rcParams['axes.labelsize'] = 11          # Size of axis labels (x-axis, y-axis)
plt.rcParams['axes.titlesize'] = 13          # Size of plot titles
plt.rcParams['xtick.labelsize'] = 9          # Size of numbers on x-axis
plt.rcParams['ytick.labelsize'] = 9          # Size of numbers on y-axis
plt.rcParams['legend.fontsize'] = 10         # Size of legend text

# ============================================================================
# CLASS DEFINITION - SpotifyLinearRegression
# ============================================================================
# This class encapsulates all the functionality needed to build and use
# a machine learning model for predicting Spotify track popularity.
# Think of a class as a blueprint for creating objects that have both
# data (attributes) and functions (methods) that work together.
# ============================================================================

class SpotifyLinearRegression:
    """
    Advanced Linear Regression Model for Spotify Tracks Prediction
    
    This class provides a complete machine learning pipeline:
    1. Loads and preprocesses data
    2. Trains multiple models and selects the best one
    3. Validates the model using cross-validation
    4. Makes predictions on new data
    5. Creates visualizations of results
    """
    
    def __init__(self, csv_path='dataset.csv', use_polynomial=True, handle_outliers=True):
        """
        Initialize the model - This is called when you create a new instance
        
        The __init__ method sets up all the variables (attributes) that the
        class will use throughout its lifetime. These are like empty containers
        that will be filled with data as we process it.
        
        Parameters:
        -----------
        csv_path : str
            Path to the dataset CSV file (the file containing our Spotify data)
        use_polynomial : bool
            Whether to create polynomial features (combinations like x*y)
            True = Create polynomial features (can improve accuracy but slower)
            False = Use only original features (faster but potentially less accurate)
        handle_outliers : bool
            Whether to remove or cap extreme values in the data
            True = Handle outliers (recommended for better model performance)
            False = Keep all data as-is (may include problematic extreme values)
        """
        # Store the path to our data file
        self.csv_path = csv_path
        
        # Store configuration options
        self.use_polynomial = use_polynomial      # Whether to use polynomial features
        self.handle_outliers = handle_outliers    # Whether to handle outliers
        
        # Data storage variables (will be filled later)
        # None means "empty" - we'll fill these as we process the data
        self.df = None                    # The full dataset loaded from CSV
        self.X = None                     # Features (input variables) - what we use to predict
        self.y = None                     # Target (output variable) - what we're trying to predict (popularity)
        
        # Training and testing data splits
        # We split data so we can train on one part and test on another
        self.X_train = None               # Features for training (used to teach the model)
        self.X_test = None                # Features for testing (used to evaluate the model)
        self.y_train = None               # Target values for training
        self.y_test = None                # Target values for testing
        
        # Model storage
        self.model = None                 # The trained machine learning model
        self.best_model_name = None       # Name of the best performing model
        
        # Preprocessing tools (will be fitted on training data)
        self.scaler = StandardScaler()    # Tool to normalize features (make them comparable)
        self.poly_features = None         # Tool to create polynomial features
        self.feature_selector = None      # Tool to select best features
        
        # Feature name tracking
        self.feature_names = None         # Names of final features used in model
        self.original_feature_names = None  # Names of original features before engineering
        
    def load_data(self):
        """
        Load and inspect the dataset
        
        This function reads the CSV file and displays information about the data.
        It's important to inspect data before processing to understand what we're working with.
        
        Returns:
        --------
        self.df : pandas.DataFrame
            The loaded dataset
        """
        # Print a header to make output easier to read
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        # Read the CSV file into a pandas DataFrame
        # A DataFrame is like a spreadsheet - rows are samples, columns are features
        self.df = pd.read_csv(self.csv_path)
        
        # Display basic information about the dataset
        # .shape returns (number_of_rows, number_of_columns)
        print(f"\nDataset shape: {self.df.shape}")
        print(f"  - Rows (samples): {self.df.shape[0]}")
        print(f"  - Columns (features): {self.df.shape[1]}")
        
        # Show all column names - helps us see what features are available
        print(f"\nColumns: {list(self.df.columns)}")
        
        # Show first 5 rows - gives us a preview of the data
        # This helps us understand what the data looks like
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        # Show data types - tells us if columns are numbers, text, etc.
        # Important because different types need different processing
        print(f"\nData types:")
        print(self.df.dtypes)
        
        # Check for missing values - empty cells in our data
        # Missing values can cause problems, so we need to handle them
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        # Get comprehensive information about the dataset
        # Shows memory usage, data types, and non-null counts
        print(f"\nDataset info:")
        print(self.df.info())
        
        # Return the loaded DataFrame
        return self.df
    
    def preprocess_data(self, target_column='popularity'):
        """
        Preprocess the data and prepare features with advanced techniques
        
        Preprocessing is crucial in machine learning - it prepares raw data
        so the model can learn effectively. This function:
        1. Selects relevant features
        2. Converts data types
        3. Handles missing values
        4. Removes or caps outliers (extreme values)
        
        Parameters:
        -----------
        target_column : str
            Name of the target variable (what we're trying to predict)
            Default is 'popularity' - the Spotify popularity score (0-100)
        
        Returns:
        --------
        self.X : pandas.DataFrame
            Processed features (input variables)
        self.y : pandas.Series
            Processed target variable (output variable)
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        
        # Create a copy of the original data
        # We do this to avoid accidentally modifying the original dataset
        # If we modify the original, we can't reload it easily
        df_processed = self.df.copy()
        
        # Convert duration from milliseconds to minutes
        # Duration is stored as milliseconds (e.g., 210000 ms = 3.5 minutes)
        # Converting to minutes makes it easier to understand and work with
        # Formula: minutes = milliseconds / (1000 ms/sec * 60 sec/min) = milliseconds / 60000
        df_processed['Duration_min'] = df_processed['duration_ms'] / 60000.0
        
        # ====================================================================
        # FEATURE SELECTION - Choosing which columns to use for prediction
        # ====================================================================
        # We select specific features that are relevant for predicting popularity
        
        # Audio features: Characteristics of the music itself
        audio_features = [
            'danceability',    # How suitable for dancing (0.0-1.0)
            'energy',          # Perceived intensity/power (0.0-1.0)
            'valence',         # Musical positiveness (0.0-1.0, higher = happier)
            'tempo',           # Speed of the track (beats per minute)
            'loudness',        # Overall loudness in decibels (usually negative)
            'Duration_min'     # Length of track in minutes
        ]
        
        # Metadata features: Information about the track
        metadata_features = [
            'explicit'         # Whether track contains explicit content (True/False or 0/1)
        ]
        
        # Combine all features into one list
        feature_columns = audio_features + metadata_features
        
        # ====================================================================
        # VALIDATION - Check if all required features exist in the dataset
        # ====================================================================
        # This prevents errors later if a column name is misspelled or missing
        missing_features = [f for f in feature_columns if f not in df_processed.columns]
        if missing_features:
            # Raise an error if any features are missing
            # This stops execution and shows what's wrong
            raise ValueError(f"Missing features: {missing_features}")
        
        # ====================================================================
        # SEPARATE FEATURES (X) AND TARGET (y)
        # ====================================================================
        # X = Input features (what we use to make predictions)
        # y = Target variable (what we're trying to predict)
        # This is standard machine learning notation
        self.X = df_processed[feature_columns].copy()  # Features
        self.y = df_processed[target_column].copy()    # Target (popularity)
        
        # ====================================================================
        # DATA TYPE CONVERSION - Ensure 'explicit' is numeric (0 or 1)
        # ====================================================================
        # Machine learning models need numbers, not True/False or text
        # Convert boolean or text values to 0 (False) or 1 (True)
        
        if self.X['explicit'].dtype == 'bool':
            # If it's already boolean (True/False), convert to integer (1/0)
            self.X['explicit'] = self.X['explicit'].astype(int)
        elif self.X['explicit'].dtype == 'object':
            # If it's text (like "True"/"False" strings), convert to integer
            # This checks if the string equals "True" and converts to 1 or 0
            self.X['explicit'] = (self.X['explicit'] == 'True').astype(int)
        
        # Store the original feature names before any transformations
        # We'll need these later for creating polynomial feature names
        self.original_feature_names = list(self.X.columns)
        
        # ====================================================================
        # HANDLE MISSING VALUES - Remove rows with empty cells
        # ====================================================================
        # Missing values (NaN) cause problems in machine learning
        # We create a "mask" (True/False array) to identify rows without missing values
        # ~ means "not" - so we keep rows that DON'T have missing values
        # .isnull().any(axis=1) checks if ANY column in a row has missing values
        mask = ~(self.X.isnull().any(axis=1) | self.y.isnull())
        
        # Apply the mask to keep only rows without missing values
        self.X = self.X[mask]
        self.y = self.y[mask]
        
        print(f"\nBefore outlier handling: {len(self.X)} samples")
        
        # ====================================================================
        # OUTLIER HANDLING - Remove or cap extreme values
        # ====================================================================
        # Outliers are values that are very different from most of the data
        # They can skew the model and make it less accurate
        # We use the IQR (Interquartile Range) method to identify outliers
        
        if self.handle_outliers:
            # Remember how many samples we started with
            initial_count = len(self.X)
            
            # Calculate IQR for the target variable (popularity)
            # IQR method: Values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR are outliers
            Q1 = self.y.quantile(0.25)  # 25th percentile (25% of values are below this)
            Q3 = self.y.quantile(0.75)  # 75th percentile (75% of values are below this)
            IQR = Q3 - Q1                # Interquartile Range (middle 50% of data)
            
            # Calculate boundaries for outliers
            # Values outside these bounds are considered extreme outliers
            lower_bound = Q1 - 1.5 * IQR  # Lower boundary
            upper_bound = Q3 + 1.5 * IQR  # Upper boundary
            
            # For features: Cap outliers instead of removing them
            # "Capping" means if a value is too high/low, set it to the boundary
            # This preserves data while reducing the impact of extremes
            for col in self.X.columns:
                # Calculate IQR for this feature column
                Q1_feat = self.X[col].quantile(0.25)
                Q3_feat = self.X[col].quantile(0.75)
                IQR_feat = Q3_feat - Q1_feat
                
                # Calculate boundaries
                lower_feat = Q1_feat - 1.5 * IQR_feat
                upper_feat = Q3_feat + 1.5 * IQR_feat
                
                # Clip values to boundaries (cap outliers)
                # Values below lower_feat become lower_feat
                # Values above upper_feat become upper_feat
                self.X[col] = self.X[col].clip(lower=lower_feat, upper=upper_feat)
            
            # For target variable: Remove extreme outliers completely
            # We're more strict with the target because outliers here can really hurt the model
            mask = (self.y >= lower_bound) & (self.y <= upper_bound)
            self.X = self.X[mask]
            self.y = self.y[mask]
            
            # Report how many samples were removed
            removed = initial_count - len(self.X)
            if removed > 0:
                print(f"Outlier handling: Removed {removed} samples ({removed/initial_count*100:.2f}%)")
        
        # ====================================================================
        # DISPLAY SUMMARY STATISTICS
        # ====================================================================
        # Show information about the processed data
        print(f"\nSelected features: {self.original_feature_names}")
        
        # .describe() shows statistics like mean, std, min, max for each feature
        print(f"\nFeature statistics:")
        print(self.X.describe())
        
        # Show statistics for the target variable
        print(f"\nTarget variable statistics:")
        print(self.y.describe())
        
        # Final dataset size
        print(f"\nFinal dataset shape: X={self.X.shape}, y={self.y.shape}")
        print(f"  - X shape: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"  - y shape: {self.y.shape[0]} target values")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        WHY SPLIT DATA?
        - We train the model on one part (training set)
        - We test the model on a different part (test set) that it hasn't seen
        - This tells us if the model can make good predictions on new data
        - Without splitting, we can't know if the model just memorized the data
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset to use for testing (default: 0.2 = 20%)
            Remaining 80% is used for training
            Common values: 0.2 (20% test) or 0.3 (30% test)
        random_state : int
            Random seed for reproducibility
            Using the same number gives the same split every time
            This is important so results are consistent
        
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,              # Features to split
            self.y,              # Target to split
            test_size=test_size, # Percentage for testing (e.g., 0.2 = 20%)
            random_state=random_state  # Seed for random number generator
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
        Scale features using StandardScaler and optionally add polynomial features
        
        WHY SCALE FEATURES?
        - Different features have different scales (e.g., tempo is 60-200, 
          danceability is 0-1)
        - Models work better when all features are on similar scales
        - StandardScaler makes all features have mean=0 and std=1
        - This prevents features with larger numbers from dominating
        
        POLYNOMIAL FEATURES:
        - Creates new features by combining existing ones (e.g., danceability * energy)
        - Can capture interactions between features
        - Example: A song might be popular if it's BOTH danceable AND energetic
        - This interaction wouldn't be captured by just danceability + energy separately
        
        FEATURE SELECTION:
        - Not all features are equally important
        - We select the K best features to reduce complexity and improve performance
        
        Returns:
        --------
        self.X_train_scaled, self.X_test_scaled
            Scaled and engineered training and testing features
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
        
        # fit_transform: Learn the mean/std from training data AND transform it
        # This calculates: (value - mean) / std for each feature
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # transform: Apply the SAME transformation to test data
        # Uses the mean/std learned from training data (not test data!)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nFeatures scaled: Mean ≈ 0, Std ≈ 1 for all features")
        
        # ====================================================================
        # STEP 2: CREATE POLYNOMIAL FEATURES (Feature Engineering)
        # ====================================================================
        # Polynomial features capture interactions between features
        # Example: If we have danceability and energy, we might create:
        # - danceability * energy (interaction term)
        # This helps the model learn that some combinations matter
        
        if self.use_polynomial:
            print("\nAdding polynomial features (degree=2 with interactions only)...")
            
            # Create polynomial feature transformer
            # degree=2: Create features up to degree 2 (x, y, x*y)
            # include_bias=False: Don't add a constant term (already handled by model)
            # interaction_only=True: Only create interaction terms (x*y), not squares (x²)
            self.poly_features = PolynomialFeatures(
                degree=2, 
                include_bias=False, 
                interaction_only=True
            )
            
            # Fit on training data and transform both sets
            # fit_transform: Learn which combinations to create AND create them
            X_train_scaled = self.poly_features.fit_transform(X_train_scaled)
            
            # transform: Create the same combinations for test data
            X_test_scaled = self.poly_features.transform(X_test_scaled)
            
            # Get names of the polynomial features (for understanding what was created)
            # Example: ['danceability', 'energy', 'danceability*energy', ...]
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
        
        # Select top K features (K = 30, or all if we have fewer than 30)
        # We limit to 30 to keep the model manageable
        k_best = min(30, X_train_scaled.shape[1])
        
        # SelectKBest: Selects K features with highest scores
        # f_regression: Statistical test that measures how well each feature
        #                predicts the target variable
        # Higher score = better predictor
        self.feature_selector = SelectKBest(
            score_func=f_regression,  # Scoring function
            k=k_best                  # Number of features to select
        )
        
        # Fit on training data: Learn which features are best
        # fit_transform: Select best features AND return only those features
        X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, self.y_train)
        
        # Transform test data: Keep only the same selected features
        X_test_scaled = self.feature_selector.transform(X_test_scaled)
        
        # Get names of selected features
        # get_support() returns True/False for each feature (True = selected)
        selected_mask = self.feature_selector.get_support()
        self.feature_names = [
            poly_feature_names[i] 
            for i in range(len(poly_feature_names)) 
            if selected_mask[i]
        ]
        
        print(f"Selected {X_train_scaled.shape[1]} best features out of {len(poly_feature_names)}")
        
        # ====================================================================
        # STEP 4: CONVERT BACK TO DATAFRAME
        # ====================================================================
        # Convert numpy arrays back to pandas DataFrames
        # This makes it easier to work with and keeps feature names
        
        self.X_train_scaled = pd.DataFrame(
            X_train_scaled,              # Scaled data
            columns=self.feature_names,  # Feature names
            index=self.X_train.index      # Original row indices
        )
        
        self.X_test_scaled = pd.DataFrame(
            X_test_scaled,               # Scaled data
            columns=self.feature_names,  # Feature names
            index=self.X_test.index      # Original row indices
        )
        
        print("\nFeatures scaled and engineered successfully!")
        print(f"\nScaled training features statistics:")
        print(self.X_train_scaled.describe())
        print(f"\nNote: All features should now have mean ≈ 0 and std ≈ 1")
        
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
        3. Lasso Regression: Linear regression that can remove unimportant features
        4. Random Forest: Ensemble method using many decision trees
        5. Gradient Boosting: Sequential ensemble that learns from mistakes
        
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
        if use_scaled:
            X_train = self.X_train_scaled  # Use scaled features
            X_test = self.X_test_scaled
        else:
            X_train = self.X_train          # Use original features
            X_test = self.X_test
        
        # Storage for models and their results
        models = {}    # Dictionary to store all trained models
        results = []   # List to store performance metrics for each model
        
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
        ridge_params = {'alpha': [1.0, 10.0, 100.0]}
        
        # Create base model
        ridge = Ridge()
        
        # GridSearchCV: Automatically tests different alpha values
        # cv=3: Uses 3-fold cross-validation to evaluate each alpha
        # scoring='r2': Optimizes for R² score
        # n_jobs=1: Use 1 CPU core (set to -1 to use all cores)
        grid_ridge = GridSearchCV(
            ridge, 
            ridge_params, 
            cv=3, 
            scoring='r2', 
            n_jobs=1, 
            verbose=0
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
        # MODEL 3: LASSO REGRESSION
        # ====================================================================
        # Linear regression with L1 regularization
        # Can set coefficients to exactly zero (removes features)
        # Useful for feature selection - automatically removes unimportant features
        # alpha parameter controls regularization strength
        
        print("\n3. Testing Lasso Regression...")
        
        # Define parameter values to test
        lasso_params = {'alpha': [1.0, 10.0, 100.0]}
        
        # Create model with increased max_iter for convergence
        # Lasso sometimes needs more iterations to find optimal solution
        lasso = Lasso(max_iter=2000)
        
        # Find best alpha using grid search
        grid_lasso = GridSearchCV(
            lasso, 
            lasso_params, 
            cv=3, 
            scoring='r2', 
            n_jobs=1, 
            verbose=0
        )
        grid_lasso.fit(X_train, self.y_train)
        
        # Evaluate best model
        y_pred = grid_lasso.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        models['Lasso Regression'] = grid_lasso.best_estimator_
        results.append({
            'model': 'Lasso Regression', 
            'r2': r2, 
            'rmse': rmse, 
            'model_obj': grid_lasso.best_estimator_
        })
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}, Best alpha: {grid_lasso.best_params_['alpha']}")
        
        # ====================================================================
        # MODEL 4: RANDOM FOREST
        # ====================================================================
        # Ensemble method: Combines many decision trees
        # Each tree makes a prediction, final prediction is average of all trees
        # Decision trees: Make predictions by asking yes/no questions about features
        # Pros: Handles non-linear relationships, feature importance, robust
        # Cons: Slower, less interpretable, can overfit
        
        print("\n4. Testing Random Forest...")
        
        # Create Random Forest model
        rf = RandomForestRegressor(
            n_estimators=50,      # Number of trees (more = better but slower)
            max_depth=12,          # Maximum depth of each tree (prevents overfitting)
            min_samples_split=10,  # Minimum samples needed to split a node
            min_samples_leaf=4,    # Minimum samples in a leaf node
            random_state=42,       # For reproducibility
            n_jobs=1              # Use 1 CPU core
        )
        
        # Train the forest
        rf.fit(X_train, self.y_train)
        
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
        
        # ====================================================================
        # MODEL 5: GRADIENT BOOSTING
        # ====================================================================
        # Sequential ensemble: Builds trees one at a time
        # Each new tree tries to correct mistakes of previous trees
        # Pros: Often very accurate, handles complex patterns
        # Cons: Slower to train, more complex
        
        print("\n5. Testing Gradient Boosting...")
        
        # Create Gradient Boosting model
        gb = GradientBoostingRegressor(
            n_estimators=50,      # Number of trees to build sequentially
            learning_rate=0.1,    # How much each tree contributes (lower = more conservative)
            max_depth=4,          # Maximum depth of each tree
            random_state=42       # For reproducibility
        )
        
        # Train the model
        gb.fit(X_train, self.y_train)
        
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
        
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        print("  - Splitting training data into {cv_folds} parts")
        print("  - Training on {cv_folds-1} parts, testing on 1 part")
        print("  - Repeating {cv_folds} times")
        
        # Use fewer folds for ensemble models (they're slower)
        if 'Forest' in self.best_model_name or 'Boosting' in self.best_model_name:
            cv_folds_actual = 3  # Reduce to 3 folds for ensemble models
            print(f"Using {cv_folds_actual} folds for faster execution...")
        else:
            cv_folds_actual = cv_folds
        
        # Create KFold object: Defines how to split data
        # shuffle=True: Randomly shuffle data before splitting
        # random_state=42: For reproducibility
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
        cv_rmse = np.sqrt(-cv_scores)
        
        # Cross-validation for R²
        cv_r2 = cross_val_score(
            self.model, 
            X_train, 
            self.y_train, 
            cv=kfold, 
            scoring='r2', 
            n_jobs=1
        )
        
        # Display cross-validation results
        # Mean: Average performance across all folds
        # Std * 2: 95% confidence interval (shows consistency)
        print(f"\nCross-Validation Results:")
        print(f"  RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        print(f"    - Lower is better")
        print(f"    - +/- shows variation across folds")
        print(f"  R² Score: {cv_r2.mean():.4f} (+/- {cv_r2.std() * 2:.4f})")
        print(f"    - Higher is better (max = 1.0)")
        print(f"    - +/- shows consistency")
        
        # ====================================================================
        # TRAINING SET EVALUATION
        # ====================================================================
        # Evaluate on data the model was trained on
        # This shows how well the model fits the training data
        # If train performance >> test performance, model may be overfitting
        
        print(f"\nEvaluating on training set...")
        
        # Make predictions on training data
        y_train_pred = self.model.predict(X_train)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        print(f"\nTraining Set Performance:")
        print(f"  RMSE: {train_rmse:.4f} (average prediction error)")
        print(f"  MAE: {train_mae:.4f} (average absolute error)")
        print(f"  R² Score: {train_r2:.4f} (how well model fits)")
        
        # ====================================================================
        # TEST SET EVALUATION
        # ====================================================================
        # Evaluate on data the model has never seen
        # This is the most important metric - shows real-world performance
        # Should be similar to cross-validation results
        
        print(f"\nEvaluating on test set (unseen data)...")
        
        # Make predictions on test data
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE: {test_rmse:.4f} (average prediction error)")
        print(f"  MAE: {test_mae:.4f} (average absolute error)")
        print(f"  R² Score: {test_r2:.4f} (how well model generalizes)")
        
        # Compare train vs test performance
        if train_r2 - test_r2 > 0.1:
            print(f"\n⚠ Warning: Large gap between train ({train_r2:.4f}) and test ({test_r2:.4f}) R²")
            print(f"  This suggests possible overfitting (model memorized training data)")
        
        # Store predictions for visualization
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred
        
        # Return all metrics as a dictionary
        return {
            'cv_rmse': cv_rmse.mean(),      # Cross-validation RMSE
            'cv_r2': cv_r2.mean(),          # Cross-validation R²
            'train_rmse': train_rmse,       # Training RMSE
            'train_mae': train_mae,          # Training MAE
            'train_r2': train_r2,           # Training R²
            'test_rmse': test_rmse,         # Test RMSE
            'test_mae': test_mae,           # Test MAE
            'test_r2': test_r2              # Test R²
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
            Predicted popularity values (0-100 scale)
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
            predictions = self.y_test_pred
            print("\nUsing test set for predictions")
            print("  - These predictions were already calculated during validation")
        else:
            # ====================================================================
            # CASE 2: Predict on new, unseen data
            # ====================================================================
            # This is what you'd use in production to predict popularity
            # for new songs
            
            print(f"\nPreprocessing new data for prediction...")
            
            # Create a copy to avoid modifying the original
            X_new_processed = X_new.copy()
            
            # Apply the same preprocessing steps used during training
            if use_scaled:
                # STEP 1: Scale features using the same scaler from training
                # transform (not fit_transform!) - use existing mean/std
                # This is critical - we can't refit on new data
                X_new_scaled = self.scaler.transform(X_new_processed)
                
                # STEP 2: Apply polynomial features if they were used
                if self.use_polynomial:
                    X_new_scaled = self.poly_features.transform(X_new_scaled)
                    print("  - Applied polynomial feature transformations")
                
                # STEP 3: Apply feature selection (keep only selected features)
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
        print(f"  Mean: {predictions.mean():.4f} (average predicted popularity)")
        print(f"  Std: {predictions.std():.4f} (variation in predictions)")
        print(f"  Min: {predictions.min():.4f} (lowest predicted popularity)")
        print(f"  Max: {predictions.max():.4f} (highest predicted popularity)")
        
        # Show first few individual predictions
        print(f"\nFirst 10 predictions:")
        for i, pred in enumerate(predictions[:10]):
            print(f"  Sample {i+1}: {pred:.4f}")
            if i == 0:
                print(f"    (Note: Popularity is typically on a 0-100 scale)")
        
        return predictions
    
    def visualize_results(self, save_plots=True):
        """
        Create enhanced visualizations for model results
        
        This function creates multiple plots to help understand model performance:
        1. Feature importance: Which features matter most
        2. Actual vs Predicted: How close predictions are to reality
        3. Residual plots: Shows prediction errors
        4. Error distributions: Shows how errors are distributed
        
        VISUALIZATIONS HELP US:
        - Understand which features are important
        - See if model has systematic biases
        - Identify if predictions are consistently too high/low
        - Check if errors are normally distributed (good sign)
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to PNG files (default: True)
            Saves: 'linear_regression_results.png' and 'feature_importance.png'
        """
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create a figure with a grid of subplots
        # figsize=(16, 12): 16 inches wide, 12 inches tall
        fig = plt.figure(figsize=(16, 12))
        
        # Create a 3x3 grid for subplots
        # hspace=0.3: Vertical spacing between subplots
        # wspace=0.3: Horizontal spacing between subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ====================================================================
        # GET FEATURE IMPORTANCE/COEFFICIENTS
        # ====================================================================
        # Different models store importance differently:
        # - Tree models (Random Forest, Gradient Boosting): feature_importances_
        # - Linear models (Linear, Ridge, Lasso): coef_ (coefficients)
        # - Coefficients show direction (+/-) and magnitude
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models: feature_importances_ shows how much each feature
            # contributes to predictions (0-1, higher = more important)
            importances = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)  # Top 15 features
        elif hasattr(self.model, 'coef_'):
            # Linear models: coef_ shows coefficients (weights)
            # Positive = increases popularity, Negative = decreases popularity
            # Absolute value shows magnitude of effect
            importances = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False).head(15)
            # Use absolute value for importance (magnitude matters)
            importances['Importance'] = np.abs(importances['Coefficient'])
        else:
            # Some models don't have feature importance
            importances = None
        
        # 1. Feature importance/coefficients (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        if importances is not None:
            colors = ['#2E86AB' if x >= 0 else '#A23B72' for x in importances.get('Coefficient', importances['Importance'])]
            bars = ax1.barh(importances['Feature'], importances['Importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            ax1.set_xlabel('Importance / |Coefficient|', fontweight='bold')
            ax1.set_title('Top Feature Importances', fontweight='bold', fontsize=12)
            for i, (bar, val) in enumerate(zip(bars, importances['Importance'])):
                ax1.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left', fontweight='bold', fontsize=8)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_facecolor('#F8F9FA')
        
        # 2. Actual vs Predicted - Training (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(self.y_train, self.y_train_pred, alpha=0.6, s=30, 
                             c=self.y_train, cmap='viridis', edgecolors='black', linewidth=0.5)
        ax2.plot([self.y_train.min(), self.y_train.max()], 
                [self.y_train.min(), self.y_train.max()], 
                'r--', lw=2.5, label='Perfect Prediction', zorder=3)
        ax2.set_xlabel('Actual Popularity', fontweight='bold')
        ax2.set_ylabel('Predicted Popularity', fontweight='bold')
        ax2.set_title('Training Set: Actual vs Predicted', fontweight='bold', fontsize=12)
        ax2.legend(loc='upper left', framealpha=0.9)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_facecolor('#F8F9FA')
        plt.colorbar(scatter, ax=ax2, label='Actual Popularity')
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        ax2.text(0.05, 0.95, f'R² = {train_r2:.4f}', transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontweight='bold')
        
        # 3. Actual vs Predicted - Test (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        scatter = ax3.scatter(self.y_test, self.y_test_pred, alpha=0.6, s=30, 
                             c=self.y_test, cmap='plasma', edgecolors='black', linewidth=0.5)
        ax3.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2.5, label='Perfect Prediction', zorder=3)
        ax3.set_xlabel('Actual Popularity', fontweight='bold')
        ax3.set_ylabel('Predicted Popularity', fontweight='bold')
        ax3.set_title('Test Set: Actual vs Predicted', fontweight='bold', fontsize=12)
        ax3.legend(loc='upper left', framealpha=0.9)
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.set_facecolor('#F8F9FA')
        plt.colorbar(scatter, ax=ax3, label='Actual Popularity')
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        ax3.text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontweight='bold')
        
        # 4. Residuals plot - Training (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        residuals_train = self.y_train - self.y_train_pred
        scatter4 = ax4.scatter(self.y_train_pred, residuals_train, alpha=0.6, s=30, 
                   c=residuals_train, cmap='coolwarm', edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Zero Residual')
        ax4.set_xlabel('Predicted Popularity', fontweight='bold')
        ax4.set_ylabel('Residuals', fontweight='bold')
        ax4.set_title('Training Set: Residuals Plot', fontweight='bold', fontsize=12)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(alpha=0.3, linestyle='--')
        ax4.set_facecolor('#F8F9FA')
        plt.colorbar(scatter4, ax=ax4, label='Residual Value')
        
        # 5. Residuals plot - Test (Middle Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        residuals_test = self.y_test - self.y_test_pred
        scatter5 = ax5.scatter(self.y_test_pred, residuals_test, alpha=0.6, s=30, 
                   c=residuals_test, cmap='coolwarm', edgecolors='black', linewidth=0.5)
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Zero Residual')
        ax5.set_xlabel('Predicted Popularity', fontweight='bold')
        ax5.set_ylabel('Residuals', fontweight='bold')
        ax5.set_title('Test Set: Residuals Plot', fontweight='bold', fontsize=12)
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(alpha=0.3, linestyle='--')
        ax5.set_facecolor('#F8F9FA')
        plt.colorbar(scatter5, ax=ax5, label='Residual Value')
        
        # 6. Distribution of residuals - Test (Middle Right)
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
        
        # 7. Feature importance (Bottom Left)
        ax7 = fig.add_subplot(gs[2, 0])
        if importances is not None:
            importances_sorted = importances.sort_values('Importance', ascending=True)
            colors_abs = ['#2E86AB' if x >= 0 else '#A23B72' for x in importances_sorted.get('Coefficient', importances_sorted['Importance'])]
            bars = ax7.barh(importances_sorted['Feature'], importances_sorted['Importance'], 
                           color=colors_abs, alpha=0.8, edgecolor='black', linewidth=1.2)
            ax7.set_xlabel('Importance', fontweight='bold')
            ax7.set_title('Feature Importance Ranking', fontweight='bold', fontsize=12)
            for i, (bar, val) in enumerate(zip(bars, importances_sorted['Importance'])):
                ax7.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left', 
                        fontweight='bold', fontsize=8)
        ax7.grid(axis='x', alpha=0.3, linestyle='--')
        ax7.set_facecolor('#F8F9FA')
        
        # 8. Prediction error distribution (Bottom Middle)
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
        
        # 9. Model performance metrics (Bottom Right)
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
        
        # Add main title
        fig.suptitle('Spotify Tracks Popularity Prediction - Model Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight', facecolor='white')
            print("\nPlot saved as 'linear_regression_results.png'")
        
        plt.show()
        
        # 2. Separate Feature importance plot
        if importances is not None:
            plt.figure(figsize=(12, 7))
            importances_display = importances.head(20).sort_values('Importance', ascending=True)
            colors_abs = ['#2E86AB' if x >= 0 else '#A23B72' for x in importances_display.get('Coefficient', importances_display['Importance'])]
            bars = plt.barh(importances_display['Feature'], importances_display['Importance'], 
                           color=colors_abs, alpha=0.8, edgecolor='black', linewidth=1.5)
            plt.xlabel('Importance / |Coefficient|', fontweight='bold', fontsize=12)
            plt.ylabel('Feature', fontweight='bold', fontsize=12)
            plt.title('Feature Importance Analysis', fontweight='bold', fontsize=14, pad=20)
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
    
    def run_full_pipeline(self, target_column='popularity', test_size=0.2, 
                         random_state=42, use_scaled=True, cv_folds=3):
        """
        Run the complete modeling pipeline
        
        Parameters:
        -----------
        target_column : str
            Name of the target variable
        test_size : float
            Proportion of dataset for testing
        random_state : int
            Random state for reproducibility
        use_scaled : bool
            Whether to use scaled features
        cv_folds : int
            Number of cross-validation folds
        """
        print("\n" + "=" * 60)
        print("SPOTIFY TRACKS PREDICTION - ADVANCED MODEL PIPELINE")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Polynomial features: {self.use_polynomial}")
        print(f"  - Outlier handling: {self.handle_outliers}")
        
        # Load data
        self.load_data()
        
        # Preprocess
        self.preprocess_data(target_column=target_column)
        
        # Split data
        self.split_data(test_size=test_size, random_state=random_state)
        
        # Scale features
        if use_scaled:
            self.scale_features()
        
        # Train best model
        self.train_best_model(use_scaled=use_scaled)
        
        # Validate model
        metrics = self.validate_model(use_scaled=use_scaled, cv_folds=cv_folds)
        
        # Make predictions
        self.make_predictions(use_scaled=use_scaled)
        
        # Visualize results
        self.visualize_results()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return metrics


# ============================================================================
# MAIN FUNCTION - Entry point when script is run directly
# ============================================================================

def main():
    """
    Main function to run the advanced linear regression modeling
    
    This function demonstrates how to use the SpotifyLinearRegression class
    to build a complete machine learning pipeline from start to finish.
    
    The pipeline includes:
    1. Loading data
    2. Preprocessing
    3. Feature engineering
    4. Model training and selection
    5. Model validation
    6. Making predictions
    7. Creating visualizations
    """
    
    # ====================================================================
    # STEP 1: INITIALIZE THE MODEL
    # ====================================================================
    # Create an instance of the SpotifyLinearRegression class
    # This sets up all the necessary components
    
    print("Initializing Spotify Popularity Prediction Model...")
    model = SpotifyLinearRegression(
        csv_path='dataset.csv',        # Path to data file
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
    
    print("\nRunning complete machine learning pipeline...")
    metrics = model.run_full_pipeline(
        target_column='popularity',    # What we're predicting
        test_size=0.2,                 # 20% for testing, 80% for training
        random_state=42,               # For reproducibility
        use_scaled=True,               # Use scaled features (recommended)
        cv_folds=3                    # 3-fold cross-validation (faster)
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
    print(f"    - Average prediction error in popularity units")
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


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================
# This code runs when the script is executed directly (not imported)
# Example: python linear_regression_modeling.py

if __name__ == "__main__":
    # Call the main function to start the pipeline
    main()
