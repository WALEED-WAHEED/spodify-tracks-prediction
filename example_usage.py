"""
Example usage of the Job Market Analysis Model for Canadian Job Postings

This script demonstrates different ways to use the JobMarketAnalysis class.
It shows various usage patterns from simple to advanced.

EXAMPLES INCLUDED:
1. Basic Usage: Run the complete pipeline with one function call
2. Step-by-Step Usage: Control each step individually
3. Analyzing Specific Regions: Focus on particular provinces/territories
4. Model Comparison: Compare different target variables

This file is a learning resource - run different examples to understand
how the model works!
"""

# ============================================================================
# IMPORT STATEMENTS
# ============================================================================

# Import the main JobMarketAnalysis class from our modeling module
# This class contains all the functionality for analyzing job market data
from linear_regression_modeling import JobMarketAnalysis

# pandas: For creating DataFrames (tables) when preparing new data for prediction
# Used in examples where we create sample data to demonstrate predictions
import pandas as pd

# numpy: For mathematical operations (not heavily used here, but good practice)
# Can be useful for data manipulation and calculations
import numpy as np

def example_basic_usage():
    """
    Basic usage example - Simplest way to use the model
    
    This example shows the easiest way to build and evaluate a model.
    Just initialize and call run_full_pipeline() - that's it!
    """
    # Print header for this example
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage - Full Pipeline")
    print("=" * 60)
    print("\nThis is the simplest way to use the model.")
    print("Just initialize and run the complete pipeline!\n")
    
    # ====================================================================
    # STEP 1: CREATE MODEL INSTANCE
    # ====================================================================
    # Create an instance of the JobMarketAnalysis class
    # This sets up everything needed for machine learning
    # The default settings are usually good for most use cases
    model = JobMarketAnalysis(
        csv_path='job-bank-open-data-all-job-postings-en-december2025.csv'  # Path to data file
    )
    
    # ====================================================================
    # STEP 2: RUN COMPLETE PIPELINE
    # ====================================================================
    # This one function call does everything:
    # - Loads data from CSV
    # - Preprocesses and cleans data
    # - Splits into training and testing sets
    # - Scales features
    # - Trains multiple models and picks the best
    # - Validates the model
    # - Makes predictions
    # - Creates visualizations
    # - Returns performance metrics
    
    print("Running complete pipeline...")
    metrics = model.run_full_pipeline(
        target_column='Vacancy Count',    # What we're predicting (number of job vacancies)
        test_size=0.2,                     # 20% for testing, 80% for training
                                         # This split ensures we have enough data for both training and testing
        random_state=42,                   # For reproducible results (same seed = same random split)
        use_scaled=True,                   # Use scaled features (recommended for better performance)
        cv_folds=5                        # 5-fold cross-validation (tests model 5 times for reliability)
    )
    
    # ====================================================================
    # STEP 3: VIEW RESULTS
    # ====================================================================
    # The metrics dictionary contains all performance measures
    # We display the most important ones
    
    print("\n" + "=" * 60)
    print("Model Performance Summary:")
    print("=" * 60)
    
    # R² Score: Proportion of variance explained by the model
    # Range: -∞ to 1.0, where 1.0 = perfect predictions
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"    - R² Score: How well the model fits (0-1, higher is better)")
    
    # RMSE: Root Mean Squared Error
    # Average prediction error in the same units as the target variable
    # Lower is better - shows how far off predictions are on average
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"    - RMSE: Root Mean Squared Error (lower is better)")
    
    # MAE: Mean Absolute Error
    # Average absolute difference between predictions and actual values
    # Easier to interpret than RMSE (doesn't penalize large errors as much)
    print(f"  Test MAE: {metrics['test_mae']:.4f}")
    print(f"    - MAE: Mean Absolute Error (lower is better)")
    
    print("\n✅ Example 1 complete!")
    print("   The model has been trained and evaluated.")
    print("   Check the generated plots for visual analysis.")


def example_step_by_step():
    """
    Step-by-step usage example - More control over the process
    
    This example shows how to run each step individually.
    Useful when you want to inspect data between steps or modify parameters.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Step-by-Step Usage")
    print("=" * 60)
    print("\nThis example shows how to run each step individually.")
    print("This gives you more control and helps you understand the process.\n")
    
    # ====================================================================
    # INITIALIZE MODEL
    # ====================================================================
    # Create the model instance (same as Example 1)
    model = JobMarketAnalysis(
        csv_path='job-bank-open-data-all-job-postings-en-december2025.csv'
    )
    
    # ====================================================================
    # STEP 1: LOAD DATA
    # ====================================================================
    # Load the CSV file and display basic information
    # You can inspect the data here if needed (e.g., print(model.df.head()))
    print("Step 1: Loading data...")
    model.load_data()
    
    # ====================================================================
    # STEP 2: PREPROCESS DATA
    # ====================================================================
    # Clean data, select features, handle outliers, encode categorical variables
    # Returns features (X) and target (y)
    # You could inspect X and y here if you want:
    #   print(f"Features shape: {X.shape}")
    #   print(f"Target shape: {y.shape}")
    print("\nStep 2: Preprocessing data...")
    X, y = model.preprocess_data(target_column='Vacancy Count')
    
    # ====================================================================
    # STEP 3: SPLIT DATA
    # ====================================================================
    # Divide data into training and testing sets
    # Training set: Used to teach the model
    # Testing set: Used to evaluate the model (simulates new, unseen data)
    print("\nStep 3: Splitting data...")
    model.split_data(test_size=0.2, random_state=42)
    
    # ====================================================================
    # STEP 4: SCALE FEATURES
    # ====================================================================
    # Normalize features (mean=0, std=1) and optionally create polynomial features
    # Scaling helps models learn more effectively
    print("\nStep 4: Scaling features...")
    model.scale_features()
    
    # ====================================================================
    # STEP 5: TRAIN MODEL
    # ====================================================================
    # Train multiple models (Linear, Ridge, Random Forest, Gradient Boosting)
    # and select the best one based on performance
    print("\nStep 5: Training model...")
    model.train_best_model(use_scaled=True)
    
    # ====================================================================
    # STEP 6: VALIDATE MODEL
    # ====================================================================
    # Evaluate model performance using cross-validation and test set
    # Cross-validation tests the model multiple times for reliability
    print("\nStep 6: Validating model...")
    metrics = model.validate_model(use_scaled=True, cv_folds=5)
    
    # ====================================================================
    # STEP 7: MAKE PREDICTIONS
    # ====================================================================
    # Generate predictions (on test set by default)
    # These predictions can be compared to actual values to see how well the model works
    print("\nStep 7: Making predictions...")
    predictions = model.make_predictions(use_scaled=True)
    
    # ====================================================================
    # STEP 8: VISUALIZE RESULTS
    # ====================================================================
    # Create plots showing model performance
    # Includes: feature importance, actual vs predicted, residuals, error distributions
    # save_plots=True saves the plots as PNG files
    print("\nStep 8: Creating visualizations...")
    model.visualize_results(save_plots=True)
    
    print("\n✅ Example 2 complete!")
    print("   You've seen each step of the machine learning pipeline.")


def example_predict_salary():
    """
    Example of predicting salary instead of vacancy count
    
    This example shows how to change the target variable to predict
    salary trends instead of job demand.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Predicting Salary Trends")
    print("=" * 60)
    print("\nThis example predicts salary instead of vacancy count.")
    print("This helps understand compensation trends in the job market.\n")
    
    # ====================================================================
    # TRAIN MODEL WITH SALARY AS TARGET
    # ====================================================================
    # Instead of predicting vacancy count, we predict annual salary
    # This helps understand what factors influence compensation
    
    print("Training model to predict salary...")
    model = JobMarketAnalysis(
        csv_path='job-bank-open-data-all-job-postings-en-december2025.csv'
    )
    model.load_data()
    
    # Note: We need to create Salary_Annual first in preprocessing
    # The preprocessing method handles this automatically by converting
    # hourly salaries to annual equivalents (assuming 40 hrs/week * 52 weeks)
    model.preprocess_data(target_column='Salary_Annual')  # Changed target to salary
    model.split_data(test_size=0.2, random_state=42)
    model.scale_features()
    model.train_best_model(use_scaled=True)
    metrics = model.validate_model(use_scaled=True, cv_folds=5)
    
    # ====================================================================
    # DISPLAY SALARY PREDICTION RESULTS
    # ====================================================================
    # Note: RMSE and MAE are now in dollars (not vacancy counts)
    # This makes it easier to interpret - e.g., "average error is $5,000"
    
    print("\n" + "=" * 60)
    print("Salary Prediction Results:")
    print("=" * 60)
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Test RMSE: ${metrics['test_rmse']:,.2f}")
    print(f"    - Average prediction error in dollars")
    print(f"    - Example: If RMSE is $5,000, predictions are off by $5,000 on average")
    print(f"  Test MAE: ${metrics['test_mae']:,.2f}")
    print(f"    - Average absolute error")
    print(f"    - Easier to interpret than RMSE (doesn't penalize large errors as much)")
    
    print("\n✅ Example 3 complete!")
    print("   You've learned how to predict salary trends!")


def example_model_comparison():
    """
    Compare different target variables
    
    This example demonstrates how different target variables
    (vacancy count vs salary) affect model performance.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Comparing Different Target Variables")
    print("=" * 60)
    print("\nThis example compares model performance for different targets.")
    print("We'll compare predicting vacancy count vs salary.\n")
    
    # Model 1: Predict Vacancy Count
    print("=" * 60)
    print("MODEL 1: Predicting Vacancy Count")
    print("=" * 60)
    print("\nThis model predicts job demand (number of vacancies).\n")
    
    model_vacancy = JobMarketAnalysis(
        csv_path='job-bank-open-data-all-job-postings-en-december2025.csv'
    )
    model_vacancy.load_data()
    model_vacancy.preprocess_data(target_column='Vacancy Count')
    model_vacancy.split_data(test_size=0.2, random_state=42)
    model_vacancy.scale_features()
    model_vacancy.train_best_model(use_scaled=True)
    metrics_vacancy = model_vacancy.validate_model(use_scaled=True, cv_folds=3)
    
    # Model 2: Predict Salary
    print("\n" + "=" * 60)
    print("MODEL 2: Predicting Salary")
    print("=" * 60)
    print("\nThis model predicts compensation levels.\n")
    
    model_salary = JobMarketAnalysis(
        csv_path='job-bank-open-data-all-job-postings-en-december2025.csv'
    )
    model_salary.load_data()
    model_salary.preprocess_data(target_column='Salary_Annual')
    model_salary.split_data(test_size=0.2, random_state=42)
    model_salary.scale_features()
    model_salary.train_best_model(use_scaled=True)
    metrics_salary = model_salary.validate_model(use_scaled=True, cv_folds=3)
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\nVacancy Count Prediction:")
    print(f"  Test R²: {metrics_vacancy['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics_vacancy['test_rmse']:.4f}")
    print(f"  Best Model: {model_vacancy.best_model_name}")
    
    print(f"\nSalary Prediction:")
    print(f"  Test R²: {metrics_salary['test_r2']:.4f}")
    print(f"  Test RMSE: ${metrics_salary['test_rmse']:,.2f}")
    print(f"  Best Model: {model_salary.best_model_name}")
    
    print("\n" + "=" * 60)
    print("INSIGHTS:")
    print("=" * 60)
    print("  - Different targets may require different models")
    print("  - Some targets are easier to predict than others")
    print("  - R² score shows how well each model explains variance")
    print("  - RMSE shows prediction error in original units")
    
    print("\n✅ Example 4 complete!")
    print("   You've compared different prediction targets!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block - runs when script is executed directly
    
    This code runs when you execute: python example_usage.py
    
    By default, only example_basic_usage() runs (the simplest example).
    Uncomment other examples to try them!
    
    Each example demonstrates different aspects:
    - Example 1: Quick start (simplest way to use the model)
    - Example 2: Step-by-step (more control, better understanding)
    - Example 3: Different target (predicting salary instead of vacancies)
    - Example 4: Model comparison (comparing different targets)
    """
    
    # ====================================================================
    # EXAMPLE 1: Basic Usage (Always runs)
    # ====================================================================
    # This is the simplest example - recommended for beginners
    # Shows how to use the model with just a few lines of code
    example_basic_usage()
    
    # ====================================================================
    # OTHER EXAMPLES (Commented out by default)
    # ====================================================================
    # Uncomment the examples you want to try:
    # Each example is independent and can be run separately
    
    # Example 2: Step-by-step usage (more control over each step)
    # Useful when you want to inspect data between steps or modify parameters
    # example_step_by_step()
    
    # Example 3: Predicting salary trends (different target variable)
    # Shows how to change what you're predicting
    # example_predict_salary()
    
    # Example 4: Comparing different targets (vacancy count vs salary)
    # Shows how different targets affect model performance
    # example_model_comparison()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 60)
    print("\nTo try other examples, uncomment them in the code above.")
    print("Each example demonstrates different aspects of using the model.")
    print("\nTips:")
    print("  - Start with Example 1 to get familiar with the model")
    print("  - Try Example 2 to understand each step in detail")
    print("  - Use Example 3 to predict different targets")
    print("  - Run Example 4 to compare model performance")
