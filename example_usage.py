"""
Example usage of the Linear Regression Model for Spotify Tracks Prediction

This script demonstrates different ways to use the SpotifyLinearRegression class.
It shows various usage patterns from simple to advanced.

EXAMPLES INCLUDED:
1. Basic Usage: Run the complete pipeline with one function call
2. Step-by-Step Usage: Control each step individually
3. Predicting New Data: Make predictions on new songs
4. Model Comparison: Compare scaled vs unscaled features

This file is a learning resource - run different examples to understand
how the model works!
"""

# ============================================================================
# IMPORT STATEMENTS
# ============================================================================

# Import the main class we'll be using
from linear_regression_modeling import SpotifyLinearRegression

# pandas: For creating DataFrames (tables) for new data
import pandas as pd

# numpy: For mathematical operations (not heavily used here, but good practice)
import numpy as np

def example_basic_usage():
    """
    Basic usage example - Simplest way to use the model
    
    This example shows the easiest way to build and evaluate a model.
    Just initialize and call run_full_pipeline() - that's it!
    
    This is recommended for beginners or when you want quick results.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage - Full Pipeline")
    print("=" * 60)
    print("\nThis is the simplest way to use the model.")
    print("Just initialize and run the complete pipeline!\n")
    
    # ====================================================================
    # STEP 1: CREATE MODEL INSTANCE
    # ====================================================================
    # Create an instance of the SpotifyLinearRegression class
    # This sets up everything needed for machine learning
    model = SpotifyLinearRegression(csv_path='dataset.csv')
    
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
        target_column='popularity',    # What we're predicting
        test_size=0.2,                 # 20% for testing, 80% for training
        random_state=42,               # For reproducible results
        use_scaled=True,               # Use scaled features (recommended)
        cv_folds=5                    # 5-fold cross-validation
    )
    
    # ====================================================================
    # STEP 3: VIEW RESULTS
    # ====================================================================
    # The metrics dictionary contains all performance measures
    
    print("\n" + "=" * 60)
    print("Model Performance Summary:")
    print("=" * 60)
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"    - R² Score: How well the model fits (0-1, higher is better)")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"    - RMSE: Root Mean Squared Error (lower is better)")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")
    print(f"    - MAE: Mean Absolute Error (lower is better)")
    
    print("\n✅ Example 1 complete!")
    print("   The model has been trained and evaluated.")
    print("   Check the generated plots for visual analysis.")


def example_step_by_step():
    """
    Step-by-step usage example - More control over the process
    
    This example shows how to run each step individually.
    Useful when you want to:
    - Inspect data between steps
    - Modify parameters at specific stages
    - Understand what each step does
    - Add custom processing between steps
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Step-by-Step Usage")
    print("=" * 60)
    print("\nThis example shows how to run each step individually.")
    print("This gives you more control and helps you understand the process.\n")
    
    # ====================================================================
    # STEP 1: INITIALIZE MODEL
    # ====================================================================
    # Create the model instance (same as Example 1)
    model = SpotifyLinearRegression(csv_path='dataset.csv')
    
    # ====================================================================
    # STEP 2: LOAD DATA
    # ====================================================================
    # Load the CSV file and display basic information
    # You can inspect the data here if needed
    print("Step 1: Loading data...")
    model.load_data()
    
    # ====================================================================
    # STEP 3: PREPROCESS DATA
    # ====================================================================
    # Clean data, select features, handle outliers
    # Returns features (X) and target (y)
    print("\nStep 2: Preprocessing data...")
    X, y = model.preprocess_data(target_column='popularity')
    
    # You could inspect X and y here if you want:
    # print(f"Features shape: {X.shape}")
    # print(f"Target shape: {y.shape}")
    
    # ====================================================================
    # STEP 4: SPLIT DATA
    # ====================================================================
    # Divide data into training and testing sets
    print("\nStep 3: Splitting data...")
    model.split_data(test_size=0.2, random_state=42)
    
    # ====================================================================
    # STEP 5: SCALE FEATURES
    # ====================================================================
    # Normalize features and optionally create polynomial features
    print("\nStep 4: Scaling features...")
    model.scale_features()
    
    # ====================================================================
    # STEP 6: TRAIN MODEL
    # ====================================================================
    # Train multiple models and select the best one
    # NOTE: The method is actually called train_best_model, not train_model
    print("\nStep 5: Training model...")
    model.train_best_model(use_scaled=True)
    
    # ====================================================================
    # STEP 7: VALIDATE MODEL
    # ====================================================================
    # Evaluate model performance using cross-validation and test set
    print("\nStep 6: Validating model...")
    metrics = model.validate_model(use_scaled=True, cv_folds=5)
    
    # ====================================================================
    # STEP 8: MAKE PREDICTIONS
    # ====================================================================
    # Generate predictions (on test set by default)
    print("\nStep 7: Making predictions...")
    predictions = model.make_predictions(use_scaled=True)
    
    # ====================================================================
    # STEP 9: VISUALIZE RESULTS
    # ====================================================================
    # Create plots showing model performance
    print("\nStep 8: Creating visualizations...")
    model.visualize_results(save_plots=True)
    
    print("\n✅ Example 2 complete!")
    print("   You've seen each step of the machine learning pipeline.")


def example_predict_new_data():
    """
    Example of making predictions on new data
    
    This example shows how to use a trained model to predict popularity
    for new songs that weren't in the training data.
    
    This is what you'd do in a real application:
    1. Train the model once (or periodically)
    2. Use it to predict popularity for new songs as they're released
    
    IMPORTANT: New data must have the same features in the same format!
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Predicting on New Data")
    print("=" * 60)
    print("\nThis example shows how to predict popularity for new songs.")
    print("First, we train the model, then use it to make predictions.\n")
    
    # ====================================================================
    # STEP 1: TRAIN THE MODEL
    # ====================================================================
    # First, we need to train the model on existing data
    # In practice, you'd do this once and save the model
    
    print("Training model on existing data...")
    model = SpotifyLinearRegression(csv_path='dataset.csv')
    model.load_data()
    model.preprocess_data(target_column='popularity')
    model.split_data(test_size=0.2, random_state=42)
    model.scale_features()
    model.train_best_model(use_scaled=True)  # Fixed: was train_model
    
    print("✅ Model trained and ready for predictions!\n")
    
    # ====================================================================
    # STEP 2: PREPARE NEW DATA
    # ====================================================================
    # Create a DataFrame with features for a new song
    # IMPORTANT: Must have the same columns as training data!
    # Features: danceability, energy, valence, tempo, loudness, Duration_min, explicit
    
    print("=" * 60)
    print("PREDICTING ON SINGLE NEW TRACK")
    print("=" * 60)
    
    # Create a DataFrame for one new track
    # Each feature is in a list [value] - DataFrame needs lists/arrays
    new_track = pd.DataFrame({
        'danceability': [0.75],      # 0.0-1.0, how danceable
        'energy': [0.65],            # 0.0-1.0, how energetic
        'valence': [0.80],           # 0.0-1.0, how positive/happy
        'tempo': [120.0],            # Beats per minute
        'loudness': [-5.0],         # Decibels (usually negative)
        'Duration_min': [3.5],       # Duration in minutes
        'explicit': [0]              # 0 = False (no explicit content), 1 = True
    })
    
    print("\nNew track features:")
    print(new_track)
    print("\nInterpreting features:")
    print("  - High danceability (0.75): Good for dancing")
    print("  - Moderate energy (0.65): Energetic but not extreme")
    print("  - High valence (0.80): Positive, happy mood")
    print("  - Moderate tempo (120 BPM): Typical pop song speed")
    print("  - Moderate loudness (-5 dB): Not too quiet or loud")
    print("  - Short duration (3.5 min): Typical song length")
    print("  - Not explicit (0): Clean version")
    
    # ====================================================================
    # STEP 3: MAKE PREDICTION
    # ====================================================================
    # Use the trained model to predict popularity
    # NOTE: We need to apply the same preprocessing (scaling) that was
    # used during training. The make_predictions method handles this!
    
    # Use the make_predictions method (recommended - handles preprocessing)
    predictions = model.make_predictions(X_new=new_track, use_scaled=True)
    prediction = predictions[0]  # Get first (and only) prediction
    
    print(f"\n{'='*60}")
    print(f"PREDICTED POPULARITY: {prediction:.2f}")
    print(f"{'='*60}")
    print(f"  - Popularity scale: 0-100")
    print(f"  - Higher = more popular")
    print(f"  - This song is predicted to have popularity of {prediction:.2f}")
    
    # ====================================================================
    # STEP 4: PREDICT MULTIPLE TRACKS
    # ====================================================================
    # You can predict multiple songs at once (more efficient)
    
    print("\n" + "=" * 60)
    print("PREDICTING ON MULTIPLE TRACKS")
    print("=" * 60)
    
    # Create DataFrame with multiple tracks (one row per track)
    multiple_tracks = pd.DataFrame({
        'danceability': [0.75, 0.50, 0.90],      # Track 1, 2, 3
        'energy': [0.65, 0.40, 0.85],
        'valence': [0.80, 0.30, 0.95],
        'tempo': [120.0, 90.0, 140.0],
        'loudness': [-5.0, -10.0, -3.0],
        'Duration_min': [3.5, 4.0, 3.0],
        'explicit': [0, 1, 0]  # Track 2 has explicit content
    })
    
    print("\nMultiple tracks features:")
    print(multiple_tracks)
    print("\nTrack descriptions:")
    print("  Track 1: Danceable, energetic, happy (pop song)")
    print("  Track 2: Less danceable, low energy, sad (ballad)")
    print("  Track 3: Very danceable, high energy, very happy (party song)")
    
    # Make predictions for all tracks at once
    predictions = model.make_predictions(X_new=multiple_tracks, use_scaled=True)
    
    print("\n" + "=" * 60)
    print("PREDICTED POPULARITIES:")
    print("=" * 60)
    for i, pred in enumerate(predictions):
        print(f"  Track {i+1}: {pred:.2f}")
        if pred > 70:
            print(f"    → High popularity! This track is predicted to be very popular.")
        elif pred > 50:
            print(f"    → Moderate popularity. Decent chance of success.")
        else:
            print(f"    → Lower popularity. May need more promotion.")
    
    print("\n✅ Example 3 complete!")
    print("   You've learned how to predict popularity for new songs!")


def example_model_comparison():
    """
    Compare scaled vs unscaled features
    
    This example demonstrates why feature scaling is important.
    We train two models (one with scaled features, one without) and
    compare their performance.
    
    EXPECTED RESULT: Scaled features usually perform better because:
    - All features are on the same scale
    - Models can learn more effectively
    - Prevents features with larger values from dominating
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Comparing Scaled vs Unscaled Features")
    print("=" * 60)
    print("\nThis example compares model performance with and without scaling.")
    print("Scaling usually improves performance!\n")
    
    # ====================================================================
    # MODEL 1: WITH SCALED FEATURES
    # ====================================================================
    print("=" * 60)
    print("TRAINING MODEL WITH SCALED FEATURES")
    print("=" * 60)
    print("\nScaling normalizes features so they have:")
    print("  - Mean = 0")
    print("  - Standard deviation = 1")
    print("This helps models learn more effectively.\n")
    
    model_scaled = SpotifyLinearRegression(csv_path='dataset.csv')
    model_scaled.load_data()
    model_scaled.preprocess_data(target_column='popularity')
    model_scaled.split_data(test_size=0.2, random_state=42)
    model_scaled.scale_features()  # Scale the features
    model_scaled.train_best_model(use_scaled=True)  # Use scaled features
    metrics_scaled = model_scaled.validate_model(use_scaled=True, cv_folds=5)
    
    # ====================================================================
    # MODEL 2: WITHOUT SCALING
    # ====================================================================
    print("\n" + "=" * 60)
    print("TRAINING MODEL WITHOUT SCALING")
    print("=" * 60)
    print("\nUsing original feature values (different scales).")
    print("This may perform worse because features have different ranges.\n")
    
    model_unscaled = SpotifyLinearRegression(csv_path='dataset.csv')
    model_unscaled.load_data()
    model_unscaled.preprocess_data(target_column='popularity')
    model_unscaled.split_data(test_size=0.2, random_state=42)
    # Note: We don't call scale_features() here
    model_unscaled.train_best_model(use_scaled=False)  # Use unscaled features
    metrics_unscaled = model_unscaled.validate_model(use_scaled=False, cv_folds=5)
    
    # ====================================================================
    # COMPARE RESULTS
    # ====================================================================
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\nScaled Features:")
    print(f"  Test R²: {metrics_scaled['test_r2']:.4f}")
    print(f"    - Higher is better (max = 1.0)")
    print(f"  Test RMSE: {metrics_scaled['test_rmse']:.4f}")
    print(f"    - Lower is better")
    print(f"  Best Model: {model_scaled.best_model_name}")
    
    print(f"\nUnscaled Features:")
    print(f"  Test R²: {metrics_unscaled['test_r2']:.4f}")
    print(f"    - Higher is better (max = 1.0)")
    print(f"  Test RMSE: {metrics_unscaled['test_rmse']:.4f}")
    print(f"    - Lower is better")
    print(f"  Best Model: {model_unscaled.best_model_name}")
    
    # Calculate improvement
    r2_improvement = metrics_scaled['test_r2'] - metrics_unscaled['test_r2']
    rmse_improvement = metrics_unscaled['test_rmse'] - metrics_scaled['test_rmse']
    
    print(f"\n{'='*60}")
    print("IMPROVEMENT FROM SCALING:")
    print(f"{'='*60}")
    if r2_improvement > 0:
        print(f"  R² improved by: {r2_improvement:.4f} (scaling helped!)")
    else:
        print(f"  R² changed by: {r2_improvement:.4f} (scaling didn't help much)")
    
    if rmse_improvement > 0:
        print(f"  RMSE improved by: {rmse_improvement:.4f} (lower is better)")
    else:
        print(f"  RMSE changed by: {rmse_improvement:.4f}")
    
    print("\n✅ Example 4 complete!")
    print("   You've seen how scaling affects model performance!")


# ============================================================================
# MAIN EXECUTION - Run examples when script is executed
# ============================================================================

if __name__ == "__main__":
    """
    This code runs when you execute: python example_usage.py
    
    By default, only example_basic_usage() runs.
    Uncomment other examples to try them!
    """
    
    # ====================================================================
    # EXAMPLE 1: Basic Usage (Always runs)
    # ====================================================================
    # This is the simplest example - recommended for beginners
    example_basic_usage()
    
    # ====================================================================
    # OTHER EXAMPLES (Commented out by default)
    # ====================================================================
    # Uncomment the examples you want to try:
    
    # Example 2: Step-by-step usage (more control)
    # example_step_by_step()
    
    # Example 3: Predicting on new data (real-world usage)
    # example_predict_new_data()
    
    # Example 4: Comparing scaled vs unscaled features
    # example_model_comparison()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 60)
    print("\nTo try other examples, uncomment them in the code above.")
    print("Each example demonstrates different aspects of using the model.")
