"""
Example usage of the Linear Regression Model for Spotify Tracks Prediction

This script demonstrates how to:
1. Load and preprocess data
2. Train a linear regression model
3. Validate the model
4. Make predictions on new data
"""

from linear_regression_modeling import SpotifyLinearRegression
import pandas as pd
import numpy as np

def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage - Full Pipeline")
    print("=" * 60)
    
    # Initialize model
    model = SpotifyLinearRegression(csv_path='dataset.csv')
    
    # Run complete pipeline
    metrics = model.run_full_pipeline(
        target_column='popularity',
        test_size=0.2,
        random_state=42,
        use_scaled=True,
        cv_folds=5
    )
    
    print("\nModel Performance Summary:")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")


def example_step_by_step():
    """Step-by-step usage example"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Step-by-Step Usage")
    print("=" * 60)
    
    # Initialize model
    model = SpotifyLinearRegression(csv_path='dataset.csv')
    
    # Step 1: Load data
    model.load_data()
    
    # Step 2: Preprocess
    X, y = model.preprocess_data(target_column='popularity')
    
    # Step 3: Split data
    model.split_data(test_size=0.2, random_state=42)
    
    # Step 4: Scale features
    model.scale_features()
    
    # Step 5: Train model
    model.train_model(use_scaled=True)
    
    # Step 6: Validate
    metrics = model.validate_model(use_scaled=True, cv_folds=5)
    
    # Step 7: Make predictions
    predictions = model.make_predictions(use_scaled=True)
    
    # Step 8: Visualize
    model.visualize_results(save_plots=True)


def example_predict_new_data():
    """Example of making predictions on new data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Predicting on New Data")
    print("=" * 60)
    
    # Initialize and train model
    model = SpotifyLinearRegression(csv_path='dataset.csv')
    model.load_data()
    model.preprocess_data(target_column='popularity')
    model.split_data(test_size=0.2, random_state=42)
    model.scale_features()
    model.train_model(use_scaled=True)
    
    # Create sample new data
    # Features: danceability, energy, valence, tempo, loudness, Duration_min, explicit
    new_track = pd.DataFrame({
        'danceability': [0.75],
        'energy': [0.65],
        'valence': [0.80],
        'tempo': [120.0],
        'loudness': [-5.0],
        'Duration_min': [3.5],
        'explicit': [0]  # 0 for False, 1 for True
    })
    
    print("\nNew track features:")
    print(new_track)
    
    # Make prediction
    prediction = model.model.predict(model.scaler.transform(new_track))
    
    print(f"\nPredicted popularity: {prediction[0]:.2f}")
    
    # Example with multiple tracks
    multiple_tracks = pd.DataFrame({
        'danceability': [0.75, 0.50, 0.90],
        'energy': [0.65, 0.40, 0.85],
        'valence': [0.80, 0.30, 0.95],
        'tempo': [120.0, 90.0, 140.0],
        'loudness': [-5.0, -10.0, -3.0],
        'Duration_min': [3.5, 4.0, 3.0],
        'explicit': [0, 1, 0]
    })
    
    print("\nMultiple tracks features:")
    print(multiple_tracks)
    
    predictions = model.model.predict(model.scaler.transform(multiple_tracks))
    
    print("\nPredicted popularities:")
    for i, pred in enumerate(predictions):
        print(f"  Track {i+1}: {pred:.2f}")


def example_model_comparison():
    """Compare scaled vs unscaled features"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Comparing Scaled vs Unscaled Features")
    print("=" * 60)
    
    # Model with scaled features
    model_scaled = SpotifyLinearRegression(csv_path='dataset.csv')
    model_scaled.load_data()
    model_scaled.preprocess_data(target_column='popularity')
    model_scaled.split_data(test_size=0.2, random_state=42)
    model_scaled.scale_features()
    model_scaled.train_model(use_scaled=True)
    metrics_scaled = model_scaled.validate_model(use_scaled=True, cv_folds=5)
    
    # Model with unscaled features
    model_unscaled = SpotifyLinearRegression(csv_path='dataset.csv')
    model_unscaled.load_data()
    model_unscaled.preprocess_data(target_column='popularity')
    model_unscaled.split_data(test_size=0.2, random_state=42)
    model_unscaled.train_model(use_scaled=False)
    metrics_unscaled = model_unscaled.validate_model(use_scaled=False, cv_folds=5)
    
    print("\nComparison Results:")
    print(f"\nScaled Features:")
    print(f"  Test R²: {metrics_scaled['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics_scaled['test_rmse']:.4f}")
    
    print(f"\nUnscaled Features:")
    print(f"  Test R²: {metrics_unscaled['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics_unscaled['test_rmse']:.4f}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    # Uncomment to run other examples:
    # example_step_by_step()
    # example_predict_new_data()
    # example_model_comparison()
