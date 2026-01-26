# Spotify Tracks Popularity Prediction Model

A machine learning model that predicts the popularity of Spotify tracks using linear regression. This model analyzes audio features and metadata to estimate track popularity scores.

## 📋 Overview

This project implements a Linear Regression model to predict Spotify track popularity based on:
- **Audio Features**: Danceability, Energy, Valence, Tempo, Loudness, Duration
- **Metadata**: Explicit content flag

The model uses scikit-learn's Linear Regression with feature scaling for optimal performance.

## 📁 Project Structure

```
spodify tracks Prediction/
│
├── linear_regression_modeling.py  # Main model implementation
├── example_usage.py               # Usage examples and demonstrations
├── dataset.csv                    # Spotify tracks dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
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
- Load and preprocess the data
- Split into training and test sets
- Scale features
- Train the model
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
from linear_regression_modeling import SpotifyLinearRegression

# Initialize the model
model = SpotifyLinearRegression(csv_path='dataset.csv')

# Run the complete pipeline
metrics = model.run_full_pipeline(
    target_column='popularity',
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
from linear_regression_modeling import SpotifyLinearRegression
import pandas as pd

# Initialize
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

# Step 7: Make predictions on new data
new_track = pd.DataFrame({
    'danceability': [0.75],
    'energy': [0.65],
    'valence': [0.80],
    'tempo': [120.0],
    'loudness': [-5.0],
    'Duration_min': [3.5],
    'explicit': [0]  # 0 for False, 1 for True
})

prediction = model.model.predict(model.scaler.transform(new_track))
print(f"Predicted popularity: {prediction[0]:.2f}")
```

### Making Predictions on New Data

```python
import pandas as pd
from linear_regression_modeling import SpotifyLinearRegression

# Train the model first
model = SpotifyLinearRegression(csv_path='dataset.csv')
model.load_data()
model.preprocess_data(target_column='popularity')
model.split_data(test_size=0.2, random_state=42)
model.scale_features()
model.train_model(use_scaled=True)

# Prepare new track data
new_tracks = pd.DataFrame({
    'danceability': [0.75, 0.50, 0.90],
    'energy': [0.65, 0.40, 0.85],
    'valence': [0.80, 0.30, 0.95],
    'tempo': [120.0, 90.0, 140.0],
    'loudness': [-5.0, -10.0, -3.0],
    'Duration_min': [3.5, 4.0, 3.0],
    'explicit': [0, 1, 0]
})

# Make predictions
predictions = model.model.predict(model.scaler.transform(new_tracks))
print("Predicted popularities:", predictions)
```

## 📊 Model Features

### Features Used

The model uses the following features:

1. **danceability** (0.0-1.0): How suitable a track is for dancing
2. **energy** (0.0-1.0): Perceptual measure of intensity and power
3. **valence** (0.0-1.0): Musical positiveness conveyed by a track
4. **tempo** (BPM): Overall estimated tempo of a track
5. **loudness** (dB): Overall loudness of a track
6. **Duration_min**: Track duration in minutes
7. **explicit** (0/1): Whether the track contains explicit content

### Model Performance

The model provides:
- **R² Score**: Coefficient of determination (how well the model fits)
- **RMSE**: Root Mean Squared Error (prediction error)
- **MAE**: Mean Absolute Error (average prediction error)
- **Cross-Validation**: K-fold cross-validation for robust evaluation

### Visualizations

The model generates:
- Feature coefficient plots
- Actual vs Predicted scatter plots
- Residual plots
- Feature importance charts

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

- `target_column`: Name of the target variable (default: 'popularity')
- `test_size`: Proportion of data for testing (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)
- `use_scaled`: Whether to use scaled features (default: True)
- `cv_folds`: Number of cross-validation folds (default: 5)

### Feature Scaling

The model uses StandardScaler by default, which:
- Centers features to have mean = 0
- Scales features to have standard deviation = 1
- Improves model performance and convergence

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
  - Penalizes large errors more than small ones

- **MAE**: 
  - Lower is better
  - Average absolute difference between predictions and actual values
  - Easier to interpret than RMSE

### Interpreting Coefficients

The model shows feature coefficients indicating:
- **Positive coefficient**: Feature increases popularity
- **Negative coefficient**: Feature decreases popularity
- **Larger absolute value**: Stronger influence on popularity

## 🛠️ Troubleshooting

### Common Issues

1. **FileNotFoundError: dataset.csv**
   - Ensure `dataset.csv` is in the same directory as the script
   - Check the file path in the `csv_path` parameter

2. **Missing dependencies**
   - Run: `pip install -r requirements.txt`
   - Ensure you're using the correct Python version

3. **Memory errors with large datasets**
   - Reduce dataset size or use data sampling
   - Close other applications to free memory

4. **Import errors**
   - Activate your virtual environment
   - Reinstall packages: `pip install --upgrade -r requirements.txt`

## 📝 Notes

- The model assumes your dataset has the required columns
- Ensure `explicit` column is boolean or can be converted to 0/1
- Duration is automatically converted from milliseconds to minutes
- Missing values are automatically removed during preprocessing

## 📧 Support

For questions or issues, please refer to the code documentation or contact the development team.

## 📄 License

This project is provided as-is for client use.

---

**Version**: 1.0  
**Last Updated**: January 2026
