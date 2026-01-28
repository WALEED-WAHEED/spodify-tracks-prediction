"""
Data Analysis Script for Canadian Job Market Dataset
Analyzes data structure, quality, and relationships before modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("COMPREHENSIVE DATA ANALYSIS")
print("=" * 80)

# Load data
print("\n1. LOADING DATA...")
try:
    df = pd.read_csv('job-bank-open-data-all-job-postings-en-december2025.csv', 
                     sep='\t', encoding='utf-16')
    print(f"[OK] Successfully loaded dataset")
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    exit(1)

print(f"\nDataset Shape: {df.shape}")
print(f"  - Rows: {df.shape[0]:,}")
print(f"  - Columns: {df.shape[1]}")

# Basic structure
print("\n" + "=" * 80)
print("2. DATA STRUCTURE")
print("=" * 80)
print(f"\nColumn Names ({len(df.columns)} total):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nData Types:")
print(df.dtypes.value_counts())

# Missing values analysis
print("\n" + "=" * 80)
print("3. MISSING VALUES ANALYSIS")
print("=" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing Count', ascending=False)

print("\nTop 20 columns with most missing values:")
print(missing_df[missing_df['Missing Count'] > 0].head(20).to_string())

# Target variable analysis
print("\n" + "=" * 80)
print("4. TARGET VARIABLE ANALYSIS")
print("=" * 80)

if 'Vacancy Count' in df.columns:
    print("\nVacancy Count:")
    print(df['Vacancy Count'].describe())
    print(f"\nUnique values: {df['Vacancy Count'].nunique()}")
    print(f"\nValue distribution:")
    print(df['Vacancy Count'].value_counts().head(15))
    print(f"\nMissing values: {df['Vacancy Count'].isnull().sum()} ({df['Vacancy Count'].isnull().sum()/len(df)*100:.2f}%)")

# Salary analysis
print("\n" + "=" * 80)
print("5. SALARY ANALYSIS")
print("=" * 80)

if 'Salary Minimum' in df.columns:
    smin = pd.to_numeric(df['Salary Minimum'], errors='coerce')
    print("\nSalary Minimum:")
    print(smin.describe())
    print(f"Missing: {smin.isnull().sum()} ({smin.isnull().sum()/len(df)*100:.2f}%)")
    print(f"Zero/negative: {(smin <= 0).sum()}")

if 'Salary Maximum' in df.columns:
    smax = pd.to_numeric(df['Salary Maximum'], errors='coerce')
    print("\nSalary Maximum:")
    print(smax.describe())
    print(f"Missing: {smax.isnull().sum()} ({smax.isnull().sum()/len(df)*100:.2f}%)")
    print(f"Zero/negative: {(smax <= 0).sum()}")

if 'Salary Minimum' in df.columns and 'Salary Maximum' in df.columns:
    smin = pd.to_numeric(df['Salary Minimum'], errors='coerce')
    smax = pd.to_numeric(df['Salary Maximum'], errors='coerce')
    srange = smax - smin
    print("\nSalary Range (Max - Min):")
    print(srange.describe())
    print(f"Negative ranges (Max < Min): {(srange < 0).sum()}")

# Check for salary condition (hourly vs annual)
salary_cols = [c for c in df.columns if 'Salary' in c]
print(f"\nAll Salary-related columns: {salary_cols}")

if 'Salary Condition' in df.columns:
    print("\nSalary Condition (Hourly/Annual):")
    print(df['Salary Condition'].value_counts())

# Calculate annual salary estimate
if 'Salary Minimum' in df.columns and 'Salary Maximum' in df.columns:
    smin = pd.to_numeric(df['Salary Minimum'], errors='coerce')
    smax = pd.to_numeric(df['Salary Maximum'], errors='coerce')
    
    # Assume hourly if < 100, annual if >= 100
    # Convert hourly to annual: hourly * 40 hours/week * 52 weeks/year
    hourly_threshold = 100
    
    smin_annual = smin.copy()
    smin_hourly_mask = (smin < hourly_threshold) & (smin > 0)
    smin_annual[smin_hourly_mask] = smin[smin_hourly_mask] * 40 * 52
    
    smax_annual = smax.copy()
    smax_hourly_mask = (smax < hourly_threshold) & (smax > 0)
    smax_annual[smax_hourly_mask] = smax[smax_hourly_mask] * 40 * 52
    
    # Use average of min and max
    salary_annual = (smin_annual + smax_annual) / 2
    
    print("\nEstimated Annual Salary (after conversion):")
    print(salary_annual.describe())
    print(f"Valid values: {salary_annual.notna().sum()}")
    print(f"Missing: {salary_annual.isnull().sum()}")

# Categorical features analysis
print("\n" + "=" * 80)
print("6. CATEGORICAL FEATURES ANALYSIS")
print("=" * 80)

if 'Province/Territory' in df.columns:
    print("\nProvince/Territory:")
    print(df['Province/Territory'].value_counts())
    print(f"Missing: {df['Province/Territory'].isnull().sum()}")

if 'Employment Type' in df.columns:
    print("\nEmployment Type:")
    print(df['Employment Type'].value_counts())
    print(f"Missing: {df['Employment Type'].isnull().sum()}")

if 'Employment Term' in df.columns:
    print("\nEmployment Term:")
    print(df['Employment Term'].value_counts())
    print(f"Missing: {df['Employment Term'].isnull().sum()}")

if 'Education LOS' in df.columns:
    print("\nEducation LOS (Level of Study):")
    print(df['Education LOS'].value_counts())
    print(f"Missing: {df['Education LOS'].isnull().sum()}")

if 'Experience Level' in df.columns:
    print("\nExperience Level:")
    print(df['Experience Level'].value_counts())
    print(f"Missing: {df['Experience Level'].isnull().sum()}")

# NOC and NAICS analysis
print("\n" + "=" * 80)
print("7. OCCUPATION & INDUSTRY ANALYSIS")
print("=" * 80)

if 'NOC21 Code' in df.columns:
    print("\nNOC21 Code (Top 20):")
    print(df['NOC21 Code'].value_counts().head(20))
    print(f"Unique codes: {df['NOC21 Code'].nunique()}")
    print(f"Missing: {df['NOC21 Code'].isnull().sum()}")
    
    # Extract major group (first 2 digits)
    noc_major = df['NOC21 Code'].astype(str).str[:2]
    print(f"\nNOC Major Groups (first 2 digits):")
    print(noc_major.value_counts().head(15))

if 'NAICS' in df.columns:
    print("\nNAICS Code (Top 20):")
    print(df['NAICS'].value_counts().head(20))
    print(f"Unique codes: {df['NAICS'].nunique()}")
    print(f"Missing: {df['NAICS'].isnull().sum()}")
    
    # Extract sector (first 2 digits)
    naics_sector = df['NAICS'].astype(str).str[:2]
    print(f"\nNAICS Sectors (first 2 digits):")
    print(naics_sector.value_counts().head(15))

# Hours analysis
print("\n" + "=" * 80)
print("8. WORK HOURS ANALYSIS")
print("=" * 80)

if 'Hours Minimum' in df.columns:
    hmin = pd.to_numeric(df['Hours Minimum'], errors='coerce')
    print("\nHours Minimum:")
    print(hmin.describe())
    print(f"Missing: {hmin.isnull().sum()}")

if 'Hours Maximum' in df.columns:
    hmax = pd.to_numeric(df['Hours Maximum'], errors='coerce')
    print("\nHours Maximum:")
    print(hmax.describe())
    print(f"Missing: {hmax.isnull().sum()}")

# Feature relationships
print("\n" + "=" * 80)
print("9. FEATURE RELATIONSHIPS")
print("=" * 80)

# Check correlation between numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 0:
    print(f"\nNumeric columns: {numeric_cols}")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        print("\nCorrelation matrix (showing strong correlations |r| > 0.5):")
        strong_corr = corr[(corr.abs() > 0.5) & (corr != 1.0)]
        if not strong_corr.isnull().all().all():
            print(strong_corr.dropna(how='all').dropna(axis=1, how='all'))

# Data quality summary
print("\n" + "=" * 80)
print("10. DATA QUALITY SUMMARY")
print("=" * 80)

total_rows = len(df)
print(f"\nTotal rows: {total_rows:,}")

# Count rows with complete data for key features
key_features = ['Province/Territory', 'Employment Type', 'NOC21 Code', 
                'NAICS', 'Salary Minimum', 'Salary Maximum']
available_key_features = [f for f in key_features if f in df.columns]

if available_key_features:
    complete_rows = df[available_key_features].notna().all(axis=1).sum()
    print(f"Rows with all key features: {complete_rows:,} ({complete_rows/total_rows*100:.2f}%)")
    print(f"Rows missing at least one key feature: {total_rows - complete_rows:,} ({(total_rows-complete_rows)/total_rows*100:.2f}%)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
