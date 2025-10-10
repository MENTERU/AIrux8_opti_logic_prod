#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Power Data Analysis Script
==========================
Analyzes the power meter data to check for negative values and understand
the statistical distribution of power consumption by unit.

This script will help identify if negative power predictions are due to:
1. Training data containing negative values
2. Model prediction issues
3. Data preprocessing problems
"""

import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_power_data(file_path):
    """
    Analyze power meter data for negative values and statistical summary

    Args:
        file_path (str): Path to the power meter CSV file
    """
    print("=" * 80)
    print("POWER DATA ANALYSIS")
    print("=" * 80)

    # Load the data
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Display basic info
    print(f"\n📊 Basic Information:")
    print(f"   Total records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

    # Check for required columns
    required_cols = ["Mesh ID", "PM Addr ID", "Total_kWh"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return

    # Create unit identifier
    df["Unit_ID"] = df["Mesh ID"].astype(str) + "-" + df["PM Addr ID"].astype(str)

    # Basic statistics for Total_kWh
    print(f"\n📈 Total_kWh Statistics (All Data):")
    print(f"   Count: {df['Total_kWh'].count():,}")
    print(f"   Mean: {df['Total_kWh'].mean():.4f} kWh")
    print(f"   Median: {df['Total_kWh'].median():.4f} kWh")
    print(f"   Std: {df['Total_kWh'].std():.4f} kWh")
    print(f"   Min: {df['Total_kWh'].min():.4f} kWh")
    print(f"   Max: {df['Total_kWh'].max():.4f} kWh")

    # Check for negative values
    negative_mask = df["Total_kWh"] < 0
    negative_count = negative_mask.sum()
    negative_percentage = (negative_count / len(df)) * 100

    print(f"\n🔍 Negative Values Analysis:")
    print(f"   Negative values count: {negative_count:,}")
    print(f"   Negative values percentage: {negative_percentage:.2f}%")

    if negative_count > 0:
        print(f"   Most negative value: {df['Total_kWh'].min():.4f} kWh")
        print(
            f"   Least negative value: {df[negative_mask]['Total_kWh'].max():.4f} kWh"
        )

        # Analyze negative values by unit
        print(f"\n📋 Units with Negative Values:")
        negative_by_unit = (
            df[negative_mask]
            .groupby("Unit_ID")["Total_kWh"]
            .agg(["count", "min", "max", "mean"])
            .round(4)
        )
        negative_by_unit = negative_by_unit.sort_values("count", ascending=False)

        print(f"   Total units with negative values: {len(negative_by_unit)}")
        print(f"   Top 10 units with most negative records:")
        print(negative_by_unit.head(10).to_string())

        # Show some examples of negative values
        print(f"\n🔍 Sample Negative Value Records:")
        sample_negative = df[negative_mask].head(10)[
            ["Unit_ID", "Datetime", "Total_kWh"]
        ]
        print(sample_negative.to_string(index=False))

    # Analyze by unit
    print(f"\n📊 Analysis by Unit:")
    unit_stats = (
        df.groupby("Unit_ID")["Total_kWh"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )

    # Add negative count per unit
    negative_by_unit_count = df[negative_mask].groupby("Unit_ID").size()
    unit_stats["negative_count"] = (
        unit_stats.index.map(negative_by_unit_count).fillna(0).astype(int)
    )
    unit_stats["negative_percentage"] = (
        unit_stats["negative_count"] / unit_stats["count"] * 100
    ).round(2)

    print(f"   Total unique units: {len(unit_stats)}")
    print(f"   Units with negative values: {(unit_stats['negative_count'] > 0).sum()}")

    # Show units with most negative values
    if negative_count > 0:
        print(f"\n📋 Units with Most Negative Values:")
        top_negative_units = (
            unit_stats[unit_stats["negative_count"] > 0]
            .sort_values("negative_count", ascending=False)
            .head(10)
        )
        print(top_negative_units.to_string())

    # Show units with highest and lowest average power
    print(f"\n📋 Units with Highest Average Power:")
    top_power_units = unit_stats.sort_values("mean", ascending=False).head(10)
    print(top_power_units[["count", "mean", "std", "min", "max"]].to_string())

    print(f"\n📋 Units with Lowest Average Power:")
    low_power_units = unit_stats.sort_values("mean", ascending=True).head(10)
    print(low_power_units[["count", "mean", "std", "min", "max"]].to_string())

    # Check for zero values
    zero_mask = df["Total_kWh"] == 0
    zero_count = zero_mask.sum()
    zero_percentage = (zero_count / len(df)) * 100

    print(f"\n🔍 Zero Values Analysis:")
    print(f"   Zero values count: {zero_count:,}")
    print(f"   Zero values percentage: {zero_percentage:.2f}%")

    # Check for very small positive values (potential measurement errors)
    small_positive_mask = (df["Total_kWh"] > 0) & (df["Total_kWh"] < 0.01)
    small_positive_count = small_positive_mask.sum()
    small_positive_percentage = (small_positive_count / len(df)) * 100

    print(f"\n🔍 Very Small Positive Values (< 0.01 kWh):")
    print(f"   Small positive values count: {small_positive_count:,}")
    print(f"   Small positive values percentage: {small_positive_percentage:.2f}%")

    # Summary recommendations
    print(f"\n💡 Summary and Recommendations:")
    if negative_count > 0:
        print(
            f"   ⚠️  WARNING: Found {negative_count:,} negative power values ({negative_percentage:.2f}%)"
        )
        print(f"   📝 This could explain negative predictions from the model")
        print(f"   🔧 Consider:")
        print(f"      - Reviewing data collection process")
        print(f"      - Filtering out negative values during preprocessing")
        print(f"      - Investigating units with most negative values")
    else:
        print(f"   ✅ No negative values found in training data")
        print(f"   📝 Negative predictions are likely due to model issues")
        print(f"   🔧 Consider:")
        print(f"      - Reviewing model training process")
        print(f"      - Adding constraints to prevent negative predictions")
        print(f"      - Checking feature engineering")

    if zero_count > 0:
        print(f"   📊 Found {zero_count:,} zero values ({zero_percentage:.2f}%)")
        print(f"   📝 These might represent OFF states or measurement gaps")

    if small_positive_count > 0:
        print(f"   📊 Found {small_positive_count:,} very small positive values")
        print(f"   📝 These might be measurement noise or standby power")

    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def check_adjusted_power_negatives(file_path):
    """
    Simple check for negative values in adjusted_power column of features file

    Args:
        file_path (str): Path to the features processed CSV file
    """
    print("\n" + "=" * 80)
    print("ADJUSTED POWER NEGATIVE VALUES CHECK")
    print("=" * 80)

    # Load the data
    print(f"Loading features data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Features data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading features data: {e}")
        return

    # Check if adjusted_power column exists
    if "adjusted_power" not in df.columns:
        print(f"❌ 'adjusted_power' column not found in the data")
        print(f"Available columns: {list(df.columns)}")
        return

    # Basic statistics for adjusted_power
    print(f"\n📈 adjusted_power Statistics:")
    print(f"   Count: {df['adjusted_power'].count():,}")
    print(f"   Mean: {df['adjusted_power'].mean():.4f}")
    print(f"   Median: {df['adjusted_power'].median():.4f}")
    print(f"   Std: {df['adjusted_power'].std():.4f}")
    print(f"   Min: {df['adjusted_power'].min():.4f}")
    print(f"   Max: {df['adjusted_power'].max():.4f}")

    # Check for negative values
    negative_mask = df["adjusted_power"] < 0
    negative_count = negative_mask.sum()
    negative_percentage = (negative_count / len(df)) * 100

    print(f"\n🔍 Negative Values in adjusted_power:")
    print(f"   Negative values count: {negative_count:,}")
    print(f"   Negative values percentage: {negative_percentage:.2f}%")

    if negative_count > 0:
        print(f"   Most negative value: {df['adjusted_power'].min():.4f}")
        print(
            f"   Least negative value: {df[negative_mask]['adjusted_power'].max():.4f}"
        )

        # Show some examples of negative values
        print(f"\n🔍 Sample Negative adjusted_power Records:")
        sample_negative = df[negative_mask].head(10)[
            ["Datetime", "zone", "adjusted_power"]
        ]
        print(sample_negative.to_string(index=False))

        # Analyze negative values by zone
        print(f"\n📋 Negative adjusted_power by Zone:")
        negative_by_zone = (
            df[negative_mask]
            .groupby("zone")["adjusted_power"]
            .agg(["count", "min", "max", "mean"])
            .round(4)
        )
        negative_by_zone = negative_by_zone.sort_values("count", ascending=False)
        print(negative_by_zone.to_string())

        print(
            f"\n⚠️  WARNING: Found {negative_count:,} negative adjusted_power values ({negative_percentage:.2f}%)"
        )
        print(f"📝 This could cause issues in model training and predictions")
    else:
        print(f"\n✅ No negative values found in adjusted_power column")
        print(f"📝 The adjusted_power data is clean for model training")

    print(f"\n" + "=" * 80)
    print("ADJUSTED POWER CHECK COMPLETE")
    print("=" * 80)


def analyze_planning_data(planning_dir):
    """
    Analyze planning data (optimization results) for negative power predictions

    Args:
        planning_dir (str): Path to the planning data directory
    """
    print("\n" + "=" * 80)
    print("PLANNING DATA NEGATIVE POWER PREDICTIONS ANALYSIS")
    print("=" * 80)

    # Get today's date in the format used in filenames
    today = datetime.now().strftime("%Y%m%d")
    control_schedule_file = os.path.join(
        planning_dir, f"control_type_schedule_{today}.csv"
    )

    print(f"Looking for planning file: {control_schedule_file}")

    # Check if file exists
    if not os.path.exists(control_schedule_file):
        print(f"❌ Planning file not found: {control_schedule_file}")

        # Try to find any control_type_schedule file in the directory
        if os.path.exists(planning_dir):
            files = [
                f
                for f in os.listdir(planning_dir)
                if f.startswith("control_type_schedule_") and f.endswith(".csv")
            ]
            if files:
                # Get the most recent file
                files.sort(reverse=True)
                control_schedule_file = os.path.join(planning_dir, files[0])
                print(f"📁 Found most recent planning file: {control_schedule_file}")
            else:
                print(f"❌ No control_type_schedule files found in: {planning_dir}")
                return
        else:
            print(f"❌ Planning directory not found: {planning_dir}")
            return

    # Load the planning data
    print(f"Loading planning data from: {control_schedule_file}")
    try:
        df = pd.read_csv(control_schedule_file)
        print(f"✅ Planning data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading planning data: {e}")
        return

    # Display basic info
    print(f"\n📊 Planning Data Information:")
    print(f"   Total records: {len(df):,}")
    print(f"   Date range: {df['Date Time'].min()} to {df['Date Time'].max()}")

    # Find all power prediction columns
    power_columns = [col for col in df.columns if col.endswith("_PredPower")]
    print(f"   Power prediction columns found: {len(power_columns)}")
    print(f"   Columns: {power_columns}")

    if not power_columns:
        print(f"❌ No power prediction columns found in the data")
        return

    # Analyze each power prediction column
    total_negative_count = 0
    total_predictions = 0

    print(f"\n📈 Power Predictions Analysis by Zone:")
    print("-" * 80)

    for power_column in power_columns:
        zone_name = power_column.replace("_PredPower", "")

        # Basic statistics
        power_data = df[power_column]
        count = power_data.count()
        mean_val = power_data.mean()
        median_val = power_data.median()
        std_val = power_data.std()
        min_val = power_data.min()
        max_val = power_data.max()

        # Check for negative values
        negative_mask = power_data < 0
        negative_count = negative_mask.sum()
        negative_percentage = (negative_count / count) * 100 if count > 0 else 0

        total_negative_count += negative_count
        total_predictions += count

        print(f"\n🏢 Zone: {zone_name}")
        print(f"   Count: {count:,}")
        print(f"   Mean: {mean_val:.4f}")
        print(f"   Median: {median_val:.4f}")
        print(f"   Std: {std_val:.4f}")
        print(f"   Min: {min_val:.4f}")
        print(f"   Max: {max_val:.4f}")
        print(f"   Negative values: {negative_count:,} ({negative_percentage:.2f}%)")

        if negative_count > 0:
            print(f"   ⚠️  WARNING: Found negative predictions!")

            # Show some examples of negative predictions
            negative_records = df[negative_mask][["Date Time", power_column]]
            print(f"   📋 Sample negative predictions:")
            print(negative_records.head(5).to_string(index=False))

    # Overall summary
    overall_negative_percentage = (
        (total_negative_count / total_predictions) * 100 if total_predictions > 0 else 0
    )

    print(f"\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total predictions analyzed: {total_predictions:,}")
    print(f"Total negative predictions: {total_negative_count:,}")
    print(f"Overall negative percentage: {overall_negative_percentage:.2f}%")

    if total_negative_count > 0:
        print(
            f"\n⚠️  WARNING: Found {total_negative_count:,} negative power predictions!"
        )
        print(
            f"📝 This indicates issues in the optimization logic or model predictions"
        )
        print(f"🔧 Recommendations:")
        print(f"   - Review model training and validation")
        print(f"   - Add constraints to prevent negative predictions in optimization")
        print(f"   - Check feature engineering and data preprocessing")
        print(f"   - Investigate specific zones with most negative predictions")
    else:
        print(f"\n✅ No negative power predictions found in planning data")
        print(f"📝 The optimization results are clean")

    print(f"\n" + "=" * 80)
    print("PLANNING DATA ANALYSIS COMPLETE")
    print("=" * 80)


def analyze_validation_results(validation_dir):
    """
    Analyze validation results files for negative power predictions

    Args:
        validation_dir (str): Path to the validation results directory
    """
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS NEGATIVE POWER PREDICTIONS ANALYSIS")
    print("=" * 80)

    # Find all CSV files in the validation directory
    csv_files = glob.glob(os.path.join(validation_dir, "*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in: {validation_dir}")
        return

    print(f"📁 Found {len(csv_files)} validation files:")
    for file_path in csv_files:
        print(f"   - {os.path.basename(file_path)}")

    # Analyze each validation file
    total_negative_count = 0
    total_predictions = 0
    files_with_negatives = 0

    print(f"\n📈 Validation Results Analysis by File:")
    print("-" * 80)

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        zone_name = filename.replace("valid_results_", "").replace(".csv", "")

        print(f"\n🏢 Zone: {zone_name}")
        print(f"📄 File: {filename}")

        # Load the validation data
        try:
            df = pd.read_csv(file_path)
            print(f"   ✅ Loaded successfully. Shape: {df.shape}")
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
            continue

        # Find power prediction columns (ending with _power_pred or similar)
        power_columns = [
            col
            for col in df.columns
            if col.endswith("_power_pred")
            or col.endswith("_PredPower")
            or "power" in col.lower()
            and "pred" in col.lower()
        ]

        if not power_columns:
            print(f"   ⚠️  No power prediction columns found")
            print(f"   Available columns: {list(df.columns)}")
            continue

        print(f"   📊 Power prediction columns: {power_columns}")

        # Analyze each power prediction column
        file_negative_count = 0
        file_total_predictions = 0

        for power_column in power_columns:
            # Basic statistics
            power_data = df[power_column]
            count = power_data.count()
            mean_val = power_data.mean()
            median_val = power_data.median()
            std_val = power_data.std()
            min_val = power_data.min()
            max_val = power_data.max()

            # Check for negative values
            negative_mask = power_data < 0
            negative_count = negative_mask.sum()
            negative_percentage = (negative_count / count) * 100 if count > 0 else 0

            file_negative_count += negative_count
            file_total_predictions += count

            print(f"\n   📈 Column: {power_column}")
            print(f"      Count: {count:,}")
            print(f"      Mean: {mean_val:.4f}")
            print(f"      Median: {median_val:.4f}")
            print(f"      Std: {std_val:.4f}")
            print(f"      Min: {min_val:.4f}")
            print(f"      Max: {max_val:.4f}")
            print(
                f"      Negative values: {negative_count:,} ({negative_percentage:.2f}%)"
            )

            if negative_count > 0:
                print(f"      ⚠️  WARNING: Found negative predictions!")

                # Show some examples of negative predictions
                negative_records = df[negative_mask][[power_column]].head(5)
                print(f"      📋 Sample negative predictions:")
                print(negative_records.to_string(index=False))

        # File summary
        file_negative_percentage = (
            (file_negative_count / file_total_predictions) * 100
            if file_total_predictions > 0
            else 0
        )

        print(f"\n   📊 File Summary:")
        print(f"      Total predictions: {file_total_predictions:,}")
        print(f"      Total negative predictions: {file_negative_count:,}")
        print(f"      Negative percentage: {file_negative_percentage:.2f}%")

        if file_negative_count > 0:
            files_with_negatives += 1
            print(f"      ⚠️  This file contains negative predictions!")

        total_negative_count += file_negative_count
        total_predictions += file_total_predictions

    # Overall summary
    overall_negative_percentage = (
        (total_negative_count / total_predictions) * 100 if total_predictions > 0 else 0
    )

    print(f"\n" + "=" * 80)
    print("VALIDATION RESULTS OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total files analyzed: {len(csv_files)}")
    print(f"Files with negative predictions: {files_with_negatives}")
    print(f"Total predictions analyzed: {total_predictions:,}")
    print(f"Total negative predictions: {total_negative_count:,}")
    print(f"Overall negative percentage: {overall_negative_percentage:.2f}%")

    if total_negative_count > 0:
        print(
            f"\n⚠️  WARNING: Found {total_negative_count:,} negative power predictions in validation results!"
        )
        print(
            f"📝 This indicates the models themselves are producing negative predictions"
        )
        print(f"🔧 This explains why optimization results have negative values")
        print(f"🔧 Recommendations:")
        print(f"   - Review model training process")
        print(f"   - Add constraints to prevent negative predictions in models")
        print(f"   - Check feature engineering and scaling")
        print(f"   - Consider using models that naturally produce positive outputs")
        print(f"   - Add post-processing to clip negative predictions to zero")
    else:
        print(f"\n✅ No negative power predictions found in validation results")
        print(f"📝 Models are producing clean predictions")
        print(f"📝 Negative values in optimization might be from other sources")

    print(f"\n" + "=" * 80)
    print("VALIDATION RESULTS ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    """Main function"""
    # Path to the power meter data
    power_file = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/02_PreprocessedData/Clea/power_meter_processed_Clea.csv"

    # Path to the features processed data
    features_file = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/02_PreprocessedData/Clea/features_processed_Clea.csv"

    # Path to the planning data directory
    planning_dir = (
        "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/04_PlanningData/Clea"
    )

    # Path to the validation results directory
    validation_dir = (
        "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/05_ValidationResults/Clea"
    )

    # Check if files exist
    if not os.path.exists(power_file):
        print(f"❌ Power file not found: {power_file}")
        return

    if not os.path.exists(features_file):
        print(f"❌ Features file not found: {features_file}")
        return

    if not os.path.exists(planning_dir):
        print(f"❌ Planning directory not found: {planning_dir}")
        return

    if not os.path.exists(validation_dir):
        print(f"❌ Validation directory not found: {validation_dir}")
        return

    # Run power data analysis
    analyze_power_data(power_file)

    # Run adjusted_power negative values check
    check_adjusted_power_negatives(features_file)

    # Run planning data analysis
    analyze_planning_data(planning_dir)

    # Run validation results analysis
    analyze_validation_results(validation_dir)


if __name__ == "__main__":
    main()
