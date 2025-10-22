#!/usr/bin/env python3
"""
Optimization Power Analysis Script

This script analyzes power consumption predictions in the optimization output CSV
to compare AC ON vs AC OFF power consumption and identify potential issues.

Usage:
    python analyze_optimization_power.py

Output:
    - Terminal analysis showing power consumption patterns
    - Comparison between AC ON and AC OFF predictions
    - Identification of unrealistic power predictions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_optimization_data():
    """Load the optimization output CSV data"""
    csv_path = Path("data/04_PlanningData/Clea/control_type_schedule_20251018.csv")

    if not csv_path.exists():
        print(f"❌ Error: Optimization CSV file not found at {csv_path}")
        print(
            "Please ensure you're running this script from the project root directory"
        )
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded optimization data: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading optimization data: {e}")
        sys.exit(1)


def analyze_power_by_zone_and_status(df):
    """Analyze power consumption by zone and AC status"""

    # Get all zone columns
    zone_columns = [col for col in df.columns if "_PredPower" in col]
    zones = [col.replace("_PredPower", "") for col in zone_columns]

    results = []

    for zone in zones:
        power_col = f"{zone}_PredPower"
        onoff_col = f"{zone}_OnOFF"

        if power_col not in df.columns or onoff_col not in df.columns:
            print(f"⚠️  Warning: Missing columns for zone {zone}")
            continue

        # Get power data for this zone
        zone_power_data = df[power_col].dropna()
        zone_onoff_data = df[onoff_col].dropna()

        # Create a combined dataframe for this zone
        zone_df = pd.DataFrame({"power": zone_power_data, "onoff": zone_onoff_data})

        # AC OFF data (OnOFF = "OFF" or empty)
        ac_off_mask = (zone_df["onoff"] == "OFF") | (zone_df["onoff"].isna())
        ac_off_data = zone_df[ac_off_mask]

        # AC ON data (OnOFF = "ON")
        ac_on_mask = zone_df["onoff"] == "ON"
        ac_on_data = zone_df[ac_on_mask]

        # Calculate statistics for AC OFF
        if len(ac_off_data) > 0:
            ac_off_stats = {
                "zone": zone,
                "status": "AC OFF",
                "count": len(ac_off_data),
                "mean_power": ac_off_data["power"].mean(),
                "median_power": ac_off_data["power"].median(),
                "min_power": ac_off_data["power"].min(),
                "max_power": ac_off_data["power"].max(),
                "std_power": ac_off_data["power"].std(),
                "q25_power": ac_off_data["power"].quantile(0.25),
                "q75_power": ac_off_data["power"].quantile(0.75),
            }
        else:
            ac_off_stats = {
                "zone": zone,
                "status": "AC OFF",
                "count": 0,
                "mean_power": 0,
                "median_power": 0,
                "min_power": 0,
                "max_power": 0,
                "std_power": 0,
                "q25_power": 0,
                "q75_power": 0,
            }

        # Calculate statistics for AC ON
        if len(ac_on_data) > 0:
            ac_on_stats = {
                "zone": zone,
                "status": "AC ON",
                "count": len(ac_on_data),
                "mean_power": ac_on_data["power"].mean(),
                "median_power": ac_on_data["power"].median(),
                "min_power": ac_on_data["power"].min(),
                "max_power": ac_on_data["power"].max(),
                "std_power": ac_on_data["power"].std(),
                "q25_power": ac_on_data["power"].quantile(0.25),
                "q75_power": ac_on_data["power"].quantile(0.75),
            }
        else:
            ac_on_stats = {
                "zone": zone,
                "status": "AC ON",
                "count": 0,
                "mean_power": 0,
                "median_power": 0,
                "min_power": 0,
                "max_power": 0,
                "std_power": 0,
                "q25_power": 0,
                "q75_power": 0,
            }

        results.extend([ac_off_stats, ac_on_stats])

    return pd.DataFrame(results)


def print_optimization_analysis(results_df):
    """Print analysis of optimization power predictions"""

    print("\n" + "=" * 120)
    print("🔌 OPTIMIZATION POWER PREDICTION ANALYSIS")
    print("=" * 120)

    # Group by zone for better organization
    zones = sorted(results_df["zone"].unique())

    for zone in zones:
        zone_data = results_df[results_df["zone"] == zone]

        print(f"\n🏢 {zone}")
        print("-" * 100)

        # Print header
        print(
            f"{'Status':<8} {'Count':<8} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std':<12} {'Q25':<12} {'Q75':<12}"
        )
        print("-" * 100)

        # Print AC OFF data
        ac_off = zone_data[zone_data["status"] == "AC OFF"].iloc[0]
        print(
            f"{'OFF':<8} {ac_off['count']:<8} {ac_off['mean_power']:<12.1f} {ac_off['median_power']:<12.1f} "
            f"{ac_off['min_power']:<12.1f} {ac_off['max_power']:<12.1f} {ac_off['std_power']:<12.1f} "
            f"{ac_off['q25_power']:<12.1f} {ac_off['q75_power']:<12.1f}"
        )

        # Print AC ON data
        ac_on = zone_data[zone_data["status"] == "AC ON"].iloc[0]
        print(
            f"{'ON':<8} {ac_on['count']:<8} {ac_on['mean_power']:<12.1f} {ac_on['median_power']:<12.1f} "
            f"{ac_on['min_power']:<12.1f} {ac_on['max_power']:<12.1f} {ac_on['std_power']:<12.1f} "
            f"{ac_on['q25_power']:<12.1f} {ac_on['q75_power']:<12.1f}"
        )

        # Calculate power ratio (ON/OFF)
        if ac_off["mean_power"] > 0:
            power_ratio = ac_on["mean_power"] / ac_off["mean_power"]
            print(f"📊 Power Ratio (ON/OFF): {power_ratio:.1f}x")

            # Check if ratio is suspiciously low (indicating AC OFF predictions are too high)
            if power_ratio < 2.0:
                print(
                    f"⚠️  WARNING: Power ratio is very low ({power_ratio:.1f}x). AC OFF predictions may be too high!"
                )
            elif power_ratio < 3.0:
                print(
                    f"⚠️  CAUTION: Power ratio is low ({power_ratio:.1f}x). AC OFF predictions may be higher than expected."
                )
        else:
            print(f"📊 Power Ratio (ON/OFF): N/A (AC OFF power is 0)")


def compare_with_historical_baseline():
    """Compare optimization predictions with historical baseline data"""

    print("\n" + "=" * 120)
    print("📊 COMPARISON WITH HISTORICAL BASELINE DATA")
    print("=" * 120)

    # Historical baseline data from our previous analysis
    historical_baseline = {
        "Area 1": 51670.0,
        "Area 2": 106718.0,
        "Area 3": 8297.0,
        "Area 4": 48457.0,
        "Meeting Room": 25428.0,
        "Break Room": 21756.0,
    }

    print(f"\n📈 Historical Baseline vs Optimization Predictions:")
    print("-" * 80)
    print(
        f"{'Zone':<15} {'Historical':<12} {'Optimization':<12} {'Difference':<12} {'Ratio':<8}"
    )
    print("-" * 80)

    # This would need to be populated with actual optimization data
    # For now, we'll show the historical baseline
    for zone, baseline in historical_baseline.items():
        print(f"{zone:<15} {baseline:<12.1f} {'TBD':<12} {'TBD':<12} {'TBD':<8}")


def print_data_quality_info(df):
    """Print data quality information"""

    print("\n" + "=" * 120)
    print("📊 OPTIMIZATION DATA QUALITY INFORMATION")
    print("=" * 120)

    total_rows = len(df)

    # Get all zone columns
    zone_columns = [col for col in df.columns if "_PredPower" in col]
    zones = [col.replace("_PredPower", "") for col in zone_columns]

    print(f"📈 Total time periods: {total_rows}")
    print(f"📈 Zones analyzed: {len(zones)}")
    print(f"📈 Zones: {', '.join(zones)}")

    # Check for missing data
    for zone in zones:
        power_col = f"{zone}_PredPower"
        onoff_col = f"{zone}_OnOFF"

        if power_col in df.columns:
            power_missing = df[power_col].isna().sum()
            print(
                f"📈 {zone} - Missing power data: {power_missing}/{total_rows} ({power_missing/total_rows*100:.1f}%)"
            )

        if onoff_col in df.columns:
            onoff_missing = df[onoff_col].isna().sum()
            print(
                f"📈 {zone} - Missing OnOFF data: {onoff_missing}/{total_rows} ({onoff_missing/total_rows*100:.1f}%)"
            )


def main():
    """Main function"""

    print("🚀 Starting Optimization Power Analysis...")

    # Load data
    df = load_optimization_data()

    # Print data quality info
    print_data_quality_info(df)

    # Analyze power consumption
    results_df = analyze_power_by_zone_and_status(df)

    # Print detailed results
    print_optimization_analysis(results_df)

    # Compare with historical baseline
    compare_with_historical_baseline()

    print("\n" + "=" * 120)
    print("✅ Optimization power analysis completed!")
    print("=" * 120)


if __name__ == "__main__":
    main()
