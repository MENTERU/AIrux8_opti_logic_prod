#!/usr/bin/env python3
"""
Power Analysis Summary Script

This script analyzes power consumption statistics for each area/zone
when AC is ON vs OFF from the features CSV data.

Usage:
    python power_analysis_summary.py

Output:
    - Terminal table showing power statistics for each zone
    - Comparison between AC ON and AC OFF power consumption
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_features_data():
    """Load the features CSV data"""
    features_path = Path("data/02_PreprocessedData/Clea/features_processed_Clea.csv")

    if not features_path.exists():
        print(f"❌ Error: Features file not found at {features_path}")
        print(
            "Please ensure you're running this script from the project root directory"
        )
        sys.exit(1)

    try:
        df = pd.read_csv(features_path)
        print(f"✅ Loaded features data: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading features data: {e}")
        sys.exit(1)


def analyze_power_by_zone_and_ac_status(df):
    """Analyze power consumption by zone and AC status"""

    # Filter out rows with missing power data
    df_clean = df.dropna(subset=["adjusted_power", "A/C ON/OFF"])

    # Get unique zones
    zones = sorted(df_clean["zone"].unique())

    results = []

    for zone in zones:
        zone_data = df_clean[df_clean["zone"] == zone]

        # AC OFF data (A/C ON/OFF = 0, meaning 0 units ON)
        ac_off_data = zone_data[zone_data["A/C ON/OFF"] == 0]

        # AC ON data (A/C ON/OFF > 0, meaning 1 or more units ON)
        ac_on_data = zone_data[zone_data["A/C ON/OFF"] > 0]

        # Calculate statistics for AC OFF
        if len(ac_off_data) > 0:
            ac_off_stats = {
                "zone": zone,
                "status": "AC OFF",
                "count": len(ac_off_data),
                "mean_power": ac_off_data["adjusted_power"].mean(),
                "median_power": ac_off_data["adjusted_power"].median(),
                "min_power": ac_off_data["adjusted_power"].min(),
                "max_power": ac_off_data["adjusted_power"].max(),
                "std_power": ac_off_data["adjusted_power"].std(),
                "q25_power": ac_off_data["adjusted_power"].quantile(0.25),
                "q75_power": ac_off_data["adjusted_power"].quantile(0.75),
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
                "mean_power": ac_on_data["adjusted_power"].mean(),
                "median_power": ac_on_data["adjusted_power"].median(),
                "min_power": ac_on_data["adjusted_power"].min(),
                "max_power": ac_on_data["adjusted_power"].max(),
                "std_power": ac_on_data["adjusted_power"].std(),
                "q25_power": ac_on_data["adjusted_power"].quantile(0.25),
                "q75_power": ac_on_data["adjusted_power"].quantile(0.75),
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


def print_summary_table(results_df, original_df):
    """Print a formatted summary table"""

    print("\n" + "=" * 120)
    print("🔌 POWER CONSUMPTION ANALYSIS BY ZONE AND AC STATUS")
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
        else:
            print(f"📊 Power Ratio (ON/OFF): N/A (AC OFF power is 0)")

        # Show count distribution for this zone
        zone_original_data = original_df[original_df["zone"] == zone]
        zone_ac_counts = zone_original_data["A/C ON/OFF"].value_counts().sort_index()
        print(f"📊 Units ON Distribution:")
        for count_val, records in zone_ac_counts.head(8).items():
            if count_val == 0:
                print(f"   {count_val} units: {records:,} records")
            else:
                print(f"   {count_val} units: {records:,} records")
        if len(zone_ac_counts) > 8:
            remaining = zone_ac_counts.iloc[8:].sum()
            print(f"   ... and {remaining:,} records with other counts")


def print_comparison_summary(results_df):
    """Print a comparison summary across all zones"""

    print("\n" + "=" * 120)
    print("📈 COMPARISON SUMMARY ACROSS ALL ZONES")
    print("=" * 120)

    # Calculate overall statistics
    ac_off_data = results_df[results_df["status"] == "AC OFF"]
    ac_on_data = results_df[results_df["status"] == "AC ON"]

    print(f"\n🔌 AC OFF Statistics:")
    print(
        f"   • Average power across all zones: {ac_off_data['mean_power'].mean():.1f} W"
    )
    print(
        f"   • Median power across all zones: {ac_off_data['median_power'].median():.1f} W"
    )
    print(f"   • Total data points: {ac_off_data['count'].sum():,}")

    print(f"\n🔌 AC ON Statistics:")
    print(
        f"   • Average power across all zones: {ac_on_data['mean_power'].mean():.1f} W"
    )
    print(
        f"   • Median power across all zones: {ac_on_data['median_power'].median():.1f} W"
    )
    print(f"   • Total data points: {ac_on_data['count'].sum():,}")

    # Calculate power ratios for each zone
    print(f"\n📊 Power Ratio (AC ON / AC OFF) by Zone:")
    print("-" * 50)

    zones = sorted(results_df["zone"].unique())
    for zone in zones:
        zone_data = results_df[results_df["zone"] == zone]
        ac_off = zone_data[zone_data["status"] == "AC OFF"].iloc[0]
        ac_on = zone_data[zone_data["status"] == "AC ON"].iloc[0]

        if ac_off["mean_power"] > 0:
            ratio = ac_on["mean_power"] / ac_off["mean_power"]
            print(f"   {zone:<15}: {ratio:>6.1f}x")
        else:
            print(f"   {zone:<15}: N/A (AC OFF = 0)")


def print_data_quality_info(df):
    """Print data quality information"""

    print("\n" + "=" * 120)
    print("📊 DATA QUALITY INFORMATION")
    print("=" * 120)

    total_rows = len(df)
    rows_with_power = len(df.dropna(subset=["adjusted_power"]))
    rows_with_ac_status = len(df.dropna(subset=["A/C ON/OFF"]))
    rows_complete = len(df.dropna(subset=["adjusted_power", "A/C ON/OFF"]))

    print(f"📈 Total rows in dataset: {total_rows:,}")
    print(
        f"📈 Rows with power data: {rows_with_power:,} ({rows_with_power/total_rows*100:.1f}%)"
    )
    print(
        f"📈 Rows with AC status: {rows_with_ac_status:,} ({rows_with_ac_status/total_rows*100:.1f}%)"
    )
    print(
        f"📈 Complete rows (power + AC status): {rows_complete:,} ({rows_complete/total_rows*100:.1f}%)"
    )

    # AC status distribution (count-based)
    ac_status_counts = df["A/C ON/OFF"].value_counts().sort_index()
    print(f"\n🔌 AC Status Distribution (Count of Units ON):")

    # Show OFF (0 units ON)
    off_count = ac_status_counts.get(0, 0)
    print(f"   OFF (0 units): {off_count:,} ({off_count/total_rows*100:.1f}%)")

    # Show ON (1+ units ON) - sum all non-zero counts
    on_count = ac_status_counts[ac_status_counts.index > 0].sum()
    print(f"   ON (1+ units): {on_count:,} ({on_count/total_rows*100:.1f}%)")

    # Show top count values for reference
    print(f"\n📊 Top Count Values (Units ON):")
    top_counts = ac_status_counts.head(10)
    for count_value, records in top_counts.items():
        if count_value == 0:
            print(f"   {count_value} units: {records:,} records")
        else:
            print(f"   {count_value} units: {records:,} records")

    # Zone distribution
    zone_counts = df["zone"].value_counts()
    print(f"\n🏢 Data Points per Zone:")
    for zone, count in zone_counts.items():
        print(f"   {zone}: {count:,}")


def main():
    """Main function"""

    print("🚀 Starting Power Analysis Summary...")

    # Load data
    df = load_features_data()

    # Print data quality info
    print_data_quality_info(df)

    # Analyze power consumption
    results_df = analyze_power_by_zone_and_ac_status(df)

    # Print detailed results
    print_summary_table(results_df, df)

    # Print comparison summary
    print_comparison_summary(results_df)

    print("\n" + "=" * 120)
    print("✅ Power analysis completed successfully!")
    print("=" * 120)


if __name__ == "__main__":
    main()
