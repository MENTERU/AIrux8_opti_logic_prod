#!/usr/bin/env python3
"""
AC OFF Power Investigation Script

This script investigates why some areas show high power consumption when AC is "OFF".
The hypothesis is that "AC OFF" might mean the zone/area is OFF, but individual
indoor units within that zone might still be ON.

Investigation:
1. Check ac_control_processed_Clea.csv - individual unit ON/OFF status
2. Check power_meter_processed_Clea.csv - actual power consumption
3. Compare zone-level "OFF" status with individual unit status
4. Identify cases where zone is "OFF" but some units are "ON"

Japanese question: その時に、エリアに含まれる室内機が全てOFFではないですよね？？たぶん
Translation: "At that time, not all indoor units in the area are OFF, right?? Probably"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_data():
    """Load the preprocessed data files"""

    ac_control_path = Path(
        "data/02_PreprocessedData/Clea/ac_control_processed_Clea.csv"
    )
    power_meter_path = Path(
        "data/02_PreprocessedData/Clea/power_meter_processed_Clea.csv"
    )
    features_path = Path("data/02_PreprocessedData/Clea/features_processed_Clea.csv")

    print("📂 Loading data files...")

    if not ac_control_path.exists():
        print(f"❌ Error: AC control file not found at {ac_control_path}")
        sys.exit(1)

    if not power_meter_path.exists():
        print(f"❌ Error: Power meter file not found at {power_meter_path}")
        sys.exit(1)

    if not features_path.exists():
        print(f"❌ Error: Features file not found at {features_path}")
        sys.exit(1)

    ac_control = pd.read_csv(ac_control_path)
    power_meter = pd.read_csv(power_meter_path)
    features = pd.read_csv(features_path)

    print(f"✅ AC Control data loaded: {len(ac_control):,} rows")
    print(f"✅ Power Meter data loaded: {len(power_meter):,} rows")
    print(f"✅ Features data loaded: {len(features):,} rows")

    return ac_control, power_meter, features


def analyze_ac_control_structure(ac_control):
    """Analyze the structure of AC control data"""

    print("\n" + "=" * 120)
    print("📊 AC CONTROL DATA STRUCTURE ANALYSIS")
    print("=" * 120)

    print(f"\n📋 Columns in AC Control data:")
    for idx, col in enumerate(ac_control.columns, 1):
        print(f"   {idx}. {col}")

    # Check for zone information
    if "zone" in ac_control.columns:
        zones = ac_control["zone"].unique()
        print(f"\n🏢 Zones found: {len(zones)}")
        for zone in sorted(zones):
            zone_data = ac_control[ac_control["zone"] == zone]
            print(f"   • {zone}: {len(zone_data):,} records")

    # Check for unit information
    unit_columns = [
        col
        for col in ac_control.columns
        if "unit" in col.lower() or "indoor" in col.lower()
    ]
    if unit_columns:
        print(f"\n🔧 Unit-related columns found:")
        for col in unit_columns:
            print(f"   • {col}")

    # Check ON/OFF columns
    onoff_columns = [
        col for col in ac_control.columns if "ON/OFF" in col or "on_off" in col.lower()
    ]
    if onoff_columns:
        print(f"\n🔌 ON/OFF columns found:")
        for col in onoff_columns:
            unique_vals = ac_control[col].dropna().unique()
            print(f"   • {col}: {unique_vals}")

    # Sample data
    print(f"\n📝 Sample AC Control data (first 5 rows):")
    print(ac_control.head().to_string())


def analyze_zone_vs_unit_status(ac_control, power_meter, features):
    """
    Investigate if zone-level OFF status means all units are OFF
    or if some units can still be ON
    """

    print("\n" + "=" * 120)
    print("🔍 ZONE vs INDIVIDUAL UNIT STATUS ANALYSIS")
    print("=" * 120)

    # Merge AC control with power meter on timestamp and zone
    if "Datetime" in ac_control.columns and "Datetime" in power_meter.columns:
        merged = pd.merge(
            ac_control,
            power_meter,
            on=(
                ["Datetime", "zone"]
                if "zone" in ac_control.columns and "zone" in power_meter.columns
                else ["Datetime"]
            ),
            how="inner",
        )
        print(f"\n✅ Merged AC control and power meter data: {len(merged):,} rows")
    else:
        print("\n⚠️  Warning: Cannot merge data - missing Datetime or zone columns")
        return

    # Check if there's a zone-level ON/OFF and individual unit ON/OFF
    if "A/C ON/OFF" in merged.columns:
        # Find cases where zone is OFF but power is high
        zone_off_data = merged[merged["A/C ON/OFF"] == 0]

        if len(zone_off_data) > 0:
            print(f"\n📊 Zone-level AC OFF cases: {len(zone_off_data):,} records")

            # Analyze power distribution when zone is OFF
            if "adjusted_power" in zone_off_data.columns:
                print(f"\n💡 Power consumption when zone AC is OFF:")
                print(f"   • Mean: {zone_off_data['adjusted_power'].mean():.1f} W")
                print(f"   • Median: {zone_off_data['adjusted_power'].median():.1f} W")
                print(f"   • Min: {zone_off_data['adjusted_power'].min():.1f} W")
                print(f"   • Max: {zone_off_data['adjusted_power'].max():.1f} W")

                # Find high power cases when zone is OFF
                high_power_threshold = zone_off_data["adjusted_power"].quantile(0.75)
                high_power_off = zone_off_data[
                    zone_off_data["adjusted_power"] > high_power_threshold
                ]

                print(
                    f"\n⚠️  High power consumption while zone AC is OFF (>{high_power_threshold:.1f} W):"
                )
                print(
                    f"   • Cases: {len(high_power_off):,} ({len(high_power_off)/len(zone_off_data)*100:.1f}%)"
                )

                # Show sample cases
                if len(high_power_off) > 0:
                    print(f"\n📝 Sample cases (high power while AC OFF):")
                    sample_cols = ["Datetime", "zone", "A/C ON/OFF", "adjusted_power"]
                    available_cols = [
                        col for col in sample_cols if col in high_power_off.columns
                    ]
                    print(high_power_off[available_cols].head(10).to_string())


def analyze_unit_level_data(ac_control):
    """
    Analyze if the data contains individual unit information
    """

    print("\n" + "=" * 120)
    print("🔧 INDIVIDUAL UNIT LEVEL ANALYSIS")
    print("=" * 120)

    # Look for columns that might indicate individual units
    potential_unit_cols = []
    for col in ac_control.columns:
        if any(
            keyword in col.lower()
            for keyword in ["unit", "indoor", "device", "equipment"]
        ):
            potential_unit_cols.append(col)

    if potential_unit_cols:
        print(f"\n📋 Potential unit identifier columns:")
        for col in potential_unit_cols:
            unique_count = ac_control[col].nunique()
            print(f"   • {col}: {unique_count} unique values")

            # Show sample values
            sample_values = ac_control[col].dropna().unique()[:5]
            print(f"     Sample values: {sample_values}")
    else:
        print("\n⚠️  No obvious individual unit identifier columns found")

    # Check if data is at zone level or unit level
    if "Datetime" in ac_control.columns and "zone" in ac_control.columns:
        # Count records per timestamp per zone
        records_per_timestamp_zone = ac_control.groupby(["Datetime", "zone"]).size()
        max_records = records_per_timestamp_zone.max()

        print(f"\n📊 Data granularity analysis:")
        print(f"   • Max records per (timestamp, zone): {max_records}")

        if max_records > 1:
            print(
                f"   • ✅ Data appears to be at UNIT level (multiple units per zone per timestamp)"
            )

            # Show distribution
            distribution = records_per_timestamp_zone.value_counts().sort_index()
            print(f"\n   Distribution of units per (timestamp, zone):")
            for num_units, count in distribution.items():
                print(f"      {num_units} units: {count:,} cases")
        else:
            print(
                f"   • ⚠️  Data appears to be at ZONE level (one record per zone per timestamp)"
            )


def investigate_mixed_unit_states(ac_control):
    """
    Investigate if some units in a zone can be ON while others are OFF
    """

    print("\n" + "=" * 120)
    print(
        "🔍 MIXED UNIT STATE INVESTIGATION (Answer to: エリアに含まれる室内機が全てOFFではないですよね？)"
    )
    print("=" * 120)

    if (
        "Datetime" in ac_control.columns
        and "zone" in ac_control.columns
        and "A/C ON/OFF" in ac_control.columns
    ):
        # Group by timestamp and zone, check ON/OFF status
        grouped = ac_control.groupby(["Datetime", "zone"])["A/C ON/OFF"].agg(
            ["count", "sum", "mean"]
        )
        grouped.columns = ["total_units", "units_on", "on_ratio"]

        # Find cases where some (but not all) units are ON
        mixed_state_cases = grouped[
            (grouped["units_on"] > 0) & (grouped["units_on"] < grouped["total_units"])
        ]

        print(f"\n📊 Analysis Results:")
        print(f"   • Total (timestamp, zone) combinations: {len(grouped):,}")
        print(
            f"   • Cases where ALL units are OFF: {len(grouped[grouped['units_on'] == 0]):,}"
        )
        print(
            f"   • Cases where ALL units are ON: {len(grouped[grouped['units_on'] == grouped['total_units']]):,}"
        )
        print(
            f"   • Cases where SOME units are ON (mixed state): {len(mixed_state_cases):,}"
        )

        if len(mixed_state_cases) > 0:
            print(
                f"\n✅ ANSWER: YES! Not all indoor units in the area are OFF at the same time!"
            )
            print(
                f"   • {len(mixed_state_cases):,} cases found where some units are ON while others are OFF"
            )
            print(
                f"   • This is {len(mixed_state_cases)/len(grouped)*100:.1f}% of all cases"
            )

            # Show examples
            print(f"\n📝 Sample cases (mixed ON/OFF states):")
            sample = mixed_state_cases.head(10)
            print(sample.to_string())

            # Analyze by zone
            print(f"\n🏢 Mixed state cases by zone:")
            if "zone" in ac_control.columns:
                zone_mixed = (
                    mixed_state_cases.reset_index()
                    .groupby("zone")
                    .size()
                    .sort_values(ascending=False)
                )
                for zone, count in zone_mixed.items():
                    total_for_zone = len(
                        grouped[grouped.index.get_level_values("zone") == zone]
                    )
                    percentage = count / total_for_zone * 100
                    print(
                        f"   • {zone}: {count:,} mixed cases ({percentage:.1f}% of zone's timestamps)"
                    )
        else:
            print(
                f"\n⚠️  No mixed state cases found - units in a zone are always all ON or all OFF together"
            )


def main():
    """Main analysis function"""

    print("🚀 Starting AC OFF Power Investigation...")
    print("=" * 120)

    # Load data
    ac_control, power_meter, features = load_data()

    # Analyze AC control structure
    analyze_ac_control_structure(ac_control)

    # Analyze unit level data
    analyze_unit_level_data(ac_control)

    # Investigate mixed unit states
    investigate_mixed_unit_states(ac_control)

    # Analyze zone vs unit status
    analyze_zone_vs_unit_status(ac_control, power_meter, features)

    print("\n" + "=" * 120)
    print("✅ Investigation completed!")
    print("=" * 120)


if __name__ == "__main__":
    main()
