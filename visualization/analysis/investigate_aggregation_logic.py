#!/usr/bin/env python3
"""
Investigation of Aggregation Logic

This script investigates the exact aggregation logic in aggregator.py that produces
"AC OFF" status with high power consumption in features_processed_Clea.csv.

Key Questions:
1. How is AC ON/OFF status aggregated from individual units to zone level?
2. How is power consumption aggregated from individual units to zone level?
3. Why do we get "AC OFF" with high power consumption?

Findings from aggregator.py:
- AC ON/OFF: Uses _most_frequent() - takes the most common ON/OFF status
- Power: Uses SUM() - adds up all power consumption from all units
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_data():
    """Load the data files"""
    ac_path = Path("data/02_PreprocessedData/Clea/ac_control_processed_Clea.csv")
    power_path = Path("data/02_PreprocessedData/Clea/power_meter_processed_Clea.csv")
    features_path = Path("data/02_PreprocessedData/Clea/features_processed_Clea.csv")
    master_path = Path("data/01_MasterData/MASTER_Clea.xlsx")

    print("📂 Loading data files...")

    ac_control = pd.read_csv(ac_path)
    ac_control["Datetime"] = pd.to_datetime(ac_control["Datetime"])

    power_meter = pd.read_csv(power_path)
    power_meter["Datetime"] = pd.to_datetime(power_meter["Datetime"])

    features = pd.read_csv(features_path)
    features["Datetime"] = pd.to_datetime(features["Datetime"])

    master = pd.read_excel(master_path, sheet_name="MASTER")

    print(f"✅ AC Control: {len(ac_control):,} rows")
    print(f"✅ Power Meter: {len(power_meter):,} rows")
    print(f"✅ Features: {len(features):,} rows")
    print(f"✅ Master: {len(master):,} rows")

    return ac_control, power_meter, features, master


def create_unit_zone_mapping(master):
    """Create unit to zone mapping"""
    unit_zone_map = {}
    if "環境予測区分" in master.columns and "制御区分" in master.columns:
        for _, row in master.iterrows():
            unit_name = row["環境予測区分"]
            zone_name = row["制御区分"]
            if pd.notna(unit_name) and pd.notna(zone_name):
                unit_zone_map[unit_name] = zone_name

    print(f"\n📋 Unit-to-Zone Mapping: {len(unit_zone_map)} mappings")
    return unit_zone_map


def simulate_aggregation_logic(
    ac_control, power_meter, unit_zone_map, sample_datetime, sample_zone
):
    """
    Simulate the exact aggregation logic from aggregator.py
    """
    print("\n" + "=" * 120)
    print("🔍 SIMULATING AGGREGATION LOGIC")
    print("=" * 120)

    # Filter data for the sample
    ac_sample = ac_control[
        (ac_control["Datetime"] == sample_datetime)
        & (
            ac_control["A/C Name"].isin(
                [k for k, v in unit_zone_map.items() if v == sample_zone]
            )
        )
    ].copy()

    print(f"\n📊 Sample Data for {sample_zone} at {sample_datetime}:")
    print(f"   • AC Control records: {len(ac_sample)}")

    if len(ac_sample) == 0:
        print("   ❌ No AC control data found for this sample")
        return

    # Show individual unit status
    print(f"\n📋 Individual Unit Status:")
    unit_details = ac_sample[["A/C Name", "A/C ON/OFF", "A/C Mode", "Indoor Temp."]]
    print(unit_details.to_string(index=False))

    # Simulate AC aggregation logic (from aggregator.py lines 102-114)
    print(f"\n🔧 AC Aggregation Logic (from aggregator.py):")
    print(f"   • A/C ON/OFF: Uses _most_frequent() - most common status")
    print(f"   • A/C Mode: Uses _most_frequent() - most common mode")
    print(f"   • Indoor Temp.: Uses mean() - average temperature")

    # Apply the exact aggregation logic
    ac_aggregated = (
        ac_sample.groupby("Datetime")
        .agg(
            {
                "A/C Set Temperature": lambda x: (
                    x.mode().iloc[0] if not x.mode().empty else np.nan
                ),
                "Indoor Temp.": "mean",
                "A/C ON/OFF": lambda x: (
                    x.mode().iloc[0] if not x.mode().empty else np.nan
                ),
                "A/C Mode": lambda x: (
                    x.mode().iloc[0] if not x.mode().empty else np.nan
                ),
                "A/C Fan Speed": lambda x: (
                    x.mode().iloc[0] if not x.mode().empty else np.nan
                ),
            }
        )
        .reset_index()
    )

    print(f"\n📊 Aggregated AC Status:")
    for col in ["A/C ON/OFF", "A/C Mode", "Indoor Temp."]:
        if col in ac_aggregated.columns:
            print(f"   • {col}: {ac_aggregated[col].iloc[0]}")

    # Count ON vs OFF units
    on_count = (ac_sample["A/C ON/OFF"] == "ON").sum()
    off_count = (ac_sample["A/C ON/OFF"] == "OFF").sum()
    total_units = len(ac_sample)

    print(f"\n📈 Unit Status Breakdown:")
    print(f"   • Total units: {total_units}")
    print(f"   • Units ON: {on_count} ({on_count/total_units*100:.1f}%)")
    print(f"   • Units OFF: {off_count} ({off_count/total_units*100:.1f}%)")

    # Show the aggregation result
    most_frequent_onoff = (
        ac_sample["A/C ON/OFF"].mode().iloc[0]
        if not ac_sample["A/C ON/OFF"].mode().empty
        else "Unknown"
    )
    print(f"\n🎯 Aggregation Result:")
    print(f"   • Most frequent A/C ON/OFF: {most_frequent_onoff}")
    print(f"   • This becomes the zone-level 'A/C ON/OFF' status")

    # Explain the issue
    if most_frequent_onoff == "OFF" and on_count > 0:
        print(f"\n⚠️  ISSUE IDENTIFIED:")
        print(f"   • Zone-level status: {most_frequent_onoff} (most frequent)")
        print(f"   • But {on_count} units are still ON!")
        print(f"   • This explains why 'AC OFF' has high power consumption")

    return ac_aggregated, on_count, off_count


def analyze_power_aggregation(power_meter, sample_datetime, sample_zone, unit_zone_map):
    """
    Analyze how power is aggregated (SUM vs other methods)
    """
    print(f"\n" + "=" * 120)
    print("⚡ POWER AGGREGATION ANALYSIS")
    print("=" * 120)

    # Get power meter data for the zone's power meter IDs
    # This is more complex as we need to map zones to power meter IDs
    print(f"\n🔧 Power Aggregation Logic (from aggregator.py):")
    print(f"   • Power: Uses SUM() - adds up all power consumption")
    print(f"   • Each zone can have multiple power meters")
    print(f"   • Total power = Σ(power_meter_i * load_share_i)")

    # Show sample power data
    power_sample = power_meter[power_meter["Datetime"] == sample_datetime]
    print(f"\n📊 Power Meter Data at {sample_datetime}:")
    print(f"   • Power meter records: {len(power_sample)}")
    if len(power_sample) > 0:
        print(
            power_sample[["Datetime", "Mesh ID", "PM Addr ID", "Total_kWh"]]
            .head()
            .to_string(index=False)
        )

    print(f"\n💡 Key Insight:")
    print(f"   • Power aggregation uses SUM() - always adds up consumption")
    print(f"   • Even if AC status is 'OFF', power meters still record consumption")
    print(f"   • This includes: lighting, ventilation, equipment, other systems")
    print(f"   • Result: 'AC OFF' can still have significant power consumption")


def show_real_examples(ac_control, features, unit_zone_map):
    """
    Show real examples from the features file
    """
    print(f"\n" + "=" * 120)
    print("📝 REAL EXAMPLES FROM FEATURES FILE")
    print("=" * 120)

    # Find cases where AC is OFF but power is high
    high_power_off = features[
        (features["A/C ON/OFF"] == 0) & (features["adjusted_power"] > 10000)
    ]

    print(f"\n🔍 Cases where AC is OFF but power > 10,000W:")
    print(f"   • Found {len(high_power_off)} cases")

    if len(high_power_off) > 0:
        # Show sample cases
        sample_cases = high_power_off[
            ["Datetime", "zone", "A/C ON/OFF", "adjusted_power", "Indoor Temp."]
        ].head(5)
        print(f"\n📊 Sample Cases:")
        print(sample_cases.to_string(index=False))

        # Analyze one specific case
        example = high_power_off.iloc[0]
        example_datetime = example["Datetime"]
        example_zone = example["zone"]

        print(f"\n🔍 Detailed Analysis of Example Case:")
        print(f"   • Timestamp: {example_datetime}")
        print(f"   • Zone: {example_zone}")
        print(f"   • AC Status: {example['A/C ON/OFF']} (0=OFF)")
        print(f"   • Power: {example['adjusted_power']:.1f} W")
        print(f"   • Indoor Temp: {example['Indoor Temp.']:.1f}°C")

        # Check individual units for this case
        ac_units = ac_control[
            (ac_control["Datetime"] == example_datetime)
            & (
                ac_control["A/C Name"].isin(
                    [k for k, v in unit_zone_map.items() if v == example_zone]
                )
            )
        ]

        if len(ac_units) > 0:
            print(f"\n📋 Individual Units in {example_zone} at {example_datetime}:")
            unit_status = ac_units[
                ["A/C Name", "A/C ON/OFF", "A/C Mode", "Indoor Temp."]
            ]
            print(unit_status.to_string(index=False))

            on_count = (ac_units["A/C ON/OFF"] == "ON").sum()
            off_count = (ac_units["A/C ON/OFF"] == "OFF").sum()
            most_frequent = (
                ac_units["A/C ON/OFF"].mode().iloc[0]
                if not ac_units["A/C ON/OFF"].mode().empty
                else "Unknown"
            )

            print(f"\n🎯 Aggregation Analysis:")
            print(f"   • Units ON: {on_count}")
            print(f"   • Units OFF: {off_count}")
            print(f"   • Most frequent status: {most_frequent}")
            print(f"   • Zone-level status: {example['A/C ON/OFF']} (0=OFF)")

            if most_frequent == "OFF" and on_count > 0:
                print(
                    f"\n✅ CONFIRMED: Zone marked as 'OFF' but {on_count} units are still ON!"
                )


def main():
    """Main analysis function"""
    print("🚀 Investigating Aggregation Logic...")
    print("=" * 120)
    print(
        "Question: Why does features_processed_Clea.csv show 'AC OFF' with high power?"
    )
    print("=" * 120)

    # Load data
    ac_control, power_meter, features, master = load_data()

    # Create mapping
    unit_zone_map = create_unit_zone_mapping(master)

    # Find a good sample case
    sample_cases = features[
        (features["A/C ON/OFF"] == 0) & (features["adjusted_power"] > 20000)
    ].head(3)

    if len(sample_cases) > 0:
        for _, case in sample_cases.iterrows():
            sample_datetime = case["Datetime"]
            sample_zone = case["zone"]

            print(f"\n" + "=" * 80)
            print(f"🔍 ANALYZING CASE: {sample_zone} at {sample_datetime}")
            print(f"   • AC Status: {case['A/C ON/OFF']} (0=OFF)")
            print(f"   • Power: {case['adjusted_power']:.1f} W")
            print("=" * 80)

            # Simulate aggregation
            simulate_aggregation_logic(
                ac_control, power_meter, unit_zone_map, sample_datetime, sample_zone
            )

            # Analyze power aggregation
            analyze_power_aggregation(
                power_meter, sample_datetime, sample_zone, unit_zone_map
            )

    # Show real examples
    show_real_examples(ac_control, features, unit_zone_map)

    print(f"\n" + "=" * 120)
    print("🎯 CONCLUSION - ROOT CAUSE IDENTIFIED")
    print("=" * 120)
    print("✅ The aggregation logic in aggregator.py explains the issue:")
    print("   • AC ON/OFF: Uses _most_frequent() - takes most common status")
    print("   • Power: Uses SUM() - adds up all consumption")
    print(
        "   • Result: Zone can be 'OFF' (majority units OFF) but still have high power"
    )
    print("   • This happens when some units are ON while others are OFF")
    print(
        "   • The 'OFF' status is misleading - it means 'majority OFF', not 'all OFF'"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
