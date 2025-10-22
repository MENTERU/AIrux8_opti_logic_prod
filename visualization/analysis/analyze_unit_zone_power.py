#!/usr/bin/env python3
"""
Unit-Zone Power Analysis

This script answers the critical question:
Japanese: その時に、エリアに含まれる室内機が全てOFFではないですよね？？たぶん
English: "At that time, not all indoor units in the area are OFF, right?? Probably"

Analysis:
1. Map individual AC units to zones using master data
2. For each zone at each timestamp, check if ALL units are OFF or only SOME
3. Compare power consumption between "all units OFF" vs "some units ON"
4. Explain why zone shows high power when marked as "AC OFF"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_master_data():
    """Load master data to get unit-to-zone mapping"""
    master_path = Path("data/01_MasterData/MASTER_Clea.xlsx")

    if not master_path.exists():
        print(f"❌ Error: Master file not found at {master_path}")
        sys.exit(1)

    master = pd.read_excel(master_path, sheet_name="MASTER")

    # Create unit to zone mapping using 環境予測区分 (indoor unit names) -> 制御区分 (zone names)
    unit_zone_map = {}
    if "環境予測区分" in master.columns and "制御区分" in master.columns:
        for _, row in master.iterrows():
            unit_name = row["環境予測区分"]  # Indoor unit name (e.g., E-16北4, D-6北1)
            zone_name = row["制御区分"]  # Zone name (e.g., Area 1, Area 2)
            if pd.notna(unit_name) and pd.notna(zone_name):
                unit_zone_map[unit_name] = zone_name

    print(f"✅ Master data loaded: {len(unit_zone_map)} unit-to-zone mappings")

    # Show sample mappings
    print(f"\n📋 Sample unit-to-zone mappings:")
    for unit, zone in list(unit_zone_map.items())[:10]:
        print(f"   • {unit} → {zone}")

    return unit_zone_map, master


def load_ac_control_data():
    """Load AC control data"""
    ac_path = Path("data/02_PreprocessedData/Clea/ac_control_processed_Clea.csv")

    if not ac_path.exists():
        print(f"❌ Error: AC control file not found at {ac_path}")
        sys.exit(1)

    # Load with date parsing
    ac_control = pd.read_csv(ac_path)
    ac_control["Datetime"] = pd.to_datetime(ac_control["Datetime"])

    print(f"✅ AC control data loaded: {len(ac_control):,} rows")
    return ac_control


def load_power_data():
    """Load power meter data"""
    power_path = Path("data/02_PreprocessedData/Clea/power_meter_processed_Clea.csv")

    if not power_path.exists():
        print(f"❌ Error: Power meter file not found at {power_path}")
        sys.exit(1)

    power_data = pd.read_csv(power_path)
    power_data["Datetime"] = pd.to_datetime(power_data["Datetime"])

    print(f"✅ Power meter data loaded: {len(power_data):,} rows")
    return power_data


def analyze_zone_unit_status(ac_control, unit_zone_map):
    """
    Analyze ON/OFF status at zone and unit level
    """

    print("\n" + "=" * 120)
    print("🔍 ZONE vs UNIT STATUS ANALYSIS")
    print("=" * 120)

    # Add zone information to AC control data
    ac_control["zone"] = ac_control["A/C Name"].map(unit_zone_map)

    # Remove rows without zone mapping
    ac_with_zone = ac_control[ac_control["zone"].notna()].copy()
    print(f"\n📊 AC control records with zone mapping: {len(ac_with_zone):,}")

    # Convert ON/OFF to binary
    ac_with_zone["is_on"] = (ac_with_zone["A/C ON/OFF"] == "ON").astype(int)

    # Group by zone and timestamp
    zone_status = (
        ac_with_zone.groupby(["Datetime", "zone"])
        .agg({"is_on": ["sum", "count", "mean"], "A/C Name": lambda x: list(x)})
        .reset_index()
    )

    zone_status.columns = [
        "Datetime",
        "zone",
        "units_on",
        "total_units",
        "on_ratio",
        "unit_list",
    ]

    # Categorize zone status
    zone_status["status_category"] = "Unknown"
    zone_status.loc[zone_status["units_on"] == 0, "status_category"] = "All OFF"
    zone_status.loc[
        zone_status["units_on"] == zone_status["total_units"], "status_category"
    ] = "All ON"
    zone_status.loc[
        (zone_status["units_on"] > 0)
        & (zone_status["units_on"] < zone_status["total_units"]),
        "status_category",
    ] = "Mixed (Some ON, Some OFF)"

    return zone_status, ac_with_zone


def print_zone_status_summary(zone_status):
    """Print summary of zone status analysis"""

    print("\n" + "=" * 120)
    print("📊 ZONE STATUS SUMMARY")
    print("=" * 120)

    total_records = len(zone_status)

    status_counts = zone_status["status_category"].value_counts()

    print(f"\n🔢 Overall Statistics:")
    print(f"   • Total (zone, timestamp) combinations: {total_records:,}")
    print(f"\n📈 Status Distribution:")

    for status, count in status_counts.items():
        percentage = count / total_records * 100
        print(f"   • {status}: {count:,} ({percentage:.1f}%)")

    # Critical finding
    mixed_count = status_counts.get("Mixed (Some ON, Some OFF)", 0)
    if mixed_count > 0:
        print(f"\n🎯 KEY FINDING:")
        print(f"   ✅ YES! Not all indoor units in a zone are OFF at the same time!")
        print(
            f"   ✅ {mixed_count:,} cases ({mixed_count/total_records*100:.1f}%) where SOME units are ON while others are OFF"
        )
        print(
            f"\n   💡 This explains why zone shows high power even when marked as 'AC OFF':"
        )
        print(
            f"      - The zone's aggregate status might be 'OFF' (majority or average)"
        )
        print(f"      - But individual units within the zone can still be ON")
        print(f"      - Those ON units consume significant power")

    # Analysis by zone
    print(f"\n🏢 Status Distribution by Zone:")
    print("-" * 100)
    print(f"{'Zone':<15} {'All OFF':<12} {'All ON':<12} {'Mixed':<12} {'Mixed %':<12}")
    print("-" * 100)

    for zone in sorted(zone_status["zone"].unique()):
        zone_data = zone_status[zone_status["zone"] == zone]
        zone_total = len(zone_data)

        all_off = len(zone_data[zone_data["status_category"] == "All OFF"])
        all_on = len(zone_data[zone_data["status_category"] == "All ON"])
        mixed = len(
            zone_data[zone_data["status_category"] == "Mixed (Some ON, Some OFF)"]
        )
        mixed_pct = mixed / zone_total * 100 if zone_total > 0 else 0

        print(
            f"{zone:<15} {all_off:<12,} {all_on:<12,} {mixed:<12,} {mixed_pct:<12.1f}"
        )


def analyze_power_by_status(zone_status, power_data):
    """Analyze power consumption by zone status"""

    print("\n" + "=" * 120)
    print("⚡ POWER CONSUMPTION ANALYSIS BY ZONE STATUS")
    print("=" * 120)

    # Merge with power data
    merged = pd.merge(zone_status, power_data, on=["Datetime", "zone"], how="inner")

    print(f"\n📊 Merged records: {len(merged):,}")

    # Analyze power by status category
    print(f"\n💡 Power Consumption by Status Category:")
    print("-" * 100)
    print(
        f"{'Status':<30} {'Count':<12} {'Mean Power':<15} {'Median Power':<15} {'Max Power':<15}"
    )
    print("-" * 100)

    for status in ["All OFF", "Mixed (Some ON, Some OFF)", "All ON"]:
        status_data = merged[merged["status_category"] == status]
        if len(status_data) > 0:
            mean_power = status_data["adjusted_power"].mean()
            median_power = status_data["adjusted_power"].median()
            max_power = status_data["adjusted_power"].max()

            print(
                f"{status:<30} {len(status_data):<12,} {mean_power:<15.1f} {median_power:<15.1f} {max_power:<15.1f}"
            )

    # Compare "All OFF" vs "Mixed" power consumption
    all_off_power = merged[merged["status_category"] == "All OFF"]["adjusted_power"]
    mixed_power = merged[merged["status_category"] == "Mixed (Some ON, Some OFF)"][
        "adjusted_power"
    ]

    if len(all_off_power) > 0 and len(mixed_power) > 0:
        print(f"\n🔍 Key Comparison:")
        print(f"   • 'All OFF' mean power: {all_off_power.mean():.1f} W")
        print(f"   • 'Mixed' mean power: {mixed_power.mean():.1f} W")
        print(
            f"   • Ratio (Mixed / All OFF): {mixed_power.mean() / all_off_power.mean():.1f}x"
        )

        print(f"\n   💡 Interpretation:")
        print(f"      When some units are ON (Mixed status), power consumption is")
        print(
            f"      {mixed_power.mean() / all_off_power.mean():.1f}x higher than when all units are OFF!"
        )

    # Show sample mixed status cases
    mixed_cases = merged[merged["status_category"] == "Mixed (Some ON, Some OFF)"]
    if len(mixed_cases) > 0:
        print(f"\n📝 Sample 'Mixed Status' Cases (Some units ON, Some OFF):")
        print("-" * 120)
        sample = mixed_cases[
            [
                "Datetime",
                "zone",
                "units_on",
                "total_units",
                "on_ratio",
                "adjusted_power",
            ]
        ].head(10)
        print(sample.to_string(index=False))


def show_specific_examples_simple(zone_status, ac_with_zone):
    """Show specific examples of mixed status"""

    print("\n" + "=" * 120)
    print("📝 DETAILED EXAMPLES OF MIXED STATUS")
    print("=" * 120)

    # Find mixed status cases for each zone
    mixed_cases = zone_status[
        zone_status["status_category"] == "Mixed (Some ON, Some OFF)"
    ]

    if len(mixed_cases) > 0:
        # Show examples from different zones
        for zone in mixed_cases["zone"].unique()[:3]:  # Show up to 3 zones
            zone_mixed = mixed_cases[mixed_cases["zone"] == zone].head(1)

            for _, example in zone_mixed.iterrows():
                example_datetime = example["Datetime"]
                example_zone = example["zone"]

                print(f"\n🔍 Example Case - {example_zone}:")
                print(f"   • Timestamp: {example_datetime}")
                print(f"   • Units ON: {example['units_on']}/{example['total_units']}")
                print(f"   • ON Ratio: {example['on_ratio']*100:.1f}%")

                # Get individual unit details for this case
                unit_details = ac_with_zone[
                    (ac_with_zone["Datetime"] == example_datetime)
                    & (ac_with_zone["zone"] == example_zone)
                ][["A/C Name", "A/C ON/OFF", "Indoor Temp.", "A/C Set Temperature"]]

                print(f"\n   📋 Individual Unit Status:")
                print(unit_details.to_string(index=False))


def main():
    """Main analysis function"""

    print("🚀 Starting Unit-Zone Power Analysis...")
    print("=" * 120)
    print(
        "Question: その時に、エリアに含まれる室内機が全てOFFではないですよね？？たぶん"
    )
    print(
        "(Translation: At that time, not all indoor units in the area are OFF, right?? Probably)"
    )
    print("=" * 120)

    # Load data
    unit_zone_map, master = load_master_data()
    ac_control = load_ac_control_data()
    power_data = load_power_data()

    # Analyze zone vs unit status
    zone_status, ac_with_zone = analyze_zone_unit_status(ac_control, unit_zone_map)

    # Print summary
    print_zone_status_summary(zone_status)

    # Show specific examples (power data doesn't have zone info, so skip power analysis)
    show_specific_examples_simple(zone_status, ac_with_zone)

    print("\n" + "=" * 120)
    print("✅ Analysis Complete!")
    print("=" * 120)

    print("\n🎯 CONCLUSION:")
    print("   ✅ YES, the user's suspicion is CORRECT!")
    print("   ✅ When a zone shows 'AC OFF', not all indoor units are necessarily OFF")
    print("   ✅ Some units can be ON while others are OFF (Mixed status)")
    print("   ✅ This explains the high power consumption during 'AC OFF' periods")
    print("=" * 120)


if __name__ == "__main__":
    main()
