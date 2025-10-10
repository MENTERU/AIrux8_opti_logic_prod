#!/usr/bin/env python3
"""
Generate unit schedule CSV file from AC control data for a specific period
This creates a CSV with the same structure as the planning unit schedule files
but uses raw data without mappings for the specified date range.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


def load_and_filter_data(file_path, start_date, end_date):
    """Load and filter data for the specified date range"""
    print(f"Loading data from: {file_path}")

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Convert datetime column
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Filter by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    df_filtered = df[(df["Datetime"] >= start_dt) & (df["Datetime"] < end_dt)]

    print(f"Loaded {len(df)} total records")
    print(
        f"Filtered to {len(df_filtered)} records for period {start_date} to {end_date}"
    )

    return df_filtered


def load_power_data(file_path, start_date, end_date):
    """Load and filter power meter data for the specified date range"""
    print(f"Loading power data from: {file_path}")

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Convert datetime column
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Filter by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    df_filtered = df[(df["Datetime"] >= start_dt) & (df["Datetime"] < end_dt)]

    print(f"Loaded {len(df)} total power records")
    print(
        f"Filtered to {len(df_filtered)} power records for period {start_date} to {end_date}"
    )

    return df_filtered


def create_unit_schedule_csv(df, power_df, output_file):
    """Create unit schedule CSV with the same structure as planning files"""

    # Get unique units
    units = sorted(df["A/C Name"].unique())
    print(f"Found {len(units)} unique units: {units}")

    # Create hourly timestamps for the period
    start_time = df["Datetime"].min().floor("h")
    end_time = df["Datetime"].max().floor("h")
    timestamps = pd.date_range(start=start_time, end=end_time, freq="h")
    print(
        f"Created {len(timestamps)} hourly timestamps from {start_time} to {end_time}"
    )

    # Create the output dataframe
    output_data = []

    for timestamp in timestamps:
        # Get data for this hour (take first record of each hour for each unit)
        hour_data = df[df["Datetime"].dt.floor("h") == timestamp]

        # Create row dictionary
        row = {
            "Date Time": timestamp.strftime("%Y/%m/%d %H:%M"),
            "outside_temp": None,  # We'll try to get this from the data
            "total_kwh": None,  # We'll get this from power data
        }

        # Add data for each unit
        for unit in units:
            unit_data = hour_data[hour_data["A/C Name"] == unit]

            if not unit_data.empty:
                unit_row = unit_data.iloc[0]  # Take first record if multiple

                # Add unit columns (OnOFF, Mode, SetTemp, FanSpeed)
                row[f"{unit}_OnOFF"] = unit_row["A/C ON/OFF"]
                row[f"{unit}_Mode"] = unit_row["A/C Mode"]
                row[f"{unit}_SetTemp"] = unit_row["A/C Set Temperature"]
                row[f"{unit}_FanSpeed"] = unit_row["A/C Fan Speed"]

                # Try to get outside temperature from this unit's data
                if row["outside_temp"] is None and "Outdoor Temp." in unit_row:
                    row["outside_temp"] = unit_row["Outdoor Temp."]
            else:
                # No data for this unit at this timestamp
                row[f"{unit}_OnOFF"] = None
                row[f"{unit}_Mode"] = None
                row[f"{unit}_SetTemp"] = None
                row[f"{unit}_FanSpeed"] = None

        # Get total_kwh from power data for this hour
        if not power_df.empty:
            power_hour_data = power_df[power_df["Datetime"].dt.floor("h") == timestamp]
            if not power_hour_data.empty:
                # Sum all power values for this hour (assuming there's a power column)
                power_columns = [col for col in power_hour_data.columns if 'power' in col.lower() or 'kwh' in col.lower()]
                if power_columns:
                    # Take the first power column found
                    power_col = power_columns[0]
                    row["total_kwh"] = power_hour_data[power_col].sum()
                else:
                    # If no power column found, try to sum all numeric columns
                    numeric_cols = power_hour_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        row["total_kwh"] = power_hour_data[numeric_cols].sum().sum()

        output_data.append(row)

    # Create DataFrame
    output_df = pd.DataFrame(output_data)

    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Unit schedule CSV saved to: {output_file}")

    return output_df


def main():
    """Main function"""

    # Configuration
    data_file = "data/02_PreprocessedData/Clea/ac_control_processed_Clea.csv"
    power_file = "data/02_PreprocessedData/Clea/power_meter_processed_Clea.csv"
    start_date = "2024-10-09 00:00:00"
    end_date = "2024-10-12 00:00:00"
    output_file = "unit_schedule_hourly_raw_20241009_20241012.csv"

    print("=" * 80)
    print("UNIT SCHEDULE CSV GENERATOR")
    print("=" * 80)
    print(f"Data file: {data_file}")
    print(f"Power file: {power_file}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    # Load and filter data
    df = load_and_filter_data(data_file, start_date, end_date)
    power_df = load_power_data(power_file, start_date, end_date)

    if df.empty:
        print("No data found in the specified date range!")
        return

    # Create unit schedule CSV
    output_df = create_unit_schedule_csv(df, power_df, output_file)

    # Print summary
    print("\n" + "=" * 80)
    print("CSV GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Rows: {len(output_df)}")
    print(f"Columns: {len(output_df.columns)}")
    print(f"Units: {len([col for col in output_df.columns if '_OnOFF' in col])}")
    print("\nFirst few rows:")
    print(output_df.head(3).to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
