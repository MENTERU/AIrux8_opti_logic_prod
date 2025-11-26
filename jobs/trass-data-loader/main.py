import logging
import os
import re
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from config.config import *
from config.config_gcp import GCPEnv
from menteru_tools.gcp_service import storage

# Configure the root logger to output to standard output with a specific format.
# Level INFO ensures we capture flow and key events, but not debug noise.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def dates_within_filename_range(
    filename: str, start_date_str: str, end_date_str: str
) -> bool:
    """
    Checks if a date range found within a filename overlaps with the requested time window.

    This function extracts a date range pattern (YYYY-MM-DD to YYYY-MM-DD) from the
    filename and compares it against the provided start and end dates.

    Args:
        filename (str): The name or path of the file to check.
        start_date_str (str): The start of the requested period (format: YYYY-MM-DD HH:MM:SS).
        end_date_str (str): The end of the requested period (format: YYYY-MM-DD HH:MM:SS).

    Returns:
        bool: True if the file's date range overlaps with the input range, False otherwise.

    Raises:
        ValueError: If date parsing fails unexpectedly (caught and logged).
    """
    try:
        # Convert input strings to datetime objects for comparison
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")

        # Normalize input range to ensure start <= end (handling potential user error)
        input_start = min(start_date, end_date)
        input_end = max(start_date, end_date)

        # Regex to find a date range pattern in the filename (e.g., "data_2023-01-01-2023-01-31.csv")
        match = re.search(r"(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})", filename)
        if not match:
            # If the filename doesn't contain the expected date pattern, we skip it.
            return False

        file_start = datetime.strptime(match.group(1), "%Y-%m-%d")
        file_end = datetime.strptime(match.group(2), "%Y-%m-%d")

        # Check for intersection between the requested interval and the file's interval
        return (
            file_start <= input_start <= file_end or file_start <= input_end <= file_end
        )

    except Exception as e:
        logger.error(f"Error processing filename {filename}: {e}")
        raise


def local_files_list(path: str) -> List[str]:
    """
    Recursively lists all files in a local directory.

    Args:
        path (str): The root directory path to search.

    Returns:
        List[str]: A list of full file paths found. Returns an empty list if the path doesn't exist.
    """
    if not os.path.exists(path):
        logger.warning(f"Local folder not found: {path}")
        return []

    logger.info(f"Scanning local folder: {path}")

    # Walk through the directory tree and collect all file paths
    return [
        os.path.join(root, filename)
        for root, _, files in os.walk(path)
        for filename in files
    ]


def load_raw(
    input_path: str, gc_storage: storage.Storage, start_date: str, end_date: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Orchestrates the loading of raw data for both AC Control (IDU) and Power Meters (ODU).

    It distinguishes between IDU and ODU files based on directory keywords in the path
    ('/ac-control/' vs '/ac-power-meter/') and filters files by date range.

    Args:
        input_path (str): The base path (local or bucket prefix) to search.
        gc_storage (storage.Storage): The GCP storage interface object.
        start_date (str): Filter start date.
        end_date (str): Filter end date.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: A tuple containing
        (IDU DataFrame, ODU DataFrame). Returns (None, None) if loading fails or data is empty.
    """
    logger.info(f"Loading raw data from: {input_path}")

    # Determine whether to list files from local disk or Google Cloud Storage
    all_paths = (
        local_files_list(input_path)
        if DATA_SOURCE_TYPE == DataSourceType.LOCAL
        else gc_storage.list(input_path)
    )

    # Filter paths for AC Control (IDU) data
    ac_paths = [
        p
        for p in all_paths
        if "/ac-control/" in p
        and p.endswith(".csv")
        and dates_within_filename_range(p, start_date, end_date)
    ]

    # Filter paths for Power Meter (ODU) data
    pm_paths = [
        p
        for p in all_paths
        if "/ac-power-meter/" in p
        and p.endswith(".csv")
        and dates_within_filename_range(p, start_date, end_date)
    ]

    logger.info(f"AC files found: {len(ac_paths)} | ODU files found: {len(pm_paths)}")

    # Load the actual CSV content
    idu = load_files(ac_paths, gc_storage)
    odu = load_files(pm_paths, gc_storage)

    logger.info(
        f"Loaded Data Shapes: IDU={idu.shape if idu is not None else 'None'} | "
        f"ODU={odu.shape if odu is not None else 'None'}"
    )

    return idu, odu


def load_files(paths: List[str], gc_storage: storage.Storage) -> Optional[pd.DataFrame]:
    """
    Reads a list of CSV file paths and concatenates them into a single DataFrame.

    Args:
        paths (List[str]): List of file paths to read.
        gc_storage (storage.Storage): GCP storage object (used if source is not local).

    Returns:
        Optional[pd.DataFrame]: A combined DataFrame, or None if no paths provided or loading failed.
    """
    if not paths:
        logger.warning("No valid file paths provided for loading.")
        return None

    dfs = []
    for p in paths:
        try:
            logger.info(f"Reading file: {p}")
            # low_memory=False is used to prevent Dtype warnings on large mixed-type files
            df = (
                pd.read_csv(p, low_memory=False)
                if DATA_SOURCE_TYPE == DataSourceType.LOCAL
                else gc_storage.read_csv(p)
            )
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load file {p}: {e}")

    # Concatenate all loaded dataframes; return None if the list is empty
    return pd.concat(dfs, ignore_index=True) if dfs else None


def unify_datetime(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Identifies and normalizes the primary datetime column in the DataFrame.

    It converts the datetime to UTC, removes timezone information (making it naive),
    and floors the time to the nearest minute.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: The modified DataFrame and the name
        of the datetime column found. Returns (None, None) if no datetime column is identified.
    """
    # Dynamic search for a column name containing "datetime" or Japanese "日時"
    cols = [c for c in df.columns if "datetime" in c.lower() or "日時" in c]
    if not cols:
        logger.warning("⚠️ No datetime-like column found in the dataset.")
        return None, None

    col = cols[0]
    logger.info(f"Datetime column identified and unified: {col}")

    # Normalize to datetime object, ensure UTC, remove timezone info, and round down to minute
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df[col] = df[col].dt.tz_localize(None).dt.floor("min")

    return df, col


def compose_idu(
    df: pd.DataFrame, start_date: str, end_date: str
) -> Optional[pd.DataFrame]:
    """
    Processes and standardizes AC Indoor Unit (IDU) data.

    Steps:
    1. Unify datetime.
    2. Rename columns to standard English keys.
    3. Filter data based on the requested date range.
    4. Convert textual ON/OFF status to binary integer.

    Args:
        df (pd.DataFrame): Raw IDU data.
        start_date (str): Filter start date.
        end_date (str): Filter end date.

    Returns:
        Optional[pd.DataFrame]: Cleaned IDU DataFrame.
    """
    if df is None:
        return None

    df, dt_col = unify_datetime(df)
    if df is None:
        return None

    # Sort by time descending
    out_df = df.sort_values(dt_col, ascending=False)

    # Standardize column names for downstream consistency
    out_df = out_df.rename(
        columns={
            "Datetime": "measured_at",
            "A/C Name": "idu_id",
            "Indoor Temp.": "indoor_temperature",
            "A/C Set Temperature": "ac_set_temperature",
            "A/C ON/OFF": "ac_on_off",
            "A/C Mode": "ac_mode",
            "A/C Fan Speed": "ac_fan_speed",
        }
    )

    # Select only relevant columns
    out_df = out_df[
        [
            "measured_at",
            "idu_id",
            "ac_set_temperature",
            "indoor_temperature",
            "ac_on_off",
            "ac_mode",
            "ac_fan_speed",
        ]
    ]

    # Apply strict date filtering
    out_df = out_df[
        (out_df["measured_at"] >= pd.to_datetime(start_date))
        & (out_df["measured_at"] <= pd.to_datetime(end_date))
    ]

    # Convert "OFF" to 0 and "ON" (or anything else) to 1
    out_df["ac_on_off"] = np.where(out_df["ac_on_off"] == "OFF", 0, 1)

    logger.info("IDU composition (cleaning/transforming) complete.")
    return out_df


def compose_odu(
    df: pd.DataFrame, start_date: str, end_date: str
) -> Optional[pd.DataFrame]:
    """
    Processes and standardizes AC Outdoor Unit (ODU) / Power Meter data.

    Steps:
    1. Unify datetime.
    2. Calculate total energy (kWh) by summing phases if available.
    3. Construct a unique ODU ID from Mesh ID and PM Addr ID.
    4. Filter by date range.

    Args:
        df (pd.DataFrame): Raw ODU data.
        start_date (str): Filter start date.
        end_date (str): Filter end date.

    Returns:
        Optional[pd.DataFrame]: Cleaned ODU DataFrame.
    """
    if df is None:
        return None

    df, dt_col = unify_datetime(df)
    if df is None:
        return None

    # Calculate total kWh: Sum Phase A/B/C if they exist, otherwise use Phase A
    phase_cols = [col for col in df.columns if col.startswith("Phase")]
    df["total_kwh"] = df[phase_cols].sum(axis=1) if phase_cols else df.get("Phase A", 0)

    out_df = df.sort_values(dt_col, ascending=False)
    out_df = out_df.rename(columns={"Datetime": "measured_at"})

    # Create a composite key for the Outdoor Unit ID
    out_df["odu_id"] = (
        out_df["Mesh ID"].astype(str) + "-" + out_df["PM Addr ID"].astype(str)
    )

    out_df = out_df[["measured_at", "odu_id", "total_kwh"]]

    # Apply strict date filtering
    out_df = out_df[
        (out_df["measured_at"] >= pd.to_datetime(start_date))
        & (out_df["measured_at"] <= pd.to_datetime(end_date))
    ]

    logger.info("ODU composition (cleaning/transforming) complete.")
    return out_df


def main():
    """
    Main execution pipeline.

    1. Determines time range (defaults to yesterday -> today if not in ENV).
    2. Connects to GCP Storage (or prepares local paths).
    3. Iterates through available 'stores' (folders).
    4. Loads, Composes, and Saves IDU and ODU data for each store.
    """
    try:
        logger.info("🚀 Starting data loading pipeline...")

        # Initialize timezone (Tokyo) for default date calculations
        tokyo_tz = pytz.timezone("Asia/Tokyo")
        now_dt = datetime.now(tokyo_tz)

        # Determine Date Range: Use Env vars if available, otherwise default to [Now-1day -> Now]
        end_date = (
            GCPEnv.END_DATE if GCPEnv.END_DATE else now_dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        start_date = (
            GCPEnv.START_DATE
            if GCPEnv.START_DATE
            else (
                datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S") - timedelta(days=1)
            ).strftime("%Y-%m-%d %H:%M:%S")
        )

        # Validate date consistency
        if datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") > datetime.strptime(
            end_date, "%Y-%m-%d %H:%M:%S"
        ):
            raise Exception("Input dates inconsistent, must be START_DATE <= END_DATE")

        logger.info(f"Loading data from {start_date} to {end_date}...")

        # Initialize GCP Storage client
        gc_storage_obj = storage.Storage(
            project_id=GCPEnv.PROJECT_ID, bucket_id=GCPEnv.BUCKET_ID
        )

        # List folders (stores) to process
        if DATA_SOURCE_TYPE == DataSourceType.LOCAL:
            store_names = [
                n
                for n in os.listdir(LOCAL_INPUT_DATA_PATH)
                if os.path.isdir(os.path.join(LOCAL_INPUT_DATA_PATH, n))
            ]
        else:
            store_names = gc_storage_obj.list_folders(GCPEnv.INPUT_DATA_PATH)

        logger.info(f"Stores detected: {store_names}")

        # Iterate over each store to process specific data
        for store_name in store_names:
            logger.info(f"📌 Processing store: {store_name}")

            # Define Input/Output paths based on environment
            input_prefix = os.path.join(
                (
                    LOCAL_INPUT_DATA_PATH
                    if DATA_SOURCE_TYPE == DataSourceType.LOCAL
                    else GCPEnv.INPUT_DATA_PATH
                ),
                store_name,
            )
            output_prefix = os.path.join(
                (
                    LOCAL_LOADED_DATA_PATH
                    if DATA_SOURCE_TYPE == DataSourceType.LOCAL
                    else GCPEnv.LOADED_DATA_PATH
                ),
                store_name,
            )

            # Ensure output directory exists
            if DATA_SOURCE_TYPE == DataSourceType.LOCAL:
                os.makedirs(output_prefix, exist_ok=True)
            else:
                gc_storage_obj.makedirs(output_prefix)

            # --- ETL Process ---
            # 1. Load Raw Data
            idu_raw, odu_raw = load_raw(
                input_prefix, gc_storage_obj, start_date, end_date
            )
            # 2. Transform Data
            idu = compose_idu(idu_raw, start_date, end_date)
            odu = compose_odu(odu_raw, start_date, end_date)

            # 3. Save IDU Data
            if idu is not None:
                idu_output = os.path.join(output_prefix, f"{IDU_FILENAME_PREFIX}.csv")
                (
                    idu.to_csv(idu_output, index=False)
                    if DATA_SOURCE_TYPE == DataSourceType.LOCAL
                    else gc_storage_obj.write_csv(idu, idu_output)
                )
                logger.info(f"IDU saved: {idu_output}")
            else:
                logger.warning(f"No IDU data to save for store {store_name}")

            # 4. Save ODU Data
            if odu is not None:
                odu_output = os.path.join(output_prefix, f"{ODU_FILENAME_PREFIX}.csv")
                (
                    odu.to_csv(odu_output, index=False)
                    if DATA_SOURCE_TYPE == DataSourceType.LOCAL
                    else gc_storage_obj.write_csv(odu, odu_output)
                )
                logger.info(f"ODU saved: {odu_output}")
            else:
                logger.warning(f"No ODU data to save for store {store_name}")

        logger.info("🎉 Data pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during data loading pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
