import logging
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from config.config import *
from config.config_gcp import GCPEnv
from menteru_tools.gcp_service import storage

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def dates_within_filename_range(
    filename: str, start_date_str: str, end_date_str: str
) -> bool:
    """
    Check if the file's date range overlaps with the input date range,
    even if the input dates are in reverse order.
    """
    try:
        # Convert input strings to datetime objects
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        # Normalize input range
        input_start = min(start_date, end_date)
        input_end = max(start_date, end_date)

        # Extract file date range
        match = re.search(r"(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})", filename)
        if not match:
            return False

        file_start = datetime.strptime(match.group(1), "%Y-%m-%d")
        file_end = datetime.strptime(match.group(2), "%Y-%m-%d")

        # Check if ranges overlap
        return input_start <= file_end and input_end >= file_start

    except Exception as e:
        logger.error(f"Error processing filename {filename}: {e}")
        return False


def local_files_list(path: str) -> List[str]:
    """List local files recursively from the given path."""
    if not os.path.exists(path):
        logger.warning(f"Local folder not found: {path}")
        return []
    logger.info(f"Scanning local folder: {path}")
    return [
        os.path.join(root, filename)
        for root, _, files in os.walk(path)
        for filename in files
    ]


def load_raw(
    input_path: str, gc_storage: storage.Storage, start_date: str, end_date: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load raw AC control (IDU) and power meter (ODU) data based on source type."""
    logger.info(f"Loading raw data from: {input_path}")

    all_paths = (
        local_files_list(input_path)
        if DATA_SOURCE_TYPE == DataSourceType.LOCAL
        else gc_storage.list(input_path)
    )

    ac_paths = [
        p
        for p in all_paths
        if "/ac-control/" in p
        and p.endswith(".csv")
        and dates_within_filename_range(p, start_date, end_date)
    ]
    pm_paths = [
        p
        for p in all_paths
        if "/ac-power-meter/" in p
        and p.endswith(".csv")
        and dates_within_filename_range(p, start_date, end_date)
    ]

    logger.info(f"AC files: {len(ac_paths)} | ODU files: {len(pm_paths)}")

    idu = load_files(ac_paths, gc_storage)
    odu = load_files(pm_paths, gc_storage)

    logger.info(
        f"Loaded: IDU={idu.shape if idu is not None else 'None'} | "
        f"ODU={odu.shape if odu is not None else 'None'}"
    )
    return idu, odu


def load_files(paths: List[str], gc_storage: storage.Storage) -> Optional[pd.DataFrame]:
    """Load CSV files into a single DataFrame with error handling."""
    if not paths:
        logger.warning("No valid file paths provided.")
        return None

    dfs = []
    for p in paths:
        try:
            logger.info(f"Reading: {p}")
            df = (
                pd.read_csv(p, low_memory=False)
                if DATA_SOURCE_TYPE == DataSourceType.LOCAL
                else gc_storage.read_csv(p)
            )
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {p}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None


def unify_datetime(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Normalize datetime column to UTC without timezone info."""
    cols = [c for c in df.columns if "datetime" in c.lower() or "日時" in c]
    if not cols:
        logger.warning("⚠️ No datetime-like column found.")
        return None, None

    col = cols[0]
    logger.info(f"Datetime column unified: {col}")

    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df[col] = df[col].dt.tz_localize(None).dt.floor("min")

    return df, col


def compose_idu(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Clean and standardize IDU dataset."""
    if df is None:
        return None

    df, dt_col = unify_datetime(df)
    if df is None:
        return None

    out_df = df.sort_values(dt_col, ascending=False)

    # Rename final columns to standardized names
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

    out_df["ac_on_off"] = np.where(out_df["ac_on_off"] == "OFF", 0, 1)

    logger.info("IDU composition complete.")
    return out_df


def compose_odu(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Clean and standardize ODU dataset and compute energy."""
    if df is None:
        return None

    df, dt_col = unify_datetime(df)
    if df is None:
        return None

    phase_cols = [col for col in df.columns if col.startswith("Phase")]
    df["total_kwh"] = df[phase_cols].sum(axis=1) if phase_cols else df.get("Phase A", 0)

    out_df = df.sort_values(dt_col, ascending=False)

    out_df = out_df.rename(columns={"Datetime": "measured_at"})

    out_df["odu_id"] = (
        out_df["Mesh ID"].astype(str) + "-" + out_df["PM Addr ID"].astype(str)
    )

    out_df = out_df[["measured_at", "odu_id", "total_kwh"]]

    logger.info("ODU composition complete.")
    return out_df


def main():

    logger.info("🚀 Starting data loading pipeline...")

    start_date = (
        GCPEnv.START_DATE if GCPEnv.START_DATE else datetime.now().strftime("%Y-%m-%d")
    )

    end_date = GCPEnv.END_DATE if GCPEnv.END_DATE else start_date

    gc_storage_obj = storage.Storage(
        project_id=GCPEnv.PROJECT_ID, bucket_id=GCPEnv.BUCKET_ID
    )

    if DATA_SOURCE_TYPE == DataSourceType.LOCAL:
        store_names = [
            n
            for n in os.listdir(LOCAL_INPUT_DATA_PATH)
            if os.path.isdir(os.path.join(LOCAL_INPUT_DATA_PATH, n))
        ]
    else:
        store_names = gc_storage_obj.list_folders(GCPEnv.INPUT_DATA_PATH)

    logger.info(f"Stores detected: {store_names}")

    for store_name in store_names:
        logger.info(f"📌 Processing store: {store_name}")

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

        if DATA_SOURCE_TYPE == DataSourceType.LOCAL:
            os.makedirs(output_prefix, exist_ok=True)
        else:
            gc_storage_obj.makedirs(output_prefix)

        idu_raw, odu_raw = load_raw(input_prefix, gc_storage_obj, start_date, end_date)
        idu = compose_idu(idu_raw)
        odu = compose_odu(odu_raw)

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


if __name__ == "__main__":
    main()
