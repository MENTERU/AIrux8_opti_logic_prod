import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from config.config import *
from config.config_gcp import GCPEnv
from google.cloud import pubsub_v1
from menteru_tools.gcp_service import bigquery, storage

# ==========================================
# Logging Configuration
# ==========================================
# Configure the root logger to output to standard output with a specific format.
# Level INFO ensures we capture flow and key events, but not debug noise.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compose_idu(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Processes and standardizes AC Indoor Unit (IDU) data.

    Steps:
    1. Rename columns to standard English keys.
    2. Convert textual ON/OFF status to binary integer.

    Args:
        df (pd.DataFrame): Raw IDU data.
        start_date (str): Filter start date.
        end_date (str): Filter end date.

    Returns:
        Optional[pd.DataFrame]: Cleaned IDU DataFrame.
    """
    if df is None:
        return None

    # Standardize column names for downstream consistency
    out_df = df.rename(
        columns={
            "Datetime": "measured_at",
            "AC_Name": "idu_id",
            "Indoor_Temp": "indoor_temperature",
            "AC_Set_Temperature": "ac_set_temperature",
            "AC_ON_OFF": "ac_on_off",
            "AC_Mode": "ac_mode",
            "AC_Fan_Speed": "ac_fan_speed",
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

    out_df["measured_at"] = (
        pd.to_datetime(out_df["measured_at"]).dt.floor("s").dt.tz_localize(None)
    )
    # Sort by time descending
    out_df = out_df.sort_values("measured_at", ascending=False)

    # Convert "OFF" to 0 and "ON" (or anything else) to 1
    out_df["ac_on_off"] = np.where(out_df["ac_on_off"] == "OFF", 0, 1)

    logger.info("IDU composition (cleaning/transforming) complete.")
    return out_df


def compose_odu(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Processes and standardizes AC Outdoor Unit (ODU) / Power Meter data.

    Steps:
    1. Calculate total energy (kWh) by summing phases if available.
    2. Construct a unique ODU ID from Mesh ID and PM Addr ID.

    Args:
        df (pd.DataFrame): Raw ODU data.
        start_date (str): Filter start date.
        end_date (str): Filter end date.

    Returns:
        Optional[pd.DataFrame]: Cleaned ODU DataFrame.
    """
    if df is None:
        return None

    # Calculate total kWh: Sum Phase A/B/C if they exist, otherwise use Phase A
    phase_cols = [col for col in df.columns if col.startswith("Phase")]
    df["total_kwh"] = df[phase_cols].sum(axis=1) if phase_cols else df.get("Phase_A", 0)

    out_df = df.rename(columns={"Datetime": "measured_at"})

    # Create a composite key for the Outdoor Unit ID
    out_df["odu_id"] = (
        out_df["Mesh_ID"].astype(str) + "-" + out_df["PM_Addr_ID"].astype(str)
    )

    out_df = out_df[["measured_at", "odu_id", "total_kwh"]]

    out_df["measured_at"] = (
        pd.to_datetime(out_df["measured_at"]).dt.floor("s").dt.tz_localize(None)
    )

    # Sort by time descending
    out_df = out_df.sort_values("measured_at", ascending=False)

    logger.info("ODU composition (cleaning/transforming) complete.")
    return out_df


def list_datasets(gc_storage_client: storage.Storage):
    master_list = (
        os.listdir(LOCAL_MASTER_DATA_PATH)
        if DATA_SOURCE_TYPE == DataSourceType.LOCAL
        else gc_storage_client.list(prefix=GCPEnv.MASTER_DATA_PATH)
    )
    datasets = [
        master_filename.split("/")[-1].removeprefix("MASTER_").removesuffix(".xlsx")
        for master_filename in master_list
        if "MASTER" in master_filename
    ]
    return datasets


def main():
    """
    Main execution pipeline.

    1. Determines time range (defaults to yesterday -> today if not in ENV).
    2. Iterates through available 'stores' (folders).
    3. Loads, Composes, and Saves IDU and ODU data for each store.
    """
    try:
        logger.info("🚀 Starting data loading pipeline...")

        # Initialize timezone (Tokyo) for default date calculations
        tokyo_tz = pytz.timezone("Asia/Tokyo")
        now_dt = datetime.now(tokyo_tz)

        # Determine Date Range: Use Env vars if available, otherwise default to [Now-1day -> Now]
        end_date = (
            GCPEnv.END_DATE
            if GCPEnv.END_DATE
            else (now_dt - timedelta(days=1)).strftime("%Y-%m-%d 23:59:59")
        )

        start_date = (
            GCPEnv.START_DATE
            if GCPEnv.START_DATE
            else (datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")).strftime(
                "%Y-%m-%d 00:00:00"
            )
        )

        # Validate date consistency
        if datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") > datetime.strptime(
            end_date, "%Y-%m-%d %H:%M:%S"
        ):
            raise Exception("Input dates inconsistent, must be START_DATE <= END_DATE")

        logger.info(f"Loading data from {start_date} to {end_date}...")

        # Initialize GCP Storage client
        gc_storage_client = storage.Storage(
            project_id=GCPEnv.PROJECT_ID, bucket_id=GCPEnv.BUCKET_ID
        )

        bigquery_client = bigquery.BigQuery(project_id=GCPEnv.PROJECT_ID)

        if GCPEnv.FACILITY_ID:
            store_names = [GCPEnv.FACILITY_ID]
        else:
            store_names = list_datasets(gc_storage_client)

        logger.info(f"Stores detected: {store_names}")

        failed_facilities = set([])

        # Iterate over each store to process specific data
        for store_name in store_names:

            try:
                logger.info(f"📌 Processing store: {store_name}")

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
                    gc_storage_client.makedirs(output_prefix)

                try:
                    # 1. Load Raw Data
                    query = f"""
                        SELECT * FROM `{GCPEnv.PROJECT_ID}.{store_name}.table_name` 
                        WHERE TIMESTAMP_TRUNC(Datetime, SECOND) >= TIMESTAMP('{start_date}') 
                        and TIMESTAMP_TRUNC(Datetime, SECOND) <= TIMESTAMP('{end_date}')
                    """

                    idu_raw = bigquery_client.query(
                        sql=query.replace("table_name", "ac_control_raw")
                    )
                    odu_raw = bigquery_client.query(
                        sql=query.replace("table_name", "ac_power_meter_raw")
                    )

                except Exception as e:
                    raise Exception(f"Error while loading raw data from BigQuery: {e}")

                # 2. Transform Data
                idu = compose_idu(idu_raw)
                odu = compose_odu(odu_raw)

                # 3. Save IDU Data
                if idu is not None and not idu.empty:
                    idu_output = os.path.join(
                        output_prefix, f"{IDU_FILENAME_PREFIX}.csv"
                    )
                    (
                        idu.to_csv(idu_output, index=False)
                        if DATA_SOURCE_TYPE == DataSourceType.LOCAL
                        else gc_storage_client.write_csv(idu, idu_output)
                    )
                    logger.info(f"IDU saved: {idu_output}")
                else:
                    logger.warning(f"No IDU data to save for store {store_name}")
                    raise Exception(f"No IDU data to save for store {store_name}")

                # 4. Save ODU Data
                if odu is not None and not odu.empty:
                    odu_output = os.path.join(
                        output_prefix, f"{ODU_FILENAME_PREFIX}.csv"
                    )
                    (
                        odu.to_csv(odu_output, index=False)
                        if DATA_SOURCE_TYPE == DataSourceType.LOCAL
                        else gc_storage_client.write_csv(odu, odu_output)
                    )
                    logger.info(f"ODU saved: {odu_output}")
                else:
                    logger.warning(f"No ODU data to save for store {store_name}")
                    raise Exception(f"No ODU data to save for store {store_name}")

                publisher = pubsub_v1.PublisherClient()
                topic = publisher.topic_path(
                    GCPEnv.PROJECT_ID,
                    f"trigger-svc-central-hvac-preprocess-{store_name}",
                )

                event = {
                    "specversion": "1.0",
                    "id": str(uuid.uuid4()),
                    "source": "//run.googleapis.com/services/job-trass-data-loader-prod",
                    "type": "job-trass-data-loader-prod.complete",
                    "datacontenttype": "application/json",
                    "subject": "job-trass-data-loader-prod",
                    "data": {
                        "facility_id": store_name,
                        "bucket_id": GCPEnv.BUCKET_ID,
                        "gcs_project_id": GCPEnv.PROJECT_ID,
                    },
                }

                future = publisher.publish(topic, json.dumps(event).encode("utf-8"))
                future.result()

                logging.info(f"✅ [{store_name}] Pub/Sub message published: {event}!")

            except Exception as e:
                logger.error(f"Facility {store_name} failed: {e}")
                failed_facilities.add(store_name)

        if failed_facilities:
            logger.warning(
                "Data pipeline completed with failures for: %s",
                ", ".join(failed_facilities),
            )
            return 1

        logger.info("🎉 Data pipeline completed successfully for all facilities!")

        return 0

    except Exception as e:
        logger.error(f"Error during data loading pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
