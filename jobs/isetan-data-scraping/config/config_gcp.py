import os
from dataclasses import dataclass


@dataclass
class GCPEnv:
    PROJECT_ID = os.environ.get("PROJECT_ID") or "airux8-opti-logic"
    PROJECT_ID_NUMBER = os.environ.get("PROJECT_ID_NUMBER") or "144706892563"
    BUCKET_NAME = os.environ.get("BUCKET_NAME") or "airux8-opti-logic-prod"

    # Store and BigQuery configuration for IsetanMitsukoshi scraping
    STORE_NAME = os.environ.get("STORE_NAME") or "IsetanMitsukoshi"
    BQ_DATASET_ISETAN = os.environ.get("BQ_DATASET_ISETAN") or "IsetanMitsukoshi"
    BQ_TABLE_AC_CONTROL_RAW = (
        os.environ.get("BQ_TABLE_AC_CONTROL_RAW") or "ac_control_raw"
    )
    BQ_TABLE_AC_POWER_METER_RAW = (
        os.environ.get("BQ_TABLE_AC_POWER_METER_RAW") or "ac_power_meter_raw"
    )
    LOGIN_INFO_SECRET_NAME = (
        os.environ.get("LOGIN_INFO_SECRET_NAME") or "AIRUX8_WEB_LOGIN_INFO"
    )

    INPUT_DATA_FOLDER = os.environ.get("INPUT_DATA_FOLDER") or "00_InputData/"

    # GCS paths for Clea data storage
    CLEA_AC_POWER_METER_PATH = "00_InputData/IsetanMitsukoshi/ac-power-meter/"
    CLEA_AC_CONTROL_PATH = "00_InputData/IsetanMitsukoshi/ac-control/"

    SERVICE_ACCOUNT_JSON = (
        os.environ.get("JOB_ISETAN_DATA_SCRAPING_SA")
        or "config/job-isetan-data-scraping-sa.json"
    )
    SERVICE_ACCOUNT_SECRET_NAME = (
        os.environ.get("JOB_ISETAN_DATA_SCRAPING_SA") or "JOB_ISETAN_DATA_SCRAPING_SA"
    )
