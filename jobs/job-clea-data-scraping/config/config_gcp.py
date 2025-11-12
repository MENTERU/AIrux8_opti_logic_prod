import os
from dataclasses import dataclass


@dataclass
class GCPEnv:
    PROJECT_ID = os.environ.get("PROJECT_ID") or "airux8-opti-logic"
    PROJECT_ID_NUMBER = os.environ.get("PROJECT_ID_NUMBER") or "144706892563"
    BUCKET_NAME = os.environ.get("BUCKET_NAME") or "airux8-opti-logic-prod"

    INPUT_DATA_FOLDER = os.environ.get("INPUT_DATA_FOLDER") or "00_InputData/"

    # GCS paths for Clea data storage
    CLEA_AC_POWER_METER_PATH = "00_InputData/Clea/ac-power-meter/"
    CLEA_AC_CONTROL_PATH = "00_InputData/Clea/ac-control/"

    SERVICE_ACCOUNT_JSON = (
        os.environ.get("JOB_CLEA_DATA_SCRAPING_SA")
        or "config/job-clea-data-scraping-sa.json"
    )
    SERVICE_ACCOUNT_SECRET_NAME = (
        os.environ.get("JOB_CLEA_DATA_SCRAPING_SA") or "JOB_CLEA_DATA_SCRAPING_SA"
    )
