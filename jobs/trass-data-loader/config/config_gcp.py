import os
from dataclasses import dataclass


@dataclass
class GCPEnv:
    PROJECT_ID = os.environ.get("PROJECT_ID") or "airux8-opti-logic"
    BUCKET_ID = os.environ.get("BUCKET_ID") or "airux8-opti-logic-prod"

    START_DATE = os.environ.get("START_DATE")
    END_DATE = os.environ.get("END_DATE")

    LOADED_DATA_PATH = os.environ.get("LOADED_DATA_PATH") or "06_LoadedData"
