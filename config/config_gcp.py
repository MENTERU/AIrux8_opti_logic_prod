import os
from dataclasses import dataclass


@dataclass
class GCPEnv:
    PROJECT_ID = os.environ.get("PROJECT_ID", "airux8-opti-logic")
    PROJECT_ID_NUMBER = os.environ.get("PROJECT_ID_NUMBER", "144706892563")
    BUCKET_NAME = os.environ.get("BUCKET_NAME", "airux8-opti-logic-prod")
    
    PREPROCESSED_DATA_FOLDER = os.environ.get(
        "PREPROCESSED_DATA_FOLDER", "00_InputData/01_PreprocessedData"
    )
    INPUT_DATA_FOLDER = os.environ.get("INPUT_DATA_FOLDER", "00_InputData/")
    CLEAR_INPUT_DATA_FOLDER = os.environ.get("CLEAR_INPUT_DATA_FOLDER", "00_InputData/01_PreprocessedData")

    MASTER_DATA_FOLDER = os.environ.get("MASTER_DATA_FOLDER", "01_MasterData/")

    PREPROCESSED_CLEAR_DATA_FOLDER = os.environ.get("PREPROCESSED_CLEAR_DATA_FOLDER", "00_InputData/01_PreprocessedData")

    
    PLANNING_CLEAR_DATA_FOLDER = os.environ.get("PLANNING_CLEAR_DATA_FOLDER", "04_PlanningData/")
    SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON", "config/gdrive_service_account.json")
    
    GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "PLACEHOLDER_FOR_GDRIVE_FOLDER_ID")
    CSV_ENCODING = os.environ.get("CSV_ENCODING", "utf-8")