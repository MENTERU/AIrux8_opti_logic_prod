import os
from dataclasses import dataclass


def _get_env_with_default(key: str, default: str) -> str:
    """Get environment variable with default, treating empty strings as missing."""
    value = os.environ.get(key, default)
    return value if value else default


@dataclass
class GCPEnv:
    PROJECT_ID = _get_env_with_default("PROJECT_ID", "airux8-opti-logic")
    PROJECT_ID_NUMBER = _get_env_with_default("PROJECT_ID_NUMBER", "144706892563")
    BUCKET_NAME = _get_env_with_default("BUCKET_NAME", "airux8-opti-logic-prod")

    PREPROCESSED_DATA_FOLDER = _get_env_with_default(
        "PREPROCESSED_DATA_FOLDER", "00_InputData/01_PreprocessedData"
    )
    INPUT_DATA_FOLDER = _get_env_with_default("INPUT_DATA_FOLDER", "00_InputData/")
    CLEAR_INPUT_DATA_FOLDER = _get_env_with_default(
        "CLEAR_INPUT_DATA_FOLDER", "00_InputData/01_PreprocessedData"
    )

    MASTER_DATA_FOLDER = _get_env_with_default("MASTER_DATA_FOLDER", "01_MasterData/")

    PREPROCESSED_CLEAR_DATA_FOLDER = _get_env_with_default(
        "PREPROCESSED_CLEAR_DATA_FOLDER", "00_InputData/01_PreprocessedData"
    )

    PLANNING_CLEAR_DATA_FOLDER = _get_env_with_default(
        "PLANNING_CLEAR_DATA_FOLDER", "04_PlanningData/"
    )
    SERVICE_ACCOUNT_JSON = _get_env_with_default(
        "SERVICE_ACCOUNT_JSON", "config/gdrive_service_account.json"
    )

    GDRIVE_FOLDER_ID = _get_env_with_default(
        "GDRIVE_FOLDER_ID", "PLACEHOLDER_FOR_GDRIVE_FOLDER_ID"
    )
    CSV_ENCODING = _get_env_with_default("CSV_ENCODING", "utf-8")
