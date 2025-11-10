import os
from dataclasses import dataclass


@dataclass
class GCPEnv:
    PROJECT_ID = os.environ.get("PROJECT_ID") or "airux8-opti-logic"
    PROJECT_ID_NUMBER = os.environ.get("PROJECT_ID_NUMBER") or "144706892563"
    BUCKET_NAME = os.environ.get("BUCKET_NAME") or "airux8-opti-logic-prod"

    PREPROCESSED_DATA_FOLDER = (
        os.environ.get("PREPROCESSED_DATA_FOLDER") or "00_InputData/01_PreprocessedData"
    )
    INPUT_DATA_FOLDER = os.environ.get("INPUT_DATA_FOLDER") or "00_InputData/"
    CLEAR_INPUT_DATA_FOLDER = (
        os.environ.get("CLEAR_INPUT_DATA_FOLDER") or "00_InputData/01_PreprocessedData"
    )

    MASTER_DATA_FOLDER = os.environ.get("MASTER_DATA_FOLDER") or "01_MasterData/"

    PREPROCESSED_CLEAR_DATA_FOLDER = (
        os.environ.get("PREPROCESSED_CLEAR_DATA_FOLDER")
        or "00_InputData/01_PreprocessedData"
    )

    PLANNING_CLEAR_DATA_FOLDER = (
        os.environ.get("PLANNING_CLEAR_DATA_FOLDER") or "04_PlanningData/"
    )
    WEATHER_DATA_FOLDER = os.environ.get("WEATHER_DATA_FOLDER") or "05_WeatherData/"
    WEATHER_FORECAST_FOLDER = (
        os.environ.get("WEATHER_FORECAST_FOLDER") or "05_WeatherData/02_ForecastData/"
    )
    WEATHER_HISTORICAL_FOLDER = (
        os.environ.get("WEATHER_HISTORICAL_FOLDER")
        or "05_WeatherData/01_HistoricalData/"
    )
    SERVICE_ACCOUNT_JSON = (
        os.environ.get("SERVICE_ACCOUNT_JSON") or "config/svc-airux8-optimize-key.json"
    )

    CLEA_OUT_GDRIVE_FOLDER_ID = (
        os.environ.get("CLEA_OUT_GDRIVE_FOLDER_ID")
        or "1VA9m_cIR5m9j7yfx2t-gnr1vowANRf7O"
    )
    CSV_ENCODING = os.environ.get("CSV_ENCODING") or "utf-8"
