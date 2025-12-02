from enum import Enum


class DataSourceType(str, Enum):
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


DATA_SOURCE_TYPE = DataSourceType.LOCAL

LOCAL_INPUT_DATA_PATH = "data/00_InputData"
LOCAL_LOADED_DATA_PATH = "data/06_LoadedData"

IDU_FILENAME_PREFIX = "idu_loaded"
ODU_FILENAME_PREFIX = "odu_loaded"
