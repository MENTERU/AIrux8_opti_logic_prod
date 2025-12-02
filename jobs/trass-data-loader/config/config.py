from enum import Enum


class DataSourceType(str, Enum):
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


DATA_SOURCE_TYPE = DataSourceType.REMOTE

LOCAL_MASTER_DATA_PATH = "data/01_MasterData"
LOCAL_LOADED_DATA_PATH = "data/06_LoadedData"

IDU_FILENAME_PREFIX = "idu_loaded"
ODU_FILENAME_PREFIX = "odu_loaded"
