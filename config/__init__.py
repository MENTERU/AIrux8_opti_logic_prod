import importlib
import os

# Resolve ACCESS_INFORMATION without hard dependency on local file
ACCESS_INFORMATION = os.getenv("ACCESS_INFORMATION")
if not ACCESS_INFORMATION:
    try:
        _pi = importlib.import_module("config.private_information")
        ACCESS_INFORMATION = getattr(_pi, "ACCESS_INFORMATION", "")
    except Exception:
        ACCESS_INFORMATION = ""

from .utils import get_data_path, load_config, upload_file  # noqa
