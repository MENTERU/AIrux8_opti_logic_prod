import os
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from config import ACCESS_INFORMATION


def ch_base_dir():
    """現在のディレクトリを変更し、元のディレクトリを保存する関数"""
    global original_dir
    original_dir = os.getcwd()  # 現在のディレクトリを保存
    home_dir = os.path.expanduser("~")  # ホームディレクトリのパスを取得
    os.chdir(home_dir)  # ホームディレクトリに移動
    return os.getcwd()  # 新しいカレントディレクトリを返す


def reverse_dir():
    """元のディレクトリに戻る関数"""
    os.chdir(original_dir)  # 保存していた元のディレクトリに戻る


def change_dir(temp_folder_path):
    """指定されたディレクトリに移動し、元のディレクトリを保存する関数"""
    global original_dir
    original_dir = os.getcwd()  # 現在のディレクトリを保存
    os.chdir(temp_folder_path)  # 指定されたディレクトリに移動


def upload_file(df: pd.DataFrame, name, temp_folder_path):
    """DataFrameをCSVファイルとして保存する関数"""
    change_dir(temp_folder_path)  # 指定されたディレクトリに移動
    df.to_csv(f"{name}.csv", index=False, encoding="utf-8")  # CSVとして保存
    reverse_dir()  # 元のディレクトリに戻る


def join_paths(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.join(*seq)


def load_config(use_remote_paths: bool = False):
    """設定ファイルを読み込む関数

    Args:
        use_remote_paths (bool): True の場合 remote_paths を使用、False の場合 local_paths を使用
    """
    root_path = os.path.dirname(os.path.dirname(__file__))  # ルートパスを取得
    config_path = "./config/config_paths.yml"  # 設定ファイルのパス
    abs_path = Path(root_path, config_path).absolute()  # 絶対パスを生成

    # カスタムタグを登録
    yaml.SafeLoader.add_constructor("!join", join_paths)

    with open(abs_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)  # YAMLファイルを読み込む

    # パス設定の選択
    if use_remote_paths:
        path_config_key = "remote_paths"
    else:
        path_config_key = "local_paths"

    # プレースホルダーを実際の値で置換
    config[path_config_key] = {
        k: v.format(ACCESS_INFORMATION=ACCESS_INFORMATION) if isinstance(v, str) else v
        for k, v in config[path_config_key].items()
    }

    # 環境変数を展開
    for key, value in config[path_config_key].items():
        if isinstance(value, str):
            config[path_config_key][key] = os.path.expandvars(value)

    return config  # 設定を返す


def detect_google_drive_language():
    """Google Driveの言語を検出する関数"""
    possible_drive_letters = [
        "G:",
        "H:",
        "I:",
        "J:",
        "K:",
    ]  # 可能性のあるドライブレター
    jp_folder = "共有ドライブ"  # 日本語フォルダ名
    en_folder = "Shared drives"  # 英語フォルダ名

    for drive in possible_drive_letters:
        jp_path = os.path.join(drive, "\\", jp_folder)
        en_path = os.path.join(drive, "\\", en_folder)

        if os.path.exists(jp_path):
            return "JP", drive  # 日本語が見つかった場合
        elif os.path.exists(en_path):
            return "EN", drive  # 英語が見つかった場合

    raise ValueError(
        "Google Drive shared folder not found. Please check your Google Drive installation."
    )


def detect_os_and_language():
    """OSと言語を検出する関数"""
    os_name = platform.system().lower()  # OSの名前を取得
    if os_name == "darwin":
        return "mac", None  # Macの場合
    elif os_name == "windows":
        lang, drive = (
            detect_google_drive_language()
        )  # Windowsの場合、Google Driveの言語も検出
        return "win", lang, drive
    else:
        raise ValueError(f"Unsupported OS: {os_name}")


def get_data_path(path_key: str, use_remote_paths: bool = False) -> str:
    """データパスを取得する関数（ローカル・リモート対応版）

    Args:
        path_key (str): config.ymlのlocal_pathsまたはremote_pathsのキー
        use_remote_paths (bool): True の場合 remote_paths を使用、False の場合 local_paths を使用

    Returns:
        str: 絶対パス
    """
    # 設定ファイルを読み込む
    config = load_config(use_remote_paths=use_remote_paths)

    if use_remote_paths:
        # リモートパス（Google Drive）の場合
        return _build_remote_path(config, path_key)
    else:
        # ローカルパスの場合
        return _build_local_path(config, path_key)


def _build_local_path(config: dict, path_key: str) -> str:
    """ローカルパスを構築する関数"""
    # プロジェクトルートを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # local_pathsからパスを取得
    local_paths = config.get("local_paths", {})
    local_path = local_paths.get(path_key, f"data/{path_key}")

    # プロジェクトルートからの相対パスを構築
    full_path = os.path.join(project_root, local_path)

    # パスの正規化と絶対パスへの変換
    abs_path = os.path.abspath(os.path.normpath(full_path))
    return str(abs_path)


def _build_remote_path(config: dict, path_key: str) -> str:
    """リモートパス（Google Drive）を構築する関数"""
    try:
        # OSと言語を検出
        os_name, lang, drive = detect_os_and_language()

        remote_paths = config.get("remote_paths", {})

        if os_name == "win":
            # Windows の場合
            base_path = remote_paths.get("win_base_path", "G:/")
            shared_folder = remote_paths.get(lang, "Shared drives")

            # ベースパス構成要素を取得
            base_components = remote_paths.get("base_path_components", [])

            # パスを構築
            path_parts = [base_path, shared_folder] + base_components

            # データフォルダのパスを追加
            data_path = remote_paths.get(path_key, f"data/{path_key}")
            if data_path.startswith("data/"):
                data_path = data_path[5:]  # "data/" を削除

            path_parts.append(data_path)

            # パスを結合
            full_path = os.path.join(*path_parts)

        elif os_name == "mac":
            # Mac の場合
            base_path = remote_paths.get(
                "mac_base_path", "./Library/CloudStorage/GoogleDrive-"
            )
            shared_folder = remote_paths.get(
                "EN", "Shared drives"
            )  # Mac は英語版を想定

            # ベースパス構成要素を取得
            base_components = remote_paths.get("base_path_components", [])

            # パスを構築
            path_parts = [base_path, shared_folder] + base_components

            # データフォルダのパスを追加
            data_path = remote_paths.get(path_key, f"data/{path_key}")
            if data_path.startswith("data/"):
                data_path = data_path[5:]  # "data/" を削除

            path_parts.append(data_path)

            # パスを結合
            full_path = os.path.join(*path_parts)

        else:
            raise ValueError(f"Unsupported OS: {os_name}")

        # パスの正規化と絶対パスへの変換
        abs_path = os.path.abspath(os.path.normpath(full_path))
        return str(abs_path)

    except Exception as e:
        print(f"[Warning] Remote path construction failed: {e}")
        print("[Warning] Falling back to local path")
        return _build_local_path(config, path_key)


def get_weather_forecast_path(store_name: str, start_date: str, end_date: str) -> str:
    """Get the logical path for weather forecast data.

    This function generates a logical path for weather forecast CSV files that can be
    used with both GCS and local storage backends.

    Args:
        store_name: Store name (e.g., "Clea")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Logical path string (e.g., "05_WeatherData/02_WeatherForecast/Clea/weather_forecast_20250929_20251004.csv")
    """
    import os

    from config.config_gcp import GCPEnv

    # Determine which config to use based on storage backend
    backend = os.getenv("STORAGE_BACKEND", "local").lower()

    if backend == "gcs":
        # Use GCP config for logical path
        base_folder = GCPEnv.WEATHER_FORECAST_FOLDER.rstrip("/")
    else:
        # Use local config for logical path (relative to data root)
        config = load_config(use_remote_paths=False)
        weather_forecast_path = config.get("local_paths", {}).get(
            "weather_forecast_path", "data/05_WeatherData/02_WeatherForecast"
        )
        # Remove "data/" prefix if present for logical path
        if weather_forecast_path.startswith("data/"):
            base_folder = weather_forecast_path[5:]  # Remove "data/" prefix
        else:
            base_folder = weather_forecast_path

    # Clean dates (remove dashes)
    start_clean = start_date.replace("-", "")
    end_clean = end_date.replace("-", "")

    # Construct path
    filename = f"weather_forecast_{start_clean}_{end_clean}.csv"
    logical_path = f"{base_folder}/{store_name}/{filename}"

    return logical_path
