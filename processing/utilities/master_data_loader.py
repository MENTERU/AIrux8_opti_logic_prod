import os
from typing import Optional

import pandas as pd


# =============================
# Excel-based Master Data Loader (for preprocessing coordinates only)
# =============================
class MasterDataLoader:
    """Excel-based master data loader specifically for preprocessing step to get coordinates"""

    def __init__(self, store_name: str):
        self.store_name = store_name

    def get_coordinates(self) -> Optional[str]:
        """Get coordinates from Excel file for weather API calls"""
        from config.utils import get_data_path

        master_dir = get_data_path("master_data_path")
        excel_path = os.path.join(master_dir, f"MASTER_{self.store_name}.xlsx")
        storage_backend = os.getenv("STORAGE_BACKEND", "local").lower()

        try:
            if storage_backend == "gcs":
                # Read from GCS
                print(
                    f"[MasterDataLoader] ExcelファイルをGCSから読み込み中: 01_MasterData/MASTER_{self.store_name}.xlsx"
                )
                from io import BytesIO

                from service.storage import get_storage_client

                client = get_storage_client()
                content = client.read_bytes(
                    f"01_MasterData/MASTER_{self.store_name}.xlsx"
                )
                facility_info_df = pd.read_excel(
                    BytesIO(content), sheet_name="施設情報"
                )
            else:
                print(f"[MasterDataLoader] Excelファイルを読み込み中: {excel_path}")
                # Read the 施設情報 sheet (local filesystem)
                facility_info_df = pd.read_excel(excel_path, sheet_name="施設情報")
            print(
                f"[MasterDataLoader] Excel 施設情報 sheet読み込み成功: shape={facility_info_df.shape}"
            )
            print(f"[MasterDataLoader] Excel columns: {list(facility_info_df.columns)}")

            # Extract coordinates from 施設情報 sheet
            coordinates = None
            if (
                "施設情報" in facility_info_df.columns
                and "値" in facility_info_df.columns
            ):
                # Find the row where 施設情報 contains "施設座標"
                coord_row = facility_info_df[facility_info_df["施設情報"] == "施設座標"]
                if not coord_row.empty:
                    coordinates = str(coord_row.iloc[0]["値"])
                    print(
                        f"[MasterDataLoader] Coordinates from 施設情報 sheet: {coordinates}"
                    )
                else:
                    print(
                        f"[MasterDataLoader] 施設座標 row not found in 施設情報 sheet"
                    )
                    return None
            else:
                print(
                    f"[MasterDataLoader] Required columns not found in 施設情報 sheet"
                )
                return None
            return coordinates

        except Exception as e:
            print(f"[MasterDataLoader] Excel読み込みエラー: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_store_info(self) -> Optional[dict]:
        """Get basic store information from Excel file"""
        from config.utils import get_data_path

        master_dir = get_data_path("master_data_path")
        excel_path = os.path.join(master_dir, f"MASTER_{self.store_name}.xlsx")
        storage_backend = os.getenv("STORAGE_BACKEND", "local").lower()

        try:
            # Read the MASTER sheet
            if storage_backend == "gcs":
                print(
                    f"[MasterDataLoader] MASTERシートをGCSから読み込み: 01_MasterData/MASTER_{self.store_name}.xlsx"
                )
                from io import BytesIO

                from service.storage import get_storage_client

                client = get_storage_client()
                content = client.read_bytes(
                    f"01_MasterData/MASTER_{self.store_name}.xlsx"
                )
                master_df = pd.read_excel(BytesIO(content), sheet_name="MASTER")
            else:
                master_df = pd.read_excel(excel_path, sheet_name="MASTER")

            # Get coordinates
            coordinates = self.get_coordinates()

            # Build basic store info
            store_info = {
                "name": self.store_name,
                "area": "Tokyo",  # Could be extracted from Excel if available
                "coordinates": coordinates,
            }

            print(f"[MasterDataLoader] Store info: {store_info}")
            return store_info

        except Exception as e:
            print(f"[MasterDataLoader] Store info読み込みエラー: {e}")
            return None

    def get_complete_master_data(self) -> Optional[dict]:
        """Get complete master data structure from consolidated 制御マスタ sheet for current month only"""
        from datetime import datetime

        from config.utils import get_data_path

        master_dir = get_data_path("master_data_path")
        excel_path = os.path.join(master_dir, f"MASTER_{self.store_name}.xlsx")
        storage_backend = os.getenv("STORAGE_BACKEND", "local").lower()

        try:
            if storage_backend == "gcs":
                print(
                    f"[MasterDataLoader] Building master data from 制御マスタ (GCS): 01_MasterData/MASTER_{self.store_name}.xlsx"
                )
                from io import BytesIO

                from service.storage import get_storage_client

                client = get_storage_client()
                content = client.read_bytes(
                    f"01_MasterData/MASTER_{self.store_name}.xlsx"
                )
                all_sheets = pd.read_excel(BytesIO(content), sheet_name=None)
            else:
                print(
                    f"[MasterDataLoader] Building master data from 制御マスタ sheet: {excel_path}"
                )
                # Read all sheets (local filesystem)
                all_sheets = pd.read_excel(excel_path, sheet_name=None)
            print(f"[MasterDataLoader] Available sheets: {list(all_sheets.keys())}")

            # Get current month
            current_month = datetime.now().month
            current_month_japanese = f"{current_month}月"
            print(
                f"[MasterDataLoader] Current month: {current_month} ({current_month_japanese})"
            )

            # Initialize master data structure
            master_data = {
                "store_info": {
                    "name": self.store_name,
                    "area": "Tokyo",  # Default value
                    "coordinates": self.get_coordinates() or "35.681236%2C139.767124",
                },
                "zones": {},
            }

            # Process MASTER sheet for equipment mapping
            if "MASTER" not in all_sheets:
                print(f"[MasterDataLoader] ERROR: MASTER sheet not found")
                return None

            master_df = all_sheets["MASTER"]
            print(
                f"[MasterDataLoader] Processing MASTER sheet for equipment mapping: shape={master_df.shape}"
            )

            # Get unique zones from MASTER sheet
            zones = master_df["制御区分"].dropna().unique()
            print(f"[MasterDataLoader] Found zones in MASTER: {list(zones)}")

            # Initialize zone structures
            for zone_name in zones:
                if pd.isna(zone_name):
                    continue
                master_data["zones"][zone_name] = {"outdoor_units": {}}
                print(f"[MasterDataLoader] Initialized zone: {zone_name}")

            # Process equipment mapping from MASTER sheet
            print(f"[MasterDataLoader] Processing equipment mapping from MASTER sheet")
            for _, row in master_df.iterrows():
                zone_name = row["制御区分"]
                outdoor_unit = row["電力予測区分"]
                indoor_unit = row["環境予測区分"]

                if pd.isna(zone_name) or pd.isna(outdoor_unit) or pd.isna(indoor_unit):
                    continue

                # Clean up outdoor unit ID
                outdoor_unit = str(outdoor_unit).rstrip(",").strip()

                if zone_name in master_data["zones"]:
                    if (
                        outdoor_unit
                        not in master_data["zones"][zone_name]["outdoor_units"]
                    ):
                        master_data["zones"][zone_name]["outdoor_units"][
                            outdoor_unit
                        ] = {"load_share": 1.0, "indoor_units": []}
                    master_data["zones"][zone_name]["outdoor_units"][outdoor_unit][
                        "indoor_units"
                    ].append(indoor_unit)

            # Process 制御マスタ sheet for current month settings
            if "制御マスタ" not in all_sheets:
                print(f"[MasterDataLoader] ERROR: 制御マスタ sheet not found")
                return None

            control_master_df = all_sheets["制御マスタ"]
            print(
                f"[MasterDataLoader] Processing 制御マスタ sheet: shape={control_master_df.shape}"
            )

            # Filter for current month only
            current_month_data = control_master_df[
                control_master_df["月"] == current_month_japanese
            ]
            print(
                f"[MasterDataLoader] Current month data rows: {len(current_month_data)}"
            )

            if len(current_month_data) == 0:
                print(
                    f"[MasterDataLoader] WARNING: No data found for current month {current_month_japanese}"
                )
                # Use default settings
                for zone_name in master_data["zones"]:
                    master_data["zones"][zone_name].update(
                        {
                            "start_time": "07:00",
                            "end_time": "20:00",
                            "comfort_min": 22.0,
                            "comfort_max": 25.0,
                            "setpoint_min": 22.0,
                            "setpoint_max": 28.0,
                            "fan_candidates": ["Low", "Medium", "High"],
                            "mode_candidates": ["COOL", "HEAT", "FAN"],
                            "target_room_temp": 25.0,
                        }
                    )
            else:
                # Process current month settings for each zone
                print(
                    f"[MasterDataLoader] Processing current month settings for each zone"
                )
                for _, row in current_month_data.iterrows():
                    zone_name = row["制御区分"]

                    if zone_name in master_data["zones"]:
                        # Extract current month settings
                        master_data["zones"][zone_name].update(
                            {
                                "start_time": (
                                    str(row["始業時間"]).split()[0]
                                    if pd.notna(row["始業時間"])
                                    else "07:00"
                                ),
                                "end_time": (
                                    str(row["就業時間"]).split()[0]
                                    if pd.notna(row["就業時間"])
                                    else "20:00"
                                ),
                                "comfort_min": (
                                    float(row["目標室内温度下限"])
                                    if pd.notna(row["目標室内温度下限"])
                                    else 22.0
                                ),
                                "comfort_max": (
                                    float(row["目標室内温度上限"])
                                    if pd.notna(row["目標室内温度上限"])
                                    else 25.0
                                ),
                                "setpoint_min": (
                                    float(row["設定温度下限"])
                                    if pd.notna(row["設定温度下限"])
                                    else 22.0
                                ),
                                "setpoint_max": (
                                    float(row["設定温度上限"])
                                    if pd.notna(row["設定温度上限"])
                                    else 28.0
                                ),
                                "fan_candidates": (
                                    str(row["風量候補"]).split(",")
                                    if pd.notna(row["風量候補"])
                                    else ["Low"]
                                ),
                                "mode_candidates": [
                                    "COOL",
                                    "HEAT",
                                    "FAN",
                                ],  # Default values
                                "target_room_temp": (
                                    (
                                        float(row["目標室内温度下限"])
                                        + float(row["目標室内温度上限"])
                                    )
                                    / 2
                                    if pd.notna(row["目標室内温度下限"])
                                    and pd.notna(row["目標室内温度上限"])
                                    else 25.0
                                ),
                            }
                        )
                        print(
                            f"[MasterDataLoader] Added current month settings for {zone_name}"
                        )

            print(
                f"[MasterDataLoader] Master data built successfully for current month ({current_month_japanese})"
            )
            print(f"[MasterDataLoader] Zones: {list(master_data['zones'].keys())}")

            # Print summary
            for zone_name, zone_data in master_data["zones"].items():
                outdoor_count = len(zone_data.get("outdoor_units", {}))
                indoor_count = sum(
                    len(ou.get("indoor_units", []))
                    for ou in zone_data.get("outdoor_units", {}).values()
                )
                print(
                    f"[MasterDataLoader] Zone {zone_name}: {outdoor_count} outdoor units, {indoor_count} indoor units, current month settings"
                )

            return master_data

        except Exception as e:
            print(f"[MasterDataLoader] Error building master data: {e}")
            import traceback

            traceback.print_exc()
            return None


def get_comfort_range(master_data: dict, zone: str, month: int) -> tuple[float, float]:
    """
    Extract comfort temperature range (目標室内温度下限, 目標室内温度上限) for given zone and month

    Args:
        master_data: Master data dictionary from get_complete_master_data()
        zone: Zone name (e.g., "Area 1", "Area 2", etc.)
        month: Month number (1-12)

    Returns:
        tuple: (min_temp, max_temp) comfort range
    """
    try:
        # Convert month to Japanese format
        month_japanese = f"{month}月"

        # Load the 制御マスタ sheet directly to get month-specific data
        import os

        from config.utils import get_data_path

        master_dir = get_data_path("master_data_path")
        excel_path = os.path.join(
            master_dir, f"MASTER_{master_data['store_info']['name']}.xlsx"
        )

        if not os.path.exists(excel_path):
            print(f"[get_comfort_range] Excel file not found: {excel_path}")
            return (22.0, 25.0)  # Default comfort range

        # Read 制御マスタ sheet
        control_master_df = pd.read_excel(excel_path, sheet_name="制御マスタ")

        # Filter for specific zone and month
        zone_month_data = control_master_df[
            (control_master_df["制御区分"] == zone)
            & (control_master_df["月"] == month_japanese)
        ]

        if len(zone_month_data) == 0:
            print(
                f"[get_comfort_range] No data found for zone {zone}, month {month_japanese}"
            )
            return (22.0, 25.0)  # Default comfort range

        row = zone_month_data.iloc[0]
        min_temp = (
            float(row["目標室内温度下限"])
            if pd.notna(row["目標室内温度下限"])
            else 22.0
        )
        max_temp = (
            float(row["目標室内温度上限"])
            if pd.notna(row["目標室内温度上限"])
            else 25.0
        )

        return (min_temp, max_temp)

    except Exception as e:
        print(f"[get_comfort_range] Error getting comfort range: {e}")
        return (22.0, 25.0)  # Default comfort range


def get_zone_operating_hours(master_data: dict, zone: str) -> tuple[int, int]:
    """
    Extract operating hours (start_time, end_time) for a given zone from master data.

    Args:
        master_data: Master data dictionary from get_complete_master_data()
        zone: Zone name (e.g., "Area 1", "Area 2", etc.)

    Returns:
        tuple: (start_hour, end_hour) in 24-hour format (0-23)
    """
    try:
        if zone not in master_data.get("zones", {}):
            print(f"[get_zone_operating_hours] Zone {zone} not found in master data")
            return (7, 20)  # Default operating hours 7:00-20:00

        zone_data = master_data["zones"][zone]
        start_time = zone_data.get("start_time", "07:00")
        end_time = zone_data.get("end_time", "20:00")

        # Parse time strings (format: "HH:MM")
        start_hour = int(start_time.split(":")[0])
        end_hour = int(end_time.split(":")[0])

        print(
            f"[get_zone_operating_hours] Zone {zone}: operating hours {start_hour}:00-{end_hour}:00"
        )
        return (start_hour, end_hour)

    except Exception as e:
        print(
            f"[get_zone_operating_hours] Error getting operating hours for zone {zone}: {e}"
        )
        return (7, 20)  # Default operating hours


def master_data_loader_runner(store_name: str) -> Optional[dict]:
    """
    Extract complete Excel master data from 制御マスタ sheet for current month

    Args:
        store_name: ストア名

    Returns:
        dict: マスタデータ
    """
    # Extract complete Excel master data from 制御マスタ sheet for current month
    complete_excel_master_data = None
    try:

        excel_loader = MasterDataLoader(store_name)
        excel_master_data = excel_loader.get_complete_master_data()

        if excel_master_data:
            print(f"[RunOptimization] Using Excel master data for current month")
            print(
                f"[RunOptimization] Excel master data zones: {list(excel_master_data.get('zones', {}).keys())}"
            )
            print(
                f"[RunOptimization] Store info: {excel_master_data.get('store_info', {})}"
            )
            return excel_master_data
        else:
            print(f"[RunOptimization] ERROR: No Excel master data found")
            return None
    except Exception as e:
        print(f"[RunOptimization] ERROR getting Excel master data: {e}")
        return None
