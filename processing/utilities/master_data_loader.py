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

        if not os.path.exists(excel_path):
            print(f"[MasterDataLoader] Excelファイルが見つかりません: {excel_path}")
            return None

        try:
            print(f"[MasterDataLoader] Excelファイルを読み込み中: {excel_path}")

            # Read the MASTER sheet
            master_df = pd.read_excel(excel_path, sheet_name="MASTER")
            print(
                f"[MasterDataLoader] Excel MASTER sheet読み込み成功: shape={master_df.shape}"
            )
            print(f"[MasterDataLoader] Excel columns: {list(master_df.columns)}")

            # Extract coordinates from Excel file
            coordinates = None
            if "座標" in master_df.columns:
                coordinates = str(
                    master_df.iloc[0]["座標"]
                )  # 現在座標はすべてのエリア共通
                print(f"[MasterDataLoader] Coordinates from Excel: {coordinates}")
            else:
                print(f"[MasterDataLoader] Coordinates column not found")
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

        if not os.path.exists(excel_path):
            print(f"[MasterDataLoader] Excelファイルが見つかりません: {excel_path}")
            return None

        try:
            # Read the MASTER sheet
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

    def get_master_data_for_aggregator(self) -> Optional[dict]:
        """Get complete master data structure for aggregator from Excel file"""
        from config.utils import get_data_path

        master_dir = get_data_path("master_data_path")
        excel_path = os.path.join(master_dir, f"MASTER_{self.store_name}.xlsx")

        if not os.path.exists(excel_path):
            print(f"[MasterDataLoader] Excelファイルが見つかりません: {excel_path}")
            return None

        try:
            print(
                f"[MasterDataLoader] Building master data for aggregator from: {excel_path}"
            )

            # Read all relevant sheets
            all_sheets = pd.read_excel(excel_path, sheet_name=None)
            print(f"[MasterDataLoader] Available sheets: {list(all_sheets.keys())}")

            # Build the master data structure
            master_data = {"store_name": self.store_name, "zones": {}}

            # Process MASTER sheet to get zone information
            if "MASTER" in all_sheets:
                master_df = all_sheets["MASTER"]
                print(
                    f"[MasterDataLoader] Processing MASTER sheet: shape={master_df.shape}"
                )

                # Get unique zones
                zones = master_df["制御区分"].dropna().unique()
                print(f"[MasterDataLoader] Found zones: {list(zones)}")

                for zone_name in zones:
                    if pd.isna(zone_name):
                        continue

                    # Initialize zone structure
                    master_data["zones"][zone_name] = {"outdoor_units": {}}
                    print(f"[MasterDataLoader] Processing zone: {zone_name}")

            # Process equipment mapping from MASTER sheet (電力予測区分 -> 環境予測区分)
            print(f"[MasterDataLoader] Processing equipment mapping from MASTER sheet")

            # Map equipment to zones using MASTER sheet data
            for _, row in master_df.iterrows():
                zone_name = row["制御区分"]
                outdoor_unit = row["電力予測区分"]  # 電力予測区分 = outdoor unit
                indoor_unit = row["環境予測区分"]  # 環境予測区分 = indoor unit

                if pd.isna(zone_name) or pd.isna(outdoor_unit) or pd.isna(indoor_unit):
                    continue

                # Clean up outdoor unit ID (remove trailing comma if present)
                outdoor_unit = str(outdoor_unit).rstrip(",").strip()

                if zone_name in master_data["zones"]:
                    if (
                        outdoor_unit
                        not in master_data["zones"][zone_name]["outdoor_units"]
                    ):
                        master_data["zones"][zone_name]["outdoor_units"][
                            outdoor_unit
                        ] = {
                            "load_share": 1.0,  # Default value as requested
                            "indoor_units": [],
                        }
                    master_data["zones"][zone_name]["outdoor_units"][outdoor_unit][
                        "indoor_units"
                    ].append(indoor_unit)
                    print(
                        f"[MasterDataLoader] Mapped {outdoor_unit} -> {indoor_unit} for zone {zone_name}"
                    )

            print(f"[MasterDataLoader] Master data for aggregator built successfully")
            print(f"[MasterDataLoader] Zones: {list(master_data['zones'].keys())}")

            # Print summary for each zone
            for zone_name, zone_data in master_data["zones"].items():
                outdoor_count = len(zone_data["outdoor_units"])
                indoor_count = sum(
                    len(ou["indoor_units"])
                    for ou in zone_data["outdoor_units"].values()
                )
                print(
                    f"[MasterDataLoader] Zone {zone_name}: {outdoor_count} outdoor units, {indoor_count} indoor units"
                )

            return master_data

        except Exception as e:
            print(f"[MasterDataLoader] Error building master data for aggregator: {e}")
            import traceback

            traceback.print_exc()
            return None
