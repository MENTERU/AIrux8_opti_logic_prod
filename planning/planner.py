import os
from typing import Dict, List

import numpy as np
import pandas as pd

from processing.utilities.category_mapping_loader import (
    get_category_mapping,
    get_inverse_category_mapping,
)

MODE_LABEL_TO_CODE = get_category_mapping("A/C Mode")
MODE_CODE_TO_LABEL = get_inverse_category_mapping("A/C Mode")
FALLBACK_MODE_LABEL = (
    "FAN" if "FAN" in MODE_LABEL_TO_CODE else next(iter(MODE_LABEL_TO_CODE.keys()))
)
FALLBACK_MODE_CODE = MODE_LABEL_TO_CODE[FALLBACK_MODE_LABEL]

FAN_LABEL_TO_CODE = get_category_mapping("A/C Fan Speed")
FAN_CODE_TO_LABEL = get_inverse_category_mapping("A/C Fan Speed")
FALLBACK_FAN_CODE = (
    FAN_LABEL_TO_CODE.get("Low")
    if "Low" in FAN_LABEL_TO_CODE
    else next(iter(FAN_LABEL_TO_CODE.values()))
)
FALLBACK_FAN_LABEL = (
    "Low" if "Low" in FAN_LABEL_TO_CODE else next(iter(FAN_LABEL_TO_CODE.keys()))
)


# =============================
# STEP4: 出力（制御区分別 & 室内機別）
# =============================
class Planner:
    def __init__(self, store_name: str, master: dict):
        self.store_name = store_name
        self.master = master

    def _adjust_timestamp_for_business_hours(
        self, timestamp: pd.Timestamp, schedule: dict
    ) -> pd.Timestamp:
        """
        Adjust timestamp to show actual business start/end times instead of hourly boundaries.

        Example: If business hours are 7:30-20:00:
        - Hour 7:00-8:00 should show 7:30 (actual start time)
        - Hour 20:00-21:00 should show 20:00 (actual end time)
        """
        display_time = timestamp  # Default to original timestamp

        # Check each zone to see if we need to adjust the timestamp
        for zone_name, zone_schedule in schedule.items():
            # Get the schedule data for this timestamp (mode, temperature, etc.)
            schedule_data = zone_schedule.get(timestamp, {})

            # Get the zone configuration from master data (business hours, etc.)
            zone_data = self.master.get("zones", {}).get(zone_name, {})

            if zone_data:
                # Extract business start and end times from master data
                start_time_str = str(
                    zone_data.get("start_time", "07:00")
                )  # e.g., "07:30"
                end_time_str = str(zone_data.get("end_time", "20:00"))  # e.g., "20:00"

                # Parse start time into hour and minute components
                start_hour = int(start_time_str.split(":")[0])  # e.g., 7
                start_minute = (
                    int(start_time_str.split(":")[1]) if ":" in start_time_str else 0
                )  # e.g., 30

                # Parse end time into hour and minute components
                end_hour = int(end_time_str.split(":")[0])  # e.g., 20
                end_minute = (
                    int(end_time_str.split(":")[1]) if ":" in end_time_str else 0
                )  # e.g., 0

                # Check if this zone is currently ON (not OFF mode)
                is_zone_on = (
                    schedule_data
                    and schedule_data.get("mode") is not None
                    and schedule_data.get("mode") != "OFF"
                )

                # ADJUSTMENT 1: Start time adjustment
                # If zone turns ON at the beginning of an hour but business starts at fractional time
                # Example: Hour 7:00-8:00, business starts at 7:30, zone is ON → show 7:30
                if (
                    is_zone_on
                    and timestamp.hour == start_hour
                    and timestamp.minute == 0
                    and start_minute > 0
                ):
                    display_time = timestamp.replace(
                        hour=start_hour, minute=start_minute
                    )
                    break  # Found adjustment, no need to check other zones

                # ADJUSTMENT 2: End time adjustment
                # If zone turns OFF at the beginning of an hour but business ends at fractional time
                # Example: Hour 20:00-21:00, business ends at 20:00, zone is OFF, previous hour was ON → show 20:00
                if (
                    not is_zone_on
                    and timestamp.hour == end_hour
                    and timestamp.minute == 0
                ):
                    # Check if the previous hour was ON (meaning this is the end of business hours)
                    prev_hour_timestamp = timestamp - pd.Timedelta(hours=1)
                    prev_schedule_data = zone_schedule.get(prev_hour_timestamp, {})
                    prev_hour_was_on = (
                        prev_schedule_data
                        and prev_schedule_data.get("mode") is not None
                        and prev_schedule_data.get("mode") != "OFF"
                    )

                    if prev_hour_was_on:
                        # Adjust to show the actual business end time
                        display_time = timestamp.replace(
                            hour=end_hour, minute=end_minute
                        )
                        break  # Found adjustment, no need to check other zones

        return display_time

    @staticmethod
    def _mode_text(n: int) -> str:
        return MODE_CODE_TO_LABEL.get(n, FALLBACK_MODE_LABEL)

    @staticmethod
    def _fan_text(n: int) -> str:
        return FAN_CODE_TO_LABEL.get(n, FALLBACK_FAN_LABEL).upper()

    def export(self, schedule: Dict[str, Dict[pd.Timestamp, dict]], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # 日付
        any_ts = None
        for z in schedule:
            if schedule[z]:
                any_ts = list(schedule[z].keys())[0]
                break
        if any_ts is None:
            from datetime import datetime

            any_ts = pd.Timestamp(pd.Timestamp.now().date())
        date_str = pd.Timestamp(any_ts).strftime("%Y%m%d")

        # 制御区分別
        rows = []
        for t in sorted({ts for z in schedule for ts in schedule[z].keys()}):
            # Get outside_temp from any zone's schedule for this timestamp
            outside_temp = None
            for z, zs in schedule.items():
                s = zs.get(t, {})
                if s and "outside_temp" in s:
                    outside_temp = s["outside_temp"]
                    break

            # Adjust timestamp to show actual business start/end times
            display_time = self._adjust_timestamp_for_business_hours(t, schedule)

            rec = {
                "Date Time": display_time.strftime("%Y/%m/%d %H:%M"),
                "outside_temp": outside_temp if outside_temp is not None else np.nan,
            }
            for z, zs in schedule.items():
                s = zs.get(t, {})
                # Check ON/OFF status using count-based value
                onoff_count = s.get("onoff_count", 0) if s else 0
                total_units = self._get_zone_unit_count(
                    z
                )  # Get total units for this zone
                is_off = onoff_count == 0
                is_all_on = onoff_count == total_units

                if is_off:
                    rec[f"{z}_OnOFF"] = "OFF"
                elif is_all_on:
                    rec[f"{z}_OnOFF"] = "ON"
                else:
                    # Partial state - show count
                    rec[f"{z}_OnOFF"] = f"{onoff_count}/{total_units}"

                # Show mode values based on ON/OFF status
                if is_off:
                    rec[f"{z}_Mode"] = (
                        "OFF"  # When room is OFF, AC Status should be OFF
                    )
                elif s and s.get("mode") is not None:
                    rec[f"{z}_Mode"] = self._mode_text(
                        s.get("mode", FALLBACK_MODE_CODE)
                    )
                else:
                    rec[f"{z}_Mode"] = self._mode_text(
                        FALLBACK_MODE_CODE
                    )  # Use fallback mode for other cases

                if s and s.get("set_temp") is not None:
                    rec[f"{z}_SetTemp"] = s.get("set_temp", 25)
                else:
                    rec[f"{z}_SetTemp"] = ""  # Empty for non-business hours

                if s and s.get("fan") is not None:
                    rec[f"{z}_FanSpeed"] = self._fan_text(
                        s.get("fan", FALLBACK_FAN_CODE)
                    )
                else:
                    rec[f"{z}_FanSpeed"] = ""  # Empty for non-business hours
                # 予測電力・予測室温（可視化用）
                rec[f"{z}_PredPower"] = (
                    round(float(s.get("pred_power", 0.0)), 2) if s else 0.0
                )
                if s and s.get("pred_temp") is not None:
                    rec[f"{z}_PredTemp"] = round(float(s.get("pred_temp")), 2)
                else:
                    rec[f"{z}_PredTemp"] = 0.0  # 0.0 if no prediction available
            rows.append(rec)
        ctrl_df = pd.DataFrame(rows)
        ctrl_path = os.path.join(out_dir, f"control_type_schedule_{date_str}.csv")
        ctrl_df.to_csv(ctrl_path, index=False, encoding="utf-8-sig")

        # 室内機別
        unit_rows = []
        zones = self.master.get("zones", {})
        zone_to_units: Dict[str, List[str]] = {}
        for z, zinfo in zones.items():
            units = []
            for _, ou in zinfo.get("outdoor_units", {}).items():
                units.extend(ou.get("indoor_units", []))
            zone_to_units[z] = list(dict.fromkeys(units))
        for t in sorted({ts for z in schedule for ts in schedule[z].keys()}):
            # Get outside_temp from any zone's schedule for this timestamp
            outside_temp = None
            for z, zs in schedule.items():
                s = zs.get(t, {})
                if s and "outside_temp" in s:
                    outside_temp = s["outside_temp"]
                    break

            # Adjust timestamp to show actual business start/end times
            display_time = self._adjust_timestamp_for_business_hours(t, schedule)

            rec = {
                "Date Time": display_time.strftime("%Y/%m/%d %H:%M"),
                "outside_temp": outside_temp if outside_temp is not None else np.nan,
            }
            for z, units in zone_to_units.items():
                s = schedule.get(z, {}).get(t, {})
                for u in units:
                    # Check ON/OFF status using count-based value
                    onoff_count = s.get("onoff_count", 0) if s else 0
                    total_units = len(units)  # Total units in this zone
                    is_off = onoff_count == 0
                    is_all_on = onoff_count == total_units

                    if is_off:
                        rec[f"{u}_OnOFF"] = "OFF"
                    elif is_all_on:
                        rec[f"{u}_OnOFF"] = "ON"
                    else:
                        # Partial state - show count
                        rec[f"{u}_OnOFF"] = f"{onoff_count}/{total_units}"

                    # Show mode values based on ON/OFF status
                    if is_off:
                        rec[f"{u}_Mode"] = (
                            "OFF"  # When room is OFF, AC Status should be OFF
                        )
                    elif s and s.get("mode") is not None:
                        rec[f"{u}_Mode"] = self._mode_text(
                            s.get("mode", FALLBACK_MODE_CODE)
                        )
                    else:
                        rec[f"{u}_Mode"] = self._mode_text(
                            FALLBACK_MODE_CODE
                        )  # Use fallback mode for other cases

                    if s and s.get("set_temp") is not None:
                        rec[f"{u}_SetTemp"] = s.get("set_temp", 25)
                    else:
                        rec[f"{u}_SetTemp"] = ""  # Empty for non-business hours

                    if s and s.get("fan") is not None:
                        rec[f"{u}_FanSpeed"] = self._fan_text(
                            s.get("fan", FALLBACK_FAN_CODE)
                        )
                    else:
                        rec[f"{u}_FanSpeed"] = ""  # Empty for non-business hours
            unit_rows.append(rec)
        unit_df = pd.DataFrame(unit_rows)
        unit_path = os.path.join(out_dir, f"unit_schedule_{date_str}.csv")
        unit_df.to_csv(unit_path, index=False, encoding="utf-8-sig")

        print(f"[Planner] control schedule: {ctrl_path}")
        print(f"[Planner] unit schedule: {unit_path}")

    def _get_zone_unit_count(self, zone_name: str) -> int:
        """
        Get the total number of indoor units for a zone from master data.

        Args:
            zone_name: Name of the zone (e.g., "Area 1", "Area 2")

        Returns:
            Total number of indoor units in the zone
        """
        if zone_name in self.master:
            zone_data = self.master[zone_name]
            return len(zone_data.get("indoor_units", []))
        return 1  # Default fallback
