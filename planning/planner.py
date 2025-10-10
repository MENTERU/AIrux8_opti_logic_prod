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

            rec = {
                "Date Time": t.strftime("%Y/%m/%d %H:%M"),
                "outside_temp": outside_temp if outside_temp is not None else np.nan,
            }
            for z, zs in schedule.items():
                s = zs.get(t, {})
                # Check if mode is OFF to determine OnOFF status
                mode = s.get("mode", FALLBACK_MODE_CODE) if s else FALLBACK_MODE_CODE
                is_off = mode == "OFF" or mode == 0 or str(mode).upper() == "OFF"
                rec[f"{z}_OnOFF"] = "OFF" if is_off else "ON"
                rec[f"{z}_Mode"] = (
                    self._mode_text(s.get("mode", FALLBACK_MODE_CODE))
                    if s
                    else FALLBACK_MODE_LABEL
                )
                rec[f"{z}_SetTemp"] = s.get("set_temp", 25) if s else 25
                rec[f"{z}_FanSpeed"] = (
                    self._fan_text(s.get("fan", FALLBACK_FAN_CODE))
                    if s
                    else FALLBACK_FAN_LABEL
                )
                # 予測電力・予測室温（可視化用）
                rec[f"{z}_PredPower"] = (
                    round(float(s.get("pred_power", 0.0)), 2) if s else 0.0
                )
                rec[f"{z}_PredTemp"] = (
                    round(float(s.get("pred_temp", np.nan)), 2) if s else np.nan
                )
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

            rec = {
                "Date Time": t.strftime("%Y/%m/%d %H:%M"),
                "outside_temp": outside_temp if outside_temp is not None else np.nan,
            }
            for z, units in zone_to_units.items():
                s = schedule.get(z, {}).get(t, {})
                for u in units:
                    # Check if mode is OFF to determine OnOFF status
                    mode = (
                        s.get("mode", FALLBACK_MODE_CODE) if s else FALLBACK_MODE_CODE
                    )
                    is_off = mode == "OFF" or mode == 0 or str(mode).upper() == "OFF"
                    rec[f"{u}_OnOFF"] = "OFF" if is_off else "ON"
                    rec[f"{u}_Mode"] = (
                        self._mode_text(s.get("mode", FALLBACK_MODE_CODE))
                        if s
                        else FALLBACK_MODE_LABEL
                    )
                    rec[f"{u}_SetTemp"] = s.get("set_temp", 25) if s else 25
                    rec[f"{u}_FanSpeed"] = (
                        self._fan_text(s.get("fan", FALLBACK_FAN_CODE))
                        if s
                        else FALLBACK_FAN_LABEL
                    )
            unit_rows.append(rec)
        unit_df = pd.DataFrame(unit_rows)
        unit_path = os.path.join(out_dir, f"unit_schedule_{date_str}.csv")
        unit_df.to_csv(unit_path, index=False, encoding="utf-8-sig")

        print(f"[Planner] control schedule: {ctrl_path}")
        print(f"[Planner] unit schedule: {unit_path}")
