"""
モデル学習用や予測用の特徴量やターゲットの設定ファイル
"""

# Base features from model_builder.py (high priority - must match exactly)
BASE_FEATURES = [
    "A/C Set Temperature",
    "Indoor Temp. Lag1",
    # "A/C ON/OFF",  # Commented out - using A/C Status instead
    # "A/C Mode",    # Commented out - using A/C Status instead
    "A/C Status",  # Combined feature: OFF=0, COOL=1, HEAT=2, FAN=3
    "A/C Fan Speed",
    "Outdoor Temp.",
    "Outdoor Humidity",
    "Solar Radiation",
    "DayOfWeek",
    "Hour",
    "Month",
    "IsWeekend",
    "IsHoliday",
]

# 特徴量
NEW_FEATURES = [
    "Wet Bulb Temp",
    "Temp Diff (Outdoor - Indoor Lag1)",
    "Temp Diff (Indoor Lag1 - Setpoint)",
    "HourOfWeek",
    "OnRunLength",
    "Outdoor Temp. lag1",
    "Outdoor Temp. lag24",
    "Outdoor Humidity lag1",
    "Outdoor Humidity lag24",
    "adjusted_power lag1",
    "adjusted_power lag24",
    "Outdoor Temp. rolling_mean3",
    "Outdoor Temp. rolling_mean24",
    "Outdoor Humidity rolling_mean3",
    "Outdoor Humidity rolling_mean24",
    "adjusted_power rolling_mean3",
    "adjusted_power rolling_mean24",
]


TEMP_FEATURE_COLS = BASE_FEATURES + NEW_FEATURES

POWER_FEATURE_COLS = TEMP_FEATURE_COLS + ["Indoor Temp."]

# ターゲット
TARGET_TEMP = "Indoor Temp."
TARGET_HUM = "室内湿度"
TARGET_POWER = "adjusted_power"
