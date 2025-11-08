from datetime import datetime

import pandas as pd
from prettytable import PrettyTable


class DataManager:
    @classmethod
    def show_status(self, df: pd.DataFrame):
        table = PrettyTable(
            ["Variable", "Type", "Missing Values", "Duplicates", "Outliers"]
        )

        for col in df.columns:
            # データ型の判定
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = "Numerical"
            else:
                col_type = "Categorical"

            # 欠損値
            missing_values = df[col].isnull().sum()

            # 重複数
            duplicates = df.duplicated(subset=[col]).sum()

            # 外れ値（数値型のみ）
            if col_type == "Numerical":
                mean = df[col].mean()
                std = df[col].std()
                outliers = ((df[col] - mean).abs() > 3 * std).sum()
            else:
                outliers = "N/A"

            table.add_row([col, col_type, missing_values, duplicates, outliers])
        print(table)

    @classmethod
    def extract_term_data(
        cls, target_col: str, df: pd.DataFrame, start_time: datetime, end_time: datetime
    ):
        df[target_col] = pd.to_datetime(df[target_col])
        filtered_df = df[(df[target_col] >= start_time) & (df[target_col] <= end_time)]
        return filtered_df

    @classmethod
    def check_date_sequence(cls, df: pd.DataFrame, date_col: str = "date"):
        dates = pd.to_datetime(df[date_col]).sort_values().reset_index(drop=True)
        expected = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
        missing = expected.difference(dates)
        return {
            "is_continuous": len(missing) == 0,
            "missing_dates": missing.strftime("%Y-%m-%d").tolist(),
        }
