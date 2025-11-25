import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta


# =============================
# IF: 天候データ取得
# =============================
class VisualCrossingWeatherAPIDataFetcher:
    """Visual Crossing Weather API で時別の天気（気温・湿度）を取得（バッチ処理対応）"""

    def __init__(
        self,
        coordinates: str,
        start_date: str,
        end_date: str,
        unit: str,
        api_key: str,
        temperature_col_name: str = "Outdoor Temp.",
        humidity_col_name: str = "Outdoor Humidity",
        solar_col_name: str = "Solar Radiation",
        batch_size_months: int = 1,
        delay_between_requests: float = 1.0,
    ):
        self.coordinates = coordinates
        self.start_date = start_date
        self.end_date = end_date
        self.unit = unit
        self.api_key = api_key
        self.temperature_col_name = temperature_col_name
        self.humidity_col_name = humidity_col_name
        self.solar_col_name = solar_col_name
        self.batch_size_months = batch_size_months
        self.delay_between_requests = delay_between_requests

    def _generate_date_batches(self):
        """
        Generate date batches for API requests to handle long date ranges

        Returns:
            list: List of tuples containing (start_date, end_date) for each batch
        """
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

        batches = []
        current_start = start_dt

        while current_start <= end_dt:
            # Calculate the end date for this batch
            current_end = (
                current_start
                + relativedelta(months=self.batch_size_months)
                - timedelta(days=1)
            )

            # Don't exceed the overall end date
            if current_end > end_dt:
                current_end = end_dt

            batches.append(
                (current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d"))
            )

            # Move to next batch
            current_start = current_end + timedelta(days=1)

        return batches

    def _fetch_single_batch(self, batch_start_date: str, batch_end_date: str):
        """
        Fetch weather data for a single date batch

        Args:
            batch_start_date (str): Start date for this batch
            batch_end_date (str): End date for this batch

        Returns:
            list: List of weather data dictionaries for this batch
        """
        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
            f"timeline/{self.coordinates}/{batch_start_date}/{batch_end_date}"
            f"?unitGroup={self.unit}&key={self.api_key}&include=hours"
        )

        print(f"[Weather] Making API request to: {url}")

        # リトライ機能付きでAPIリクエストを実行
        max_retries = 3
        retry_delay = 5.0

        for attempt in range(max_retries):
            try:
                print(f"[Weather] Attempt {attempt + 1}/{max_retries}")
                res = requests.get(url, timeout=60)  # タイムアウトを60秒に延長
                print(f"[Weather] HTTP Status Code: {res.status_code}")

                if res.status_code != 200:
                    print(f"[Weather] HTTP Error: {res.status_code}")
                    print(f"[Weather] Response text: {res.text[:500]}")
                    if attempt < max_retries - 1:
                        print(f"[Weather] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    return []

                res.raise_for_status()
                break  # 成功した場合はループを抜ける

            except requests.exceptions.Timeout:
                print(f"[Weather] Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    print(f"[Weather] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("[Weather] Max retries exceeded for timeout")
                    return []

            except requests.exceptions.ConnectionError as e:
                print(f"[Weather] Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"[Weather] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("[Weather] Max retries exceeded for connection error")
                    return []

            except Exception as e:
                print(f"[Weather] Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"[Weather] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("[Weather] Max retries exceeded for unexpected error")
                    return []

        # 成功した場合の処理
        try:
            data = res.json()

            print(f"[Weather] API Response keys: {list(data.keys())}")
            print(f"[Weather] Number of days in response: {len(data.get('days', []))}")

            rows = []
            for d in data.get("days", []):
                hours = d.get("hours", [])
                for h in hours:
                    rows.append(
                        {
                            "datetime": f"{d['datetime']} {h['datetime']}",
                            self.temperature_col_name: h.get("temp"),
                            self.humidity_col_name: h.get("humidity"),
                            self.solar_col_name: h.get("solarradiation", 0),
                        }
                    )

            print(f"[Weather] Total rows collected for batch: {len(rows)}")
            return rows

        except requests.exceptions.RequestException as e:
            print(f"[Weather] Request Error: {e}")
            return []
        except Exception as e:
            print(f"[Weather] Unexpected Error: {e}")
            print(f"[Weather] Error type: {type(e).__name__}")
            return []

    def fetch(self) -> Optional[pd.DataFrame]:
        """
        Fetch weather data using batched requests to handle large date ranges

        Returns:
            pd.DataFrame: Combined weather data from all batches
        """
        try:
            # Generate date batches for the entire date range
            date_batches = self._generate_date_batches()

            print(f"[Weather] Fetching weather data in {len(date_batches)} batches...")
            print(f"[Weather] Date range: {self.start_date} to {self.end_date}")
            print(f"[Weather] Batch size: {self.batch_size_months} month(s)")

            all_rows = []

            # Process each batch
            for i, (batch_start, batch_end) in enumerate(date_batches, 1):
                print(
                    f"[Weather] Processing batch {i}/{len(date_batches)}: "
                    f"{batch_start} to {batch_end}"
                )

                # Fetch data for this batch
                batch_rows = self._fetch_single_batch(batch_start, batch_end)
                all_rows.extend(batch_rows)

                # Add delay between requests to avoid rate limiting
                if i < len(date_batches):  # Don't delay after the last batch
                    print(
                        f"[Weather] Waiting {self.delay_between_requests} seconds "
                        f"before next batch..."
                    )
                    time.sleep(self.delay_between_requests)

            print(f"[Weather] Completed fetching {len(all_rows)} total records")

            if not all_rows:
                print("[Weather] No data collected from any batch")
                return None

            # Create DataFrame from all collected rows
            df = pd.DataFrame(all_rows)
            if df.empty:
                print("[Weather] DataFrame is empty after processing")
                return None

            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            print(f"[Weather] Final DataFrame shape: {df.shape}")
            print(
                f"[Weather] Date range: {df['datetime'].min()} to {df['datetime'].max()}"
            )

            return df

        except Exception as e:
            print(f"[Weather] Unexpected Error in fetch: {e}")
            print(f"[Weather] Error type: {type(e).__name__}")
            return None
