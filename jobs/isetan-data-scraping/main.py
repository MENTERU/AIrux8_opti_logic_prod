#!/usr/bin/env python3
"""
Cloud Run Job entrypoint - runs scraping directly
"""
import asyncio
import sys
from datetime import datetime, timedelta

import pytz
from config.config_gcp import GCPEnv
from service.airux8_scraper import Alrux8Scraper
from service.secretmanager import SecretManagerClient


async def main():
    """
    Main function to run scraping job


    Args:
        None

    Returns:
        None
    """
    scraper = None
    try:
        print("=" * 60)
        print("📥 Cloud Run Job: スクレイピング開始")
        print("=" * 60)

        # ログイン情報をSecret Managerから取得
        secret_manager = SecretManagerClient()
        login_info = secret_manager.get_secret_as_dict(GCPEnv.LOGIN_INFO_SECRET_NAME)

        if not login_info:
            print("❌ Failed to retrieve login information from Secret Manager")
            sys.exit(1)

        # ストア名に応じた認証情報を取得
        store_credentials = login_info.get(GCPEnv.STORE_NAME)
        if not store_credentials:
            print(
                f"❌ Login information for store '{GCPEnv.STORE_NAME}' not found in secret"
            )
            print(f"Available stores: {list(login_info.keys())}")
            sys.exit(1)

        username = store_credentials.get("username")
        password = store_credentials.get("password")

        if not username or not password:
            print(
                f"❌ Login information for '{GCPEnv.STORE_NAME}' is missing username or password"
            )
            sys.exit(1)

        print(
            f"✅ Successfully retrieved login information for '{GCPEnv.STORE_NAME}' from Secret Manager"
        )

        # スクレイパー作成（BigQueryテーブル設定を渡す）
        scraper = Alrux8Scraper(
            bq_dataset_id=GCPEnv.BQ_DATASET_ISETAN,
            bq_table_ac_control_raw=GCPEnv.BQ_TABLE_AC_CONTROL_RAW,
            bq_table_ac_power_meter_raw=GCPEnv.BQ_TABLE_AC_POWER_METER_RAW,
        )

        # Original date calculation
        today = datetime.now(pytz.timezone("Asia/Tokyo"))
        yesterday_date = (today - timedelta(days=1)).date()
        # Set both start and end date to yesterday (date only, no time components)
        start_date = datetime(
            yesterday_date.year,
            yesterday_date.month,
            yesterday_date.day,
            tzinfo=pytz.timezone("Asia/Tokyo"),
        )
        end_date = datetime(
            yesterday_date.year,
            yesterday_date.month,
            yesterday_date.day,
            tzinfo=pytz.timezone("Asia/Tokyo"),
        )

        # You can specify the date range: 2024-08-01 to 2024-08-31
        # start_date = datetime(
        #     2024,
        #     8,
        #     1,
        #     tzinfo=pytz.timezone("Asia/Tokyo"),
        # )
        # end_date = datetime(
        #     2024,
        #     8,
        #     31,
        #     tzinfo=pytz.timezone("Asia/Tokyo"),
        # )
        data_types = ["A/C Power Meter", "A/C制御"]

        print(f"=== {GCPEnv.STORE_NAME} データ取得開始 ===")
        print(
            f"期間: {start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"データ: {', '.join(data_types)}")

        # スクレイピング実行
        success = await scraper.run_scraping(
            username=username,
            password=password,
            store_name=GCPEnv.STORE_NAME,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
        )

        if success:
            print("=" * 60)
            print("✅ データ取得完了")
            print("=" * 60)
            sys.exit(0)
        else:
            print("=" * 60)
            print("❌ データ取得失敗")
            print("=" * 60)
            sys.exit(1)
    except Exception as error:
        print("=" * 60)
        print(f"❌ メイン実行エラー: {error}")
        import traceback

        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
    finally:
        if scraper is not None:
            await scraper.close()
        print("🔄 ジョブ終了")


if __name__ == "__main__":
    asyncio.run(main())
