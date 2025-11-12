from datetime import datetime

import pytz
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from service.airux8_scraper import Alrux8Scraper
from service.secretmanager import SecretManagerClient

app = FastAPI()

# Track if scraping is currently running to prevent concurrent runs
scraping_in_progress = False


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    global scraping_in_progress
    return JSONResponse(
        {
            "status": "healthy",
            "scraping_in_progress": scraping_in_progress,
            "message": "サーバーは稼働中です",
        },
        status_code=200,
    )


@app.post("/run_scraping")
async def run_scraping():
    """スクレイピング実行エンドポイント"""
    global scraping_in_progress

    # Prevent concurrent scraping runs
    if scraping_in_progress:
        print(
            "⚠️ スクレイピングは既に実行中です。前のリクエストが完了するまで待機してください。"
        )
        return JSONResponse(
            {"message": "スクレイピングは既に実行中です", "status": "busy"},
            status_code=409,  # Conflict
        )

    scraper = None
    try:
        scraping_in_progress = True
        print("=" * 60)
        print("📥 新しいスクレイピングリクエストを受信しました")
        print("=" * 60)

        # ログイン情報をSecret Managerから取得
        secret_manager = SecretManagerClient()
        login_info = secret_manager.get_secret_as_dict("AIRUX8_WEB_LOGIN_INFO")

        if not login_info:
            scraping_in_progress = False
            return JSONResponse(
                {
                    "message": "Failed to retrieve login information from Secret Manager",
                    "status": "error",
                },
                status_code=500,
            )

        username = login_info.get("username")
        password = login_info.get("password")

        if not username or not password:
            scraping_in_progress = False
            return JSONResponse(
                {
                    "message": "Login information is missing username or password",
                    "status": "error",
                },
                status_code=500,
            )

        print("✅ Successfully retrieved login information from Secret Manager")

        # スクレイパー作成
        scraper = Alrux8Scraper()
        store_name = "Clea"
        start_date = datetime(2025, 11, 12, tzinfo=pytz.timezone("Asia/Tokyo"))
        end_date = datetime(2025, 11, 12, tzinfo=pytz.timezone("Asia/Tokyo"))
        data_types = ["A/C Power Meter", "A/C制御"]

        print(f"=== {store_name} データ取得開始 ===")
        print(
            f"期間: {start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"データ: {', '.join(data_types)}")

        # スクレイピング実行
        success = await scraper.run_scraping(
            username=username,
            password=password,
            store_name=store_name,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
        )

        if success:
            print("=" * 60)
            print("✅ データ取得完了 - サーバーは待機中です")
            print("=" * 60)
            scraping_in_progress = False
            return JSONResponse(
                {
                    "message": "データ取得完了",
                    "status": "success",
                    "store": store_name,
                    "date_range": f"{start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}",
                },
                status_code=200,
            )
        else:
            print("=" * 60)
            print("❌ データ取得失敗 - サーバーは待機中です")
            print("=" * 60)
            scraping_in_progress = False
            return JSONResponse(
                {
                    "message": "データ取得失敗",
                    "status": "failed",
                    "store": store_name,
                },
                status_code=200,  # Return 200 even on failure, but status indicates failure
            )
    except Exception as error:
        print("=" * 60)
        print(f"❌ メイン実行エラー: {error}")
        print("=" * 60)
        scraping_in_progress = False
        return JSONResponse(
            {
                "message": f"メイン実行エラー: {str(error)}",
                "status": "error",
            },
            status_code=500,
        )
    finally:
        scraping_in_progress = False
        if scraper is not None:
            await scraper.close()
        print("🔄 サーバーは次のリクエストを待機中...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
