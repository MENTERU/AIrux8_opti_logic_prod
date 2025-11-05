#!/usr/bin/env python3
"""
Alrux8データ取得スクレイパー（Playwright版）
本番用の整理されたバージョン
"""

import asyncio
import logging
import os
import shutil
from datetime import datetime

from playwright.async_api import async_playwright

from config.private_information import ALRUX8_PASSWORD, ALRUX8_USERNAME

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("logs/alrux8_scraper.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Alrux8Scraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.context = None
        self.playwright = None
        self.download_summary = {}  # ダウンロード結果の要約

    async def setup_browser(self):
        """ブラウザのセットアップ"""
        try:
            logger.info("ブラウザセットアップ開始")

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=False,  # デバッグのため表示
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--ignore-certificate-errors",
                    "--disable-extensions",
                    "--disable-plugins",
                ],
            )

            self.context = await self.browser.new_context(accept_downloads=True)
            self.page = await self.context.new_page()

            logger.info("ブラウザセットアップ完了")
            return True

        except Exception as e:
            logger.error(f"ブラウザセットアップエラー: {e}")
            return False

    async def login(self, username, password):
        """ログイン処理"""
        try:
            logger.info(f"ログイン開始: {username}")

            await self.page.goto("https://www.airux8.com/login")
            await self.page.wait_for_load_state("networkidle")

            await self.page.fill("input[name='username']", username)
            await self.page.fill("input[name='password']", password)
            await self.page.wait_for_timeout(1000)

            await self.page.click("button[type='submit']")
            await self.page.wait_for_timeout(5000)

            current_url = self.page.url
            logger.info(f"ログイン後のURL: {current_url}")

            if "login" not in current_url.lower():
                logger.info("ログイン成功")
                return True
            else:
                logger.error("ログイン失敗")
                await self.page.screenshot(path="screenshots/login_failed.png")
                return False

        except Exception as e:
            logger.error(f"ログインエラー: {e}")
            return False

    async def navigate_to_logs(self):
        """Logsページに移動"""
        try:
            logger.info("Logsページに移動")
            await self.page.goto("https://www.airux8.com/airux-admin/logs")
            await self.page.wait_for_load_state("networkidle")
            return True
        except Exception as e:
            logger.error(f"Logsページ移動エラー: {e}")
            return False

    async def select_date_range(self, start_date, end_date):
        """日付範囲選択（開始月と終了月が異なる場合も対応）"""
        try:
            logger.info(
                f"日付範囲選択: {start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}"
            )

            # 日付ピッカーが表示されるまで待機
            logger.info("日付ピッカー要素の表示を待機中...")
            await self.page.wait_for_selector("select", state="visible", timeout=30000)
            await self.page.wait_for_timeout(1000)  # 追加の安定化待機

            # 年・月の選択（開始日）
            year_selector = self.page.locator("select").nth(1)
            await year_selector.wait_for(state="visible", timeout=30000)
            await year_selector.select_option(str(start_date.year))
            await self.page.wait_for_timeout(500)

            month_selector = self.page.locator("select").nth(0)
            await month_selector.wait_for(state="visible", timeout=30000)
            await month_selector.select_option(str(start_date.month - 1))
            await self.page.wait_for_timeout(1000)

            # 開始日クリック
            logger.info("開始日ボタンの表示を待機中...")
            await self.page.wait_for_selector(
                "button.rdrDay:not(.rdrDayPassive):not(.rdrDayDisabled)",
                state="visible",
                timeout=30000,
            )
            start_day = (
                self.page.locator(
                    "button.rdrDay:not(.rdrDayPassive):not(.rdrDayDisabled)"
                )
                .filter(has_text=str(start_date.day))
                .first
            )
            await start_day.wait_for(state="visible", timeout=30000)
            await start_day.scroll_into_view_if_needed()
            await start_day.click()
            await self.page.wait_for_timeout(500)

            # 終了日が開始月と異なる場合、カレンダーを切り替え
            if start_date.year != end_date.year or start_date.month != end_date.month:
                await year_selector.select_option(str(end_date.year))
                await self.page.wait_for_timeout(500)
                await month_selector.select_option(str(end_date.month - 1))
                await self.page.wait_for_timeout(1000)

            # 終了日クリック
            logger.info("終了日ボタンの表示を待機中...")
            end_day = (
                self.page.locator(
                    "button.rdrDay:not(.rdrDayPassive):not(.rdrDayDisabled)"
                )
                .filter(has_text=str(end_date.day))
                .first
            )
            await end_day.wait_for(state="visible", timeout=30000)
            await end_day.scroll_into_view_if_needed()
            await end_day.click()
            await self.page.wait_for_timeout(500)

            logger.info("日付範囲選択完了")
            return True

        except Exception as e:
            logger.error(f"日付範囲選択エラー: {e}")
            return False

    async def get_available_floors(self):
        """利用可能なフロア一覧を取得"""
        try:
            logger.info("フロア一覧取得開始")

            floor_combobox = self.page.locator("input[placeholder='検索中...']").first
            await floor_combobox.click()
            await self.page.wait_for_timeout(1000)

            floor_options = self.page.locator("li[role='option']")
            option_count = await floor_options.count()

            floors = []
            for i in range(option_count):
                text = await floor_options.nth(i).text_content()
                if text and text.strip():
                    floors.append(text.strip())

            logger.info(f"フロア一覧: {floors}")
            return floors

        except Exception as e:
            logger.error(f"フロア一覧取得エラー: {e}")
            return []

    async def select_floor(self, floor_name):
        """フロア選択"""
        try:
            logger.info(f"フロア選択: {floor_name}")

            floor_option = (
                self.page.locator("li[role='option']").filter(has_text=floor_name).first
            )
            await floor_option.click()
            await self.page.wait_for_timeout(1000)

            return True

        except Exception as e:
            logger.error(f"フロア選択エラー: {e}")
            return False

    async def get_floor_ac_master(self):
        """フロアごとのA/C機器リストをマスタとして取得"""
        logger.info("フロア・A/Cマスタ情報取得開始")
        master = {}
        floors = await self.get_available_floors()
        for floor in floors:
            # フロア選択
            await self.select_floor(floor)
            await self.page.wait_for_timeout(1000)
            # A/C機器リスト取得
            ac_combobox = self.page.locator("input[aria-multiselectable='true']").first
            await ac_combobox.click()
            await self.page.wait_for_timeout(1000)
            ac_options = self.page.locator("li[role='option']")
            ac_count = await ac_options.count()
            ac_list = []
            for i in range(ac_count):
                ac_text = await ac_options.nth(i).text_content()
                if ac_text and ac_text.strip():
                    ac_list.append(ac_text.strip())
            # フロア名と一致するものは除外
            ac_list = [ac for ac in ac_list if ac not in floors]
            master[floor] = ac_list
            # A/C選択解除（次のフロアのため）
            await ac_combobox.press("Escape")
            await self.page.wait_for_timeout(500)
        logger.info(f"マスタ情報: {master}")
        self.floor_ac_master = master
        return master

    async def select_ac_units_by_names(self, ac_names):
        """A/C機器名リストで選択"""
        try:
            logger.info(f"A/C機器選択: {ac_names}")
            ac_combobox = self.page.locator("input[aria-multiselectable='true']").first
            await ac_combobox.click()
            await self.page.wait_for_timeout(1000)
            ac_options = self.page.locator("li[role='option']")
            selected = 0
            for name in ac_names:
                option = ac_options.filter(has_text=name).first
                await option.click()
                await self.page.wait_for_timeout(500)
                selected += 1
            logger.info(f"A/C機器選択完了: {selected}/{len(ac_names)}台")
            return selected
        except Exception as e:
            logger.error(f"A/C機器名指定選択エラー: {e}")
            return 0

    async def select_all_ac_units(self):
        """全A/C機器選択（安定版）"""
        try:
            logger.info("全A/C機器選択開始")

            ac_combobox = self.page.locator("input[aria-multiselectable='true']").first
            await ac_combobox.click()
            await self.page.wait_for_timeout(1000)

            ac_options = self.page.locator("li[role='option']")
            option_count = await ac_options.count()
            logger.info(f"A/Cオプション数: {option_count}")

            # 最初の5台のみ選択（安定性重視）
            max_selections = min(5, option_count)
            selected_count = 0

            for i in range(max_selections):
                try:
                    option = ac_options.nth(i)
                    await option.click()
                    await self.page.wait_for_timeout(500)  # 処理間隔を延長
                    selected_count += 1
                    logger.info(f"A/C機器 {i} 選択完了")
                except Exception as e:
                    logger.warning(f"A/C機器 {i} 選択失敗: {e}")
                    # 失敗しても続行
                    continue

            logger.info(f"A/C機器選択完了: {selected_count}/{max_selections}台")
            return selected_count

        except Exception as e:
            logger.error(f"A/C機器選択エラー: {e}")
            return 0

    async def download_data_type(self, data_type):
        """データタイプのダウンロード"""
        try:
            logger.info(f"データダウンロード開始: {data_type}")

            data_button = self.page.locator("a").filter(has_text=data_type).first
            href = await data_button.get_attribute("href")
            logger.info(f"ボタンのhref: {href}")

            async with self.page.expect_download() as download_info:
                await data_button.click()

            download = await download_info.value
            filename = download.suggested_filename
            await download.save_as(f"downloads/{filename}")
            logger.info(f"ダウンロード完了: {filename}")

            return True

        except Exception as e:
            logger.error(f"データダウンロードエラー: {e}")
            return False

    async def organize_downloaded_files(self, store_name, start_date, end_date):
        """ダウンロードファイルの整理（ストア別フォルダ分け）"""
        try:
            logger.info("ダウンロードファイルの整理開始")

            # ストア別フォルダ作成（月別分けなし）
            store_folder = f"alrux8_data/{store_name}"
            os.makedirs(store_folder, exist_ok=True)

            # ダウンロードフォルダ内の全CSVファイルを移動
            downloads_dir = "downloads"
            if os.path.exists(downloads_dir):
                csv_files = [f for f in os.listdir(downloads_dir) if f.endswith(".csv")]
                for csv_file in csv_files:
                    source_path = os.path.join(downloads_dir, csv_file)
                    # ストア名をファイル名に含めない
                    dest_path = os.path.join(store_folder, csv_file)

                    shutil.move(source_path, dest_path)
                    logger.info(f"ファイル移動完了: {dest_path}")

            return True

        except Exception as e:
            logger.error(f"ファイル整理エラー: {e}")
            return False

    def log_download_summary(self):
        """ダウンロード結果の要約ログ"""
        logger.info("=" * 50)
        logger.info("📊 データ取得結果要約")
        logger.info("=" * 50)

        total_floors = len(self.download_summary)
        total_files = sum(len(files) for files in self.download_summary.values())

        logger.info(f"処理フロア数: {total_floors}")
        logger.info(f"取得ファイル数: {total_files}")
        logger.info("")

        for floor, files in self.download_summary.items():
            logger.info(f"🏢 {floor}: {len(files)}ファイル")
            for file in files:
                logger.info(f"  📄 {file}")

        logger.info("=" * 50)

    async def run_scraping(
        self, username, password, store_name, start_date, end_date, data_types=None
    ):
        """スクレイピング実行"""
        try:
            logger.info(f"スクレイピング開始: {store_name}")

            if data_types is None:
                data_types = ["A/C Table", "A/C Power Meter"]

            # ブラウザセットアップ
            if not await self.setup_browser():
                return False

            # ログイン
            if not await self.login(username, password):
                return False

            # Logsページに移動
            if not await self.navigate_to_logs():
                return False

            # 日付範囲選択
            if not await self.select_date_range(start_date, end_date):
                return False

            # フロア・A/Cマスタ取得
            master = await self.get_floor_ac_master()
            if not master:
                logger.error("フロア・A/Cマスタ情報取得失敗")
                return False

            # フロアを逆順で処理
            floors_reversed = list(reversed(list(master.keys())))
            logger.info(f"フロア処理順序（逆順）: {floors_reversed}")

            for floor in floors_reversed:
                logger.info(f"フロア処理開始: {floor}")
                floor_files = []
                try:
                    # フロア選択
                    if not await self.select_floor(floor):
                        logger.warning(f"フロア {floor} の選択に失敗、スキップ")
                        continue

                    # マスタからA/C機器リスト取得
                    ac_list = master[floor]
                    if not ac_list:
                        logger.error(f"フロア {floor} にA/C機器がありません")
                        continue

                    print(f"A/C機器リスト: {ac_list}")
                    # 最大5台、最低3台
                    if len(ac_list) < 3:
                        ac_to_select = ac_list
                    else:
                        ac_to_select = ac_list[: min(len(ac_list), 5)]
                    selected_count = await self.select_ac_units_by_names(ac_to_select)
                    if selected_count < 3:
                        logger.warning(
                            f"フロア {floor} でA/C機器が不足しています ({selected_count}台)"
                        )
                        continue

                    # データタイプごとにダウンロード
                    for data_type in data_types:
                        try:
                            success = await self.download_data_type(data_type)
                            if success:
                                floor_files.append(data_type)
                            await self.page.wait_for_timeout(
                                2000
                            )  # ダウンロード間隔を調整
                        except Exception as e:
                            logger.error(
                                f"データタイプ {data_type} のダウンロードエラー: {e}"
                            )
                            continue

                    # フロア処理結果を記録
                    self.download_summary[floor] = floor_files
                    logger.info(
                        f"フロア {floor} 処理完了: {len(floor_files)}ファイル取得"
                    )

                except Exception as e:
                    logger.error(f"フロア {floor} 処理エラー: {e}")
                    # エラーが発生しても次のフロアに進む
                    continue

                await self.page.wait_for_timeout(1000)  # フロア間の待機時間を調整

            # ダウンロードファイルの整理
            await self.organize_downloaded_files(store_name, start_date, end_date)

            # ダウンロード結果の要約ログ
            self.log_download_summary()

            logger.info("スクレイピング完了")
            return True

        except Exception as e:
            logger.error(f"スクレイピングエラー: {e}")
            return False

    async def close(self):
        """ブラウザを閉じる"""
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("ブラウザを閉じました")
        except Exception as e:
            logger.error(f"ブラウザクローズエラー: {e}")


async def main():
    """メイン実行"""
    try:
        # ディレクトリ作成
        os.makedirs("logs", exist_ok=True)
        os.makedirs("downloads", exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)
        os.makedirs("alrux8_data", exist_ok=True)

        # スクレイパー作成
        scraper = Alrux8Scraper()

        # 設定
        username = ALRUX8_USERNAME
        password = ALRUX8_PASSWORD
        store_name = "クレア様"
        start_date = datetime(2025, 8, 1)
        end_date = datetime(2025, 9, 30)
        data_types = ["A/C Table", "A/C Power Meter"]

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
            print("✅ データ取得完了")
        else:
            print("❌ データ取得失敗")

    except Exception as e:
        logger.error(f"メイン実行エラー: {e}")
    finally:
        if "scraper" in locals():
            await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
