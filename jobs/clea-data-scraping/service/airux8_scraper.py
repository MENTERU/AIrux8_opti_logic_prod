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
from pathlib import Path

import pandas as pd
from config.config_gcp import GCPEnv
from playwright.async_api import async_playwright
from service.bigquery import BigQuery
from service.storage import GCSClient

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Alrux8Scraper:
    def __init__(
        self,
        bq_dataset_id: str,
        bq_table_ac_control_raw: str,
        bq_table_ac_power_meter_raw: str,
    ):
        self.browser = None
        self.page = None
        self.context = None
        self.playwright = None
        self.download_summary = {}  # ダウンロード結果の要約
        self.downloaded_files = []  # ダウンロードファイルとデータタイプのマッピング

        # Data type name mapping (Japanese to English)
        # The website displays buttons in English, not Japanese
        self.data_type_mapping = {
            "A/C制御": "A/C Control",
            "A/C Power Meter": "A/C Power Meter",  # Already in English
        }

        # Get the base directory (job-clea-data-scraping directory)
        # This file is in service/, so we go up one level to get the job root
        script_dir = Path(__file__).parent.parent
        self.base_dir = script_dir.resolve()

        # Initialize GCS client with service account credentials if available
        service_account_path = GCPEnv.SERVICE_ACCOUNT_JSON
        if os.path.exists(service_account_path):
            logger.info(f"GCS認証情報を使用: {service_account_path}")
        else:
            logger.info("Application Default Credentialsを使用")
        self.gcs_client = GCSClient(
            project_id=GCPEnv.PROJECT_ID,
            bucket_id=GCPEnv.BUCKET_NAME,
            credentials_path=(
                service_account_path if os.path.exists(service_account_path) else None
            ),
        )
        # Initialize BigQuery client for ingesting scraped data
        self.bq_client = BigQuery(project_id=GCPEnv.PROJECT_ID)
        self.bq_dataset_id = bq_dataset_id
        self.bq_table_ac_control_raw = bq_table_ac_control_raw
        self.bq_table_ac_power_meter_raw = bq_table_ac_power_meter_raw
        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """必要なディレクトリを作成"""
        logs_dir = self.base_dir / "logs"
        downloads_dir = self.base_dir / "downloads"
        alrux8_data_dir = self.base_dir / "alrux8_data"

        os.makedirs(str(logs_dir), exist_ok=True)
        os.makedirs(str(downloads_dir), exist_ok=True)
        os.makedirs(str(alrux8_data_dir), exist_ok=True)
        logger.info(f"必要なディレクトリを作成しました: {self.base_dir}")

    async def setup_browser(self):
        """ブラウザのセットアップ"""
        try:
            logger.info("ブラウザセットアップ開始")

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,  # Required for Docker/Cloud Run (no display server)
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--ignore-certificate-errors",
                    "--disable-extensions",
                    "--disable-plugins",
                ],
            )

            self.context = await self.browser.new_context(
                accept_downloads=True,
                viewport={
                    "width": 1920,
                    "height": 1080,
                },  # Set viewport for headless mode
            )
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

            # Wait for the combobox input to be visible
            # Try to find the input - it might have "検索中..." placeholder while loading
            # or a different placeholder when ready
            logger.info("フロアコンボボックスを探しています...")

            # First, try to find any input that looks like a combobox/search input
            # Wait for the page to be ready after date selection
            await self.page.wait_for_timeout(2000)

            # Try multiple selectors - the placeholder might change after loading
            floor_combobox = None
            selectors = [
                "input[placeholder*='検索']",  # Any input with "検索" in placeholder
                "input[placeholder='検索中...']",  # Loading state
                "input[type='text']",  # Generic text input (fallback)
            ]

            for selector in selectors:
                try:
                    test_locator = self.page.locator(selector).first
                    await test_locator.wait_for(state="visible", timeout=10000)
                    floor_combobox = test_locator
                    logger.info(f"フロアコンボボックスを見つけました: {selector}")
                    break
                except Exception as e:
                    logger.debug(f"セレクター {selector} で見つかりませんでした: {e}")
                    continue

            if floor_combobox is None:
                raise Exception("フロアコンボボックスが見つかりませんでした")

            # Wait for element to be enabled and ready
            await self.page.wait_for_timeout(2000)
            is_enabled = await floor_combobox.is_enabled()
            if not is_enabled:
                logger.warning("フロアコンボボックスが有効ではありません。待機中...")
                await self.page.wait_for_timeout(5000)

            # Scroll into view and click
            await floor_combobox.scroll_into_view_if_needed()
            await floor_combobox.click(timeout=30000)
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
        """データタイプのダウンロード（ヘッドレスモード最適化版）"""
        max_retries = 3
        retry_delay = 5000  # 5 seconds between retries

        # Translate data type name if needed (Japanese to English)
        display_name = self.data_type_mapping.get(data_type, data_type)
        if display_name != data_type:
            logger.info(
                f"データタイプ名変換: '{data_type}' → '{display_name}' (ウェブサイト表示名)"
            )

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"リトライ {attempt + 1}/{max_retries}: {data_type}")
                    await self.page.wait_for_timeout(retry_delay)

                logger.info(
                    f"データダウンロード開始: {data_type} (表示名: {display_name})"
                )

                # Wait for page to be stable after previous operations
                await self.page.wait_for_load_state("domcontentloaded", timeout=30000)
                await self.page.wait_for_timeout(3000)

                # Check for and dismiss any modal/overlay that might be blocking the page
                try:
                    # Common modal close button selectors
                    modal_close_selectors = [
                        "button[aria-label='Close']",
                        "button.close",
                        ".modal-close",
                        "[data-dismiss='modal']",
                        ".overlay-close",
                    ]
                    for close_selector in modal_close_selectors:
                        close_button = self.page.locator(close_selector).first
                        if await close_button.is_visible(timeout=1000):
                            await close_button.click()
                            logger.info(
                                f"モーダル/オーバーレイを閉じました: {close_selector}"
                            )
                            await self.page.wait_for_timeout(1000)
                            break
                except Exception as modal_error:
                    # It's ok if no modal is found
                    logger.debug(f"モーダルチェック: {modal_error}")

                # Press Escape key to dismiss any potential overlays
                try:
                    await self.page.keyboard.press("Escape")
                    await self.page.wait_for_timeout(500)
                except Exception as escape_error:
                    logger.debug(f"Escapeキー押下エラー: {escape_error}")

                # Debug: Log all available link texts with comprehensive info
                try:
                    all_links_locator = self.page.locator("a")
                    link_count = await all_links_locator.count()
                    logger.info(f"ページ上のリンク総数: {link_count}")

                    # Get all text contents including those with the data type
                    all_links = await all_links_locator.all_text_contents()
                    links_with_data_type = [
                        link for link in all_links if display_name in link
                    ]
                    logger.info(
                        f"表示名 '{display_name}' を含むリンク: {links_with_data_type if links_with_data_type else '見つかりません'}"
                    )

                    # Log links containing "A/C" or "制御" or "Power"
                    relevant_links = [
                        link
                        for link in all_links
                        if any(
                            keyword in link
                            for keyword in ["A/C", "制御", "Power", "Meter", "control"]
                        )
                    ]
                    logger.info(f"関連リンク: {relevant_links[:20]}")
                except Exception as debug_error:
                    logger.warning(f"リンクデバッグログ取得失敗: {debug_error}")

                # Try multiple selector strategies for the button
                data_button = None

                # Determine href pattern based on data type (use original data_type for href pattern)
                href_pattern = None
                if data_type == "A/C Power Meter":
                    href_pattern = "/csv_logs/ac/power_meter/"
                elif data_type == "A/C制御" or display_name == "A/C Control":
                    href_pattern = "/csv_logs/ac/control/"

                selectors_to_try = []

                # Strategy 1: By href pattern (most reliable)
                if href_pattern:
                    selectors_to_try.append(
                        (
                            "href pattern",
                            self.page.locator(f"a[href*='{href_pattern}']").first,
                        )
                    )

                # Strategy 2: Exact text match using display_name
                selectors_to_try.append(
                    (
                        "exact text",
                        self.page.locator("a").filter(has_text=display_name).first,
                    )
                )

                # Strategy 3: Contains text (more flexible) using display_name
                selectors_to_try.append(
                    (
                        "contains text",
                        self.page.locator(f"a:has-text('{display_name}')").first,
                    )
                )

                # Strategy 4: Case-insensitive text match using display_name
                selectors_to_try.append(
                    (
                        "case-insensitive",
                        self.page.locator(f"a:text-is('{display_name}')").first,
                    )
                )

                # Strategy 5: Look for any link with display name words
                if " " in display_name:
                    words = display_name.split()
                    for word in words:
                        if len(word) > 2:  # Skip very short words
                            selectors_to_try.append(
                                (
                                    f"word: {word}",
                                    self.page.locator("a").filter(has_text=word).first,
                                )
                            )

                for selector_index, (strategy_name, selector) in enumerate(
                    selectors_to_try
                ):
                    try:
                        logger.info(
                            f"セレクター戦略 {selector_index + 1} ({strategy_name}) を試行中: {data_type}"
                        )
                        await selector.wait_for(state="visible", timeout=10000)

                        # Verify this is actually the right button by checking text or href
                        button_text = await selector.text_content()
                        button_href = await selector.get_attribute("href")
                        logger.info(
                            f"候補ボタン発見 - テキスト: '{button_text}', href: {button_href}"
                        )

                        # Validate it's the correct button
                        is_correct = False
                        if display_name in button_text:
                            is_correct = True
                        elif data_type in button_text:
                            is_correct = True
                        elif (
                            href_pattern and button_href and href_pattern in button_href
                        ):
                            is_correct = True

                        if is_correct:
                            data_button = selector
                            logger.info(
                                f"✓ 正しいボタン発見（戦略{selector_index + 1}: {strategy_name}）"
                            )
                            break
                        else:
                            logger.warning(f"候補は正しいボタンではありませんでした")

                    except Exception as selector_error:
                        logger.debug(
                            f"セレクター戦略 {selector_index + 1} ({strategy_name}) 失敗: {selector_error}"
                        )
                        continue

                if data_button is None:
                    raise Exception(
                        f"すべてのセレクター戦略でボタンが見つかりませんでした: {data_type}"
                    )

                # Wait for button to be attached to DOM
                logger.info(f"ボタンのDOM接続を確認中: {data_type}")
                await data_button.wait_for(state="attached", timeout=30000)

                # Scroll button into view to ensure it's interactable in headless mode
                logger.info(f"ボタンをスクロール表示: {data_type}")
                await data_button.scroll_into_view_if_needed()
                await self.page.wait_for_timeout(2000)

                # Verify button is enabled
                is_enabled = await data_button.is_enabled()
                if not is_enabled:
                    logger.warning(f"ボタンが無効です、待機中: {data_type}")
                    await self.page.wait_for_timeout(5000)

                # Get href for logging
                try:
                    href = await data_button.get_attribute("href", timeout=10000)
                    logger.info(f"ボタンのhref: {href}")
                except Exception as href_error:
                    logger.warning(f"href取得失敗（続行します）: {href_error}")

                # Perform download with extended timeout
                logger.info(f"ダウンロードクリック実行: {data_type}")
                async with self.page.expect_download(timeout=60000) as download_info:
                    await data_button.click(timeout=30000, force=True)

                download = await download_info.value
                filename = download.suggested_filename
                download_path = self.base_dir / "downloads" / filename
                await download.save_as(str(download_path))
                logger.info(f"ダウンロード完了: {download_path}")

                # Track downloaded file with its data type
                self.downloaded_files.append(
                    {
                        "filename": filename,
                        "data_type": data_type,
                        "local_path": str(download_path),
                    }
                )

                # Wait for page to stabilize after download
                await self.page.wait_for_load_state("networkidle", timeout=30000)
                logger.info(f"ダウンロード後の安定化完了: {data_type}")

                return True

            except Exception as error:
                logger.error(
                    f"データダウンロードエラー (試行 {attempt + 1}/{max_retries}): {error}"
                )

                if attempt < max_retries - 1:
                    # Take screenshot for debugging
                    try:
                        screenshot_path = (
                            self.base_dir / "logs" / f"error_{data_type}_{attempt}.png"
                        )
                        await self.page.screenshot(path=str(screenshot_path))
                        logger.info(
                            f"デバッグ用スクリーンショット保存: {screenshot_path}"
                        )
                    except Exception as screenshot_error:
                        logger.warning(
                            f"スクリーンショット保存失敗: {screenshot_error}"
                        )

                    # Don't reload page - just wait longer as AC selections would be lost
                    logger.info(f"待機後に再試行します（ページリロードなし）...")
                    await self.page.wait_for_timeout(5000)
                else:
                    logger.error(f"最大リトライ回数に達しました: {data_type}")
                    return False

        return False

    def _get_gcs_path_for_data_type(self, data_type: str) -> str:
        """データタイプに応じたGCSパスを取得"""
        if data_type == "A/C Power Meter":
            return GCPEnv.CLEA_AC_POWER_METER_PATH
        elif data_type == "A/C制御":
            return GCPEnv.CLEA_AC_CONTROL_PATH
        else:
            logger.warning(f"未知のデータタイプ: {data_type}, デフォルトパスを使用")
            return GCPEnv.INPUT_DATA_FOLDER

    def _transform_dataframe_for_bigquery(
        self, data_frame: pd.DataFrame, data_type: str
    ) -> pd.DataFrame:
        """スクレイピング結果のDataFrameをBigQueryスキーマに合わせて変換"""

        if data_type == "A/C制御":
            rename_mapping = {
                "A/C Name": "AC_Name",
                "Datetime": "Datetime",
                "Outdoor Temp.": "Outdoor_Temp",
                "Indoor Temp.": "Indoor_Temp",
                "A/C Set Temperature": "AC_Set_Temperature",
                "A/C ON/OFF": "AC_ON_OFF",
                "A/C Mode": "AC_Mode",
                "A/C Fan Speed": "AC_Fan_Speed",
                "Naive Energy Level": "Naive_Energy_Level",
                "Airux Energy Level": "Airux_Energy_Level",
                "Outdoor Room Temp.": "Outdoor_Room_Temp",
                "Outdoor Set Temp.": "Outdoor_Set_Temp",
                "Room Set Temp.": "Room_Set_Temp",  # ✔ FIX
            }

            data_frame = data_frame.rename(columns=rename_mapping)

            target_columns = [
                "AC_Name",
                "Datetime",
                "Outdoor_Temp",
                "Indoor_Temp",
                "AC_Set_Temperature",
                "AC_ON_OFF",
                "AC_Mode",
                "AC_Fan_Speed",
                "Naive_Energy_Level",
                "Airux_Energy_Level",
                "Outdoor_Room_Temp",
                "Outdoor_Set_Temp",
                "Room_Set_Temp",  # ✔ FIX
            ]

        elif data_type == "A/C Power Meter":
            rename_mapping = {
                "Mesh ID": "Mesh_ID",
                "PM Addr ID": "PM_Addr_ID",
                "Datetime": "Datetime",
                "Phase A": "Phase_A",
                "Phase B": "Phase_B",
                "Phase C": "Phase_C",
            }

            data_frame = data_frame.rename(columns=rename_mapping)

            target_columns = [
                "Mesh_ID",
                "PM_Addr_ID",
                "Datetime",
                "Phase_A",
                "Phase_B",
                "Phase_C",
            ]

        else:
            return data_frame

        # Keep only expected columns
        existing_columns = [c for c in target_columns if c in data_frame.columns]
        data_frame = data_frame[existing_columns]

        # Ensure timestamp type
        if "Datetime" in data_frame.columns:
            data_frame["Datetime"] = pd.to_datetime(
                data_frame["Datetime"], errors="coerce", utc=True
            )

        return data_frame

    async def organize_downloaded_files(self, store_name, start_date, end_date):
        """ダウンロードファイルの整理（ストア別フォルダ分け）、GCSへのアップロード、BigQueryへの書き込み"""
        try:
            logger.info("ダウンロードファイルの整理開始")

            # ストア別フォルダ作成（月別分けなし）
            store_folder = self.base_dir / "alrux8_data" / store_name
            os.makedirs(str(store_folder), exist_ok=True)

            # ダウンロードフォルダ内の全CSVファイルを移動
            downloads_dir = self.base_dir / "downloads"
            if downloads_dir.exists():
                csv_files = [
                    f for f in os.listdir(str(downloads_dir)) if f.endswith(".csv")
                ]

                # Track files by data type for GCS upload
                files_by_data_type = {}
                for file_info in self.downloaded_files:
                    data_type = file_info["data_type"]
                    filename = file_info["filename"]
                    if filename in csv_files:
                        if data_type not in files_by_data_type:
                            files_by_data_type[data_type] = []
                        files_by_data_type[data_type].append(file_info)

                # Move files and upload to GCS
                for csv_file in csv_files:
                    source_path = downloads_dir / csv_file
                    dest_path = store_folder / csv_file

                    # Find the data type for this file
                    file_data_type = None
                    for data_type, file_list in files_by_data_type.items():
                        if any(f["filename"] == csv_file for f in file_list):
                            file_data_type = data_type
                            break

                    # Move file locally
                    shutil.move(str(source_path), str(dest_path))
                    logger.info(f"ファイル移動完了: {dest_path}")

                    # Upload to GCS
                    if file_data_type:
                        gcs_path = self._get_gcs_path_for_data_type(file_data_type)
                        gcs_file_path = f"{gcs_path}{csv_file}"

                        try:
                            df = pd.read_csv(dest_path, low_memory=False)
                            # GCS upload
                            self.gcs_client.write_csv(df, gcs_file_path)
                            logger.info(f"GCSアップロード完了: {gcs_file_path}")

                            # BigQuery ingest (append to raw tables)
                            try:
                                transformed_df = self._transform_dataframe_for_bigquery(
                                    df, file_data_type
                                )
                                if not transformed_df.empty:
                                    if file_data_type == "A/C制御":
                                        table_name = self.bq_table_ac_control_raw
                                    elif file_data_type == "A/C Power Meter":
                                        table_name = self.bq_table_ac_power_meter_raw
                                    else:
                                        table_name = None

                                    if table_name is not None:
                                        self.bq_client.write_dataframe(
                                            transformed_df,
                                            table_name=table_name,
                                            dataset_id=self.bq_dataset_id,
                                            if_exists="append",
                                        )
                                        logger.info(
                                            f"BigQuery書き込み完了: {self.bq_dataset_id}.{table_name} "
                                            f"({len(transformed_df)} rows)"
                                        )
                                    else:
                                        logger.warning(
                                            f"BigQueryテーブルが未定義のデータタイプ: {file_data_type}"
                                        )
                                else:
                                    logger.warning(
                                        f"BigQueryに書き込む行がありません: {csv_file}"
                                    )
                            except Exception as bq_error:
                                logger.error(
                                    f"BigQuery書き込みエラー ({csv_file}): {bq_error}"
                                )
                        except Exception as upload_error:
                            logger.error(
                                f"GCSアップロードエラー ({csv_file}): {upload_error}"
                            )
                    else:
                        logger.warning(f"データタイプが見つかりません: {csv_file}")

                # Delete downloads folder after all files are processed
                try:
                    if downloads_dir.exists():
                        shutil.rmtree(str(downloads_dir))
                        logger.info("downloadsフォルダを削除しました")
                except Exception as cleanup_error:
                    logger.warning(f"downloadsフォルダの削除エラー: {cleanup_error}")

                # Delete alrux8_data folder after all files are uploaded to GCS
                try:
                    alrux8_data_dir = self.base_dir / "alrux8_data"
                    if alrux8_data_dir.exists():
                        shutil.rmtree(str(alrux8_data_dir))
                        logger.info("alrux8_dataフォルダを削除しました")
                except Exception as cleanup_error:
                    logger.warning(f"alrux8_dataフォルダの削除エラー: {cleanup_error}")

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
                data_types = ["A/C Power Meter", "A/C制御"]

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
                    for download_index, data_type in enumerate(data_types):
                        try:
                            success = await self.download_data_type(data_type)
                            if success:
                                floor_files.append(data_type)
                            else:
                                logger.warning(
                                    f"データタイプ {data_type} のダウンロードに失敗しました"
                                )
                            # Pause between downloads - longer for subsequent downloads
                            if download_index < len(data_types) - 1:
                                wait_time = 3000  # 3 seconds between downloads
                                logger.info(
                                    f"次のダウンロードまで {wait_time}ms 待機中..."
                                )
                                await self.page.wait_for_timeout(wait_time)
                        except Exception as error:
                            logger.error(
                                f"データタイプ {data_type} のダウンロードエラー: {error}"
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

            # ダウンロードファイルの整理とGCSアップロード
            await self.organize_downloaded_files(store_name, start_date, end_date)

            # Clear downloaded files tracking for next run
            self.downloaded_files = []

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
