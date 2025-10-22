# =============================================================================
# エアコン最適化システム - 実行サンプル
# =============================================================================

import argparse
import sys
from typing import Optional

from config.private_information import WEATHER_API_KEY
from processing.aggregator import aggregation_runner
from processing.preprocessor import preprocessing_runner
from processing.utilities.master_data_loader import master_data_loader_runner


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="エアコン最適化システム - 実行スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        実行例:
        uv run run_optimization.py --preprocess-only
        uv run run_optimization.py --aggregate-only
                """,
    )

    # Store selection
    parser.add_argument(
        "--store", type=str, default="Clea", help="対象ストア(default:Clea)"
    )
    parser.add_argument("--preprocess-only", action="store_true", help="前処理のみ実行")
    parser.add_argument("--aggregate-only", action="store_true", help="集約のみ実行")
    parser.add_argument("--start-date", type=str, help="最適化開始日 (YYYY-MM-DD形式)")
    parser.add_argument("--end-date", type=str, help="最適化終了日 (YYYY-MM-DD形式)")

    return parser.parse_args()


def run_optimization_for_store(
    store_name,
    execution_mode: str = "full",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    指定されたストアの最適化を実行

    Args:
        store_name: ストア名
        execution_mode: 実行モード ("preprocess", "aggregate", "full")
        start_date: 開始日 (オプション)
        end_date: 終了日 (オプション)

    Returns:
        bool: 実行成功時True、失敗時False
    """
    print(f"🚀 {store_name}の最適化パイプライン開始 (モード: {execution_mode})")
    store_master_file = master_data_loader_runner(store_name)
    if store_master_file is None:
        print(f"❌ エラーが発生しました: ストアマスタデータの読み込みに失敗しました")
        return False

    try:
        if execution_mode == "preprocess":
            print("📊 前処理のみ実行")
            preprocessing_runner(
                store_name=store_name,
                store_master_file=store_master_file,
                weather_api_key=WEATHER_API_KEY,
                temperature_std_multiplier=5.0,
                power_std_multiplier=5.0,
                export_temp_range_stats=False,
            )
            print("✅ 前処理完了")
            return True

        elif execution_mode == "aggregate":
            logging.info("=" * 50)
            print("🔄 集約のみ実行")
            aggregation_runner(
                store_name=store_name,
                store_master_file=store_master_file,
                start_date=start_date,
                end_date=end_date,
                weather_api_key=WEATHER_API_KEY,
                freq="1H",
            )
            print("✅ 集約完了")
            return True
        elif execution_mode == "optimize":
            print("=" * 50)
            print("🔄 最適化のみ実行")
            # Get target date from environment or default to tomorrow
            target_date = os.environ.get(
                "TARGET_DATE", (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            print.info(f"Optimizing for date: {target_date}")

            try:
                # Initialize optimization runner
                print.info("Initializing Optimization Runner...")
                runner = OptimizationRunner(data_dir=PREPROCESSED_DATA_DIR)

                # Load historical data
                print.info("Loading historical HVAC data...")
                runner.load_all_data(target_date=target_date)

                # Run complete optimization
                print.info(f"Running optimization for {target_date}...")
                results = runner.run_optimization(target_date)

                print("Optimization completed successfully!")
                return 0

            except Exception as e:
                print(f"Optimization failed: {e}")
                return 1

        else:  # full
            print("🔄 フルパイプライン実行")
            # TODO: フルパイプライン実行
            print("✅ フルパイプライン完了")
            return True

    except Exception as error:
        print(f"❌ エラーが発生しました: {error}")
        return False


def main():
    """メイン実行関数"""
    # コマンドライン引数の解析
    args = parse_arguments()

    # 実行モードの決定 (複数のフラグを組み合わせ可能)
    execution_modes = []
    if args.preprocess_only:
        execution_modes.append("preprocess")
    if args.aggregate_only:
        execution_modes.append("aggregate")

    # store_name
    if args.store is None:
        store_name = "Clea"  # default store name for development
    else:
        store_name = args.store

    if not execution_modes:
        execution_mode = "full"
        success = run_optimization_for_store(
            store_name=store_name,
            execution_mode=execution_mode,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    elif len(execution_modes) == 1:
        execution_mode = execution_modes[0]
        success = run_optimization_for_store(
            store_name=store_name,
            execution_mode=execution_mode,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        # Multiple modes specified - execute them in sequence
        print(f"🔄 Multiple execution modes specified: {execution_modes}")
        print(f"🔄 Will execute them in sequence")

        success = True
        # Execute each mode in sequence
        for i, mode in enumerate(execution_modes):
            print(f"\n{'='*70}")
            print(f"🏢 {store_name} - Step {i+1}/{len(execution_modes)}: {mode}")
            print(f"{'='*70}")

            step_success = run_optimization_for_store(
                store_name=store_name,
                execution_mode=mode,
                start_date=args.start_date,
                end_date=args.end_date,
            )

            if not step_success:
                print(f"❌ Step {i+1} ({mode}) failed. Stopping execution.")
                success = False
                break

    # 終了コードの設定
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
