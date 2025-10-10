# =============================================================================
# エアコン最適化システム - 実行サンプル
# =============================================================================

import argparse
import sys
from typing import Optional

import pandas as pd

from analysis.reporting import generate_all_reports, reset_outputs
from config.private_information import WEATHER_API_KEY
from optimization.aircon_optimizer import AirconOptimizer


def print_table(
    data,
    title=None,
    headers=None,
    column_widths=None,
    show_index=False,
    hide_headers=False,
):
    """
    Print data in a formatted table style

    Args:
        data: List of lists, dict, or pandas DataFrame
        title: Optional title for the table
        headers: Optional list of column headers
        column_widths: Optional list of column widths
        show_index: Whether to show row indices
        hide_headers: Whether to hide headers completely
    """
    if isinstance(data, dict):
        # Convert dict to list of [key, value] pairs
        data = [[str(k), str(v)] for k, v in data.items()]
        if headers is None:
            headers = ["Key", "Value"]

    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to list of lists
        data = data.values.tolist()
        if headers is None:
            headers = list(data.columns) if hasattr(data, "columns") else []

    if not data:
        print("No data to display")
        return

    # Ensure data is list of lists
    if not isinstance(data[0], (list, tuple)):
        data = [[str(item)] for item in data]
        if headers is None:
            headers = ["Value"]

    # Auto-generate headers if not provided and not hiding headers
    if headers is None and not hide_headers:
        headers = [f"Column {i+1}" for i in range(len(data[0]))]

    # Calculate column widths
    if column_widths is None:
        column_widths = []
        if headers and not hide_headers:
            # Include header length in width calculation
            for i, header in enumerate(headers):
                max_width = len(str(header))
                for row in data:
                    if i < len(row):
                        max_width = max(max_width, len(str(row[i])))
                column_widths.append(max_width + 2)  # Add padding
        else:
            # Only consider data width
            for i in range(len(data[0])):
                max_width = 0
                for row in data:
                    if i < len(row):
                        max_width = max(max_width, len(str(row[i])))
                column_widths.append(max_width + 2)  # Add padding

    # Print title
    if title:
        total_width = sum(column_widths) + len(column_widths) - 1
        print(f"\n{'='*total_width}")
        print(f"{title:^{total_width}}")
        print(f"{'='*total_width}")

    # Print headers only if not hiding them
    if headers and not hide_headers:
        header_line = ""
        for i, (header, width) in enumerate(zip(headers, column_widths)):
            header_line += f"{header:^{width}}"
            if i < len(headers) - 1:
                header_line += "|"
        print(header_line)

        # Print separator
        separator = ""
        for i, width in enumerate(column_widths):
            separator += "-" * width
            if i < len(column_widths) - 1:
                separator += "+"
        print(separator)

    # Print data rows
    for row_idx, row in enumerate(data):
        row_line = ""
        for i, (cell, width) in enumerate(zip(row, column_widths)):
            if i == 0 and show_index:
                cell_str = f"{row_idx}: {str(cell)}"
            else:
                cell_str = str(cell)
            row_line += f"{cell_str:<{width}}"
            if i < len(row) - 1:
                row_line += "|"
        print(row_line)

    # Print bottom border
    if title:
        total_width = sum(column_widths) + len(column_widths) - 1
        print(f"{'='*total_width}")


def print_optimization_summary(store_name, results, processing_times=None):
    """
    Print optimization results in a formatted table

    Args:
        store_name: Name of the store
        results: Optimization results
        processing_times: Optional processing time information
    """
    # Handle different result types
    if isinstance(results, dict):
        # Full optimization results
        status = "✅ Completed" if results else "❌ Failed"
        zones_optimized = len(results) if results else 0

        # Check if results contain EnvPowerModels (from training) or schedules (from optimization)
        if results and hasattr(next(iter(results.values())), "temp_model"):
            # Training results - EnvPowerModels objects
            total_hours = "N/A (Training)"
        else:
            # Optimization results - schedule objects with length
            total_hours = (
                sum(len(zone_schedule) for zone_schedule in results.values())
                if results
                else 0
            )
    elif isinstance(results, bool):
        # Boolean results (preprocessing, training, etc.)
        status = "✅ Completed" if results else "❌ Failed"
        zones_optimized = "N/A"
        total_hours = "N/A"
    elif hasattr(results, "shape"):  # DataFrame or similar
        # DataFrame results (aggregation, etc.)
        status = (
            "✅ Completed" if results is not None and not results.empty else "❌ Failed"
        )
        zones_optimized = f"{results.shape[0]} rows" if results is not None else "N/A"
        total_hours = f"{results.shape[1]} cols" if results is not None else "N/A"
    else:
        # Other types
        status = "✅ Completed" if results else "❌ Failed"
        zones_optimized = "N/A"
        total_hours = "N/A"

    # Determine appropriate title and headers based on result type
    if (
        isinstance(results, dict)
        and results
        and hasattr(next(iter(results.values())), "temp_model")
    ):
        # Training results
        title = "🤖 Training Summary"
        headers = {
            "Store": store_name,
            "Status": status,
            "Models Trained": zones_optimized,
            "Details": total_hours,
        }
    else:
        # Optimization or other results
        title = "🎯 Optimization Summary"
        headers = {
            "Store": store_name,
            "Status": status,
            "Zones Optimized": zones_optimized,
            "Total Hours": total_hours,
        }

    print_table(
        data=headers,
        title=title,
        column_widths=[20, 25, 18, 15],
    )

    if isinstance(results, dict) and results:
        # Zone breakdown (only for optimization results, not training results)
        if not hasattr(next(iter(results.values())), "temp_model"):
            zone_data = []
            for zone_name, zone_schedule in results.items():
                total_power = sum(
                    settings.get("pred_power", 0) for settings in zone_schedule.values()
                )
                avg_temp = sum(
                    settings.get("pred_temp", 25) for settings in zone_schedule.values()
                ) / len(zone_schedule)
                zone_data.append(
                    [
                        zone_name,
                        len(zone_schedule),
                        f"{total_power:.1f} kWh",
                        f"{avg_temp:.1f}°C",
                    ]
                )

            print_table(
                data=zone_data,
                title="📊 Zone Details",
                headers=["Zone", "Hours", "Total Power", "Avg Temp"],
                column_widths=[15, 10, 15, 12],
            )

    if processing_times:
        # Processing times
        time_data = []
        for process_name, duration in processing_times.items():
            time_data.append(
                [
                    process_name,
                    f"{duration:.2f}s",
                    f"{(duration/sum(processing_times.values())*100):.1f}%",
                ]
            )

        print_table(
            data=time_data,
            title="⏱️ Processing Times",
            headers=["Process", "Duration", "Percentage"],
            column_widths=[20, 12, 12],
        )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="エアコン最適化システム - 実行スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実行例:
  uv run run_optimization.py --preprocess-only
  uv run run_optimization.py --aggregate-only
  uv run run_optimization.py --train-only
  uv run run_optimization.py --optimize-only
  uv run run_optimization.py  # フルパイプライン実行
  uv run run_optimization.py --start-date 2024-01-01 --end-date 2024-01-02
  uv run run_optimization.py --store Clea --skip-visualization
        """,
    )

    # Store selection
    parser.add_argument(
        "--store", type=str, default="Clea", help="対象ストア名 (デフォルト: Clea)"
    )

    # Execution mode flags (can be combined)
    parser.add_argument("--preprocess-only", action="store_true", help="前処理のみ実行")
    parser.add_argument("--aggregate-only", action="store_true", help="集約のみ実行")
    parser.add_argument("--train-only", action="store_true", help="モデル学習のみ実行")
    parser.add_argument("--optimize-only", action="store_true", help="最適化のみ実行")

    # Date range parameters
    parser.add_argument("--start-date", type=str, help="最適化開始日 (YYYY-MM-DD形式)")
    parser.add_argument("--end-date", type=str, help="最適化終了日 (YYYY-MM-DD形式)")

    # Visualization
    parser.add_argument(
        "--skip-visualization", action="store_true", help="可視化をスキップ"
    )

    return parser.parse_args()


def run_optimization_for_store(
    store_name,
    execution_mode: str = "full",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    skip_visualization: bool = False,
):
    """
    指定されたストアの最適化を実行

    Args:
        store_name (str): 対象ストア名
        execution_mode (str): 実行モード ("full", "preprocess", "aggregate", "train", "optimize")
        start_date (str): 最適化開始日
        end_date (str): 最適化終了日
        skip_visualization (bool): 可視化をスキップするかどうか
    """
    print(f"🚀 {store_name}の最適化パイプライン開始 (モード: {execution_mode})")

    # 最適化システムの初期化
    enable_preprocessing = execution_mode in ["full", "preprocess"]
    skip_aggregation = execution_mode in ["train", "optimize"]

    optimizer = AirconOptimizer(
        store_name,
        enable_preprocessing=enable_preprocessing,
        skip_aggregation=skip_aggregation,
    )

    # Hardcoded values (not configurable via command line)
    temperature_std_multiplier = 5.0
    power_std_multiplier = 5.0
    weather_api_key = WEATHER_API_KEY
    coordinates = None  # Will use master data coordinates
    freq = "1H"  # Default frequency

    # 実行モードに応じた処理
    if execution_mode == "preprocess":
        print("📊 前処理のみ実行")
        results = optimizer.run_preprocessing_only(
            weather_api_key=weather_api_key,
            coordinates=coordinates,
            temperature_std_multiplier=temperature_std_multiplier,
            power_std_multiplier=power_std_multiplier,
        )
    elif execution_mode == "aggregate":
        print("🔄 集約のみ実行")
        results = optimizer.run_aggregation_only(
            start_date=start_date,
            end_date=end_date,
            weather_api_key=weather_api_key,
            coordinates=coordinates,
            freq=freq,
        )
    elif execution_mode == "train":
        print("🤖 モデル学習のみ実行")
        results = optimizer.run_training_only()
    elif execution_mode == "optimize":
        print("⚡ 最適化のみ実行")
        results = optimizer.run_optimization_only(
            start_date=start_date,
            end_date=end_date,
            weather_api_key=weather_api_key,
            coordinates=coordinates,
            freq=freq,
        )
    else:  # full
        print("🔄 フルパイプライン実行")
        results = optimizer.run(
            weather_api_key=weather_api_key,
            coordinates=coordinates,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            temperature_std_multiplier=temperature_std_multiplier,
            power_std_multiplier=power_std_multiplier,
        )

    # Check if results indicate success (handle different types)
    success = False
    if isinstance(results, dict):
        success = bool(results)
    elif isinstance(results, bool):
        success = results
    elif hasattr(results, "shape"):  # DataFrame or similar
        success = results is not None and not results.empty
    else:
        success = bool(results)

    if success:
        # Print optimization summary in table format
        print_optimization_summary(store_name, results)

        # Print output files in table format
        output_files = [
            [
                "Control Schedule",
                f"data/04_OutputData/{store_name}/control_type_schedule.csv",
            ],
            ["Unit Schedule", f"data/04_OutputData/{store_name}/unit_schedule.csv"],
        ]

        print_table(
            data=output_files,
            title="📁 Output Files",
            headers=["File Type", "Path"],
            column_widths=[20, 50],
        )

        # 可視化の実行（スキップフラグがFalseの場合のみ）
        if not skip_visualization and execution_mode in ["full", "optimize"]:
            print(f"\n📊 {store_name}の結果可視化を開始...")
            try:
                # 出力をリセットしてから全レポート生成
                reset_outputs(store_name)
                stats_df = None
                try:
                    generate_all_reports(store_name)
                except Exception as re:
                    print(f"⚠️ レポート生成でエラー: {re}")

                print(f"✅ {store_name}の可視化が完了しました")

                # Print visualization files in table format
                viz_files = [
                    ["Zone Analysis", "analysis/output/*_analysis.html (各ゾーン分析)"],
                    [
                        "Summary Analysis",
                        "analysis/output/summary_analysis.html (全体サマリー)",
                    ],
                    [
                        "Statistics",
                        "analysis/output/summary_statistics.csv (統計データ)",
                    ],
                ]

                print_table(
                    data=viz_files,
                    title="📁 Visualization Files",
                    headers=["File Type", "Path"],
                    column_widths=[20, 50],
                )

                if stats_df is not None:
                    print_table(data=stats_df, title="📊 Statistics Summary")

            except Exception as e:
                print(f"⚠️ 可視化でエラーが発生しました: {e}")
                print("最適化結果は正常に生成されています")
        elif skip_visualization:
            print("⏭️ 可視化をスキップしました")

        return success
    else:
        print_optimization_summary(store_name, results)
        return success


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
    if args.train_only:
        execution_modes.append("train")
    if args.optimize_only:
        execution_modes.append("optimize")

    if not execution_modes:
        execution_mode = "full"
    elif len(execution_modes) == 1:
        execution_mode = execution_modes[0]
    else:
        # Multiple modes specified - execute them in sequence
        print(f"🔄 Multiple execution modes specified: {execution_modes}")
        print(f"🔄 Will execute them in sequence")

        # Execute each mode in sequence
        for i, mode in enumerate(execution_modes):
            print(f"\n{'='*70}")
            print(f"🏢 {args.store} - Step {i+1}/{len(execution_modes)}: {mode}")
            print(f"{'='*70}")

            success = run_optimization_for_store(
                store_name=args.store,
                execution_mode=mode,
                start_date=args.start_date,
                end_date=args.end_date,
                skip_visualization=args.skip_visualization,
            )

            if not success:
                print(f"❌ Step {i+1} ({mode}) failed. Stopping execution.")
                return False

        print(
            f"\n✅ All {len(execution_modes)} execution modes completed successfully!"
        )
        return True

    # Single mode execution
    print(f"\n{'='*70}")
    print(f"🏢 {args.store} の最適化開始 (モード: {execution_mode})")
    print(f"{'='*70}")

    # 最適化の実行
    success = run_optimization_for_store(
        store_name=args.store,
        execution_mode=execution_mode,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_visualization=args.skip_visualization,
    )

    # 実行結果の表示
    execution_results = [
        [
            args.store,
            "✅ Completed" if success else "❌ Failed",
            "Success" if success else "Error",
        ]
    ]

    print_table(
        data=execution_results,
        title="🎯 Final Execution Summary",
        headers=["Store", "Status", "Result"],
        column_widths=[15, 15, 10],
    )

    # 終了コードの設定
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
