# =============================================================================
# エアコン最適化システム - 実行サンプル
# =============================================================================

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
    print_table(
        data={
            "Store": store_name,
            "Status": "✅ Completed" if results else "❌ Failed",
            "Zones Optimized": len(results) if results else 0,
            "Total Hours": (
                sum(len(zone_schedule) for zone_schedule in results.values())
                if results
                else 0
            ),
        },
        title="🎯 Optimization Summary",
        column_widths=[20, 25, 18, 15],
    )

    if results:
        # Zone breakdown
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


def run_optimization_for_store(
    store_name,
    temperature_std_multiplier=5.0,
    power_std_multiplier=5.0,
    skip_aggregation=True,
):
    """
    指定されたストアの最適化を実行

    Args:
        store_name (str): 対象ストア名
        temperature_std_multiplier (float): 温度データの外れ値判定係数（デフォルト: 5.0）
        power_std_multiplier (float): 電力データの外れ値判定係数（デフォルト: 5.0）
        skip_aggregation (bool): 集約をスキップして既存のfeatures_processed_*.csvを直接読み込む（デフォルト: False）
    """
    print(f"🚀 {store_name}の最適化パイプライン開始")

    # 最適化システムの初期化（前処理をスキップ）
    enable_preprocessing = True  # 前処理を行うかどうか
    skip_aggregation = False  # 集約をスキップするかどうか
    optimizer = AirconOptimizer(
        store_name,
        enable_preprocessing=enable_preprocessing,
        skip_aggregation=skip_aggregation,
    )

    # フルパイプラインの実行（座標はマスタから自動取得）
    results = optimizer.run(
        weather_api_key=WEATHER_API_KEY,
        temperature_std_multiplier=temperature_std_multiplier,
        power_std_multiplier=power_std_multiplier,
        preference="energy",  # 電力優先で最適化
    )

    if results:
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

        # 可視化の実行
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
                ["Statistics", "analysis/output/summary_statistics.csv (統計データ)"],
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

        return True
    else:
        print_optimization_summary(store_name, results)
        return False


def main():
    """メイン実行関数"""
    # 対象ストアのリスト（Cleaのみ）
    target_stores = ["Clea"]

    # Store execution results
    execution_results = []

    # 各ストアの最適化を実行
    for store_name in target_stores:
        print(f"\n{'='*70}")
        print(f"🏢 {store_name} の最適化開始")
        print(f"{'='*70}")

        success = run_optimization_for_store(
            store_name=store_name,
            temperature_std_multiplier=5.0,
            power_std_multiplier=5.0,
        )

        # Store result for summary
        execution_results.append(
            [
                store_name,
                "✅ Completed" if success else "❌ Failed",
                "Success" if success else "Error",
            ]
        )

    # Print final summary
    print_table(
        data=execution_results,
        title="🎯 Final Execution Summary",
        headers=["Store", "Status", "Result"],
        column_widths=[15, 15, 10],
    )


if __name__ == "__main__":
    main()
