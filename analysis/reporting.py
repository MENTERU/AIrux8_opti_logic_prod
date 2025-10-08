# -*- coding: utf-8 -*-
"""
レポーティング統合（ダッシュボードのみ生成）
 - 出力リセット
 - 既存の可視化呼び出しを一本化
 - 拡張分析HTMLの生成は停止
"""

import os
import shutil

from analysis.dashboards import (
    create_historical_dashboard,
    create_plan_validation_dashboard,
)


def reset_outputs(store_name: str = "Clea") -> None:
    """分析/可視化の出力をリセット（SHAPファイルを保持）"""
    out_dir = "analysis/output"
    if os.path.isdir(out_dir):
        # Preserve SHAP PNG files before deletion
        shap_files_backup = {}
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                if file.startswith("shap_") and file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    try:
                        # Read and backup the file content
                        with open(file_path, 'rb') as f:
                            shap_files_backup[file_path] = f.read()
                        print(f"💾 Backing up SHAP file: {file_path}")
                    except Exception as e:
                        print(f"⚠️ Could not backup SHAP file {file_path}: {e}")
        
        # Delete the directory
        shutil.rmtree(out_dir)
        
        # Recreate directory structure
        os.makedirs(out_dir, exist_ok=True)
        
        # Restore SHAP files
        for file_path, file_content in shap_files_backup.items():
            try:
                # Recreate the directory structure
                shap_dir = os.path.dirname(file_path)
                os.makedirs(shap_dir, exist_ok=True)
                
                # Restore the file content
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                print(f"✅ Restored SHAP file: {file_path}")
            except Exception as e:
                print(f"⚠️ Could not restore SHAP file {file_path}: {e}")
        
        if shap_files_backup:
            print(f"🧹 出力をリセットしました: {out_dir} ({len(shap_files_backup)}個のSHAPファイルを保持)")
        else:
            print(f"🧹 出力をリセットしました: {out_dir}")
    else:
        os.makedirs(out_dir, exist_ok=True)
        print(f"📁 出力ディレクトリを作成しました: {out_dir}")


def generate_all_reports(store_name: str = "Clea"):
    """全レポート生成を統合呼び出し（ダッシュボードのみ）"""
    # 実績ダッシュボード（時別/日別）
    create_historical_dashboard(store_name, freq="H")
    create_historical_dashboard(store_name, freq="D")
    # 計画妥当性ダッシュボード
    create_plan_validation_dashboard(store_name, lookback_days=7)
    print("📦 全ダッシュボード生成が完了しました")
