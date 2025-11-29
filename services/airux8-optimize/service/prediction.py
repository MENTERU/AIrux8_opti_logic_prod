from typing import Optional

import pandas as pd

from deep_reinforcement_learning.environment.prediction.model import (
    dump_residual_model,
    fit_predict_eval_xgb_residual_only,
)
from deep_reinforcement_learning.environment.prediction.transform_input_data import (
    build_features_residualized,
    build_targets_residualized,
)
from deep_reinforcement_learning.environment.prediction.visualization import (
    plot_feature_importance_for_targets_compat,
    plot_residual_metrics,
)
from input_info.crea_building_information import CreaBuilding


class AreaHVACModelManager:
    def __init__(self):
        self.building_info: CreaBuilding = CreaBuilding()
        self.origin_data: Optional[pd.DataFrame] = None
        self.original_data_columns = None

        # XGBoost の共通パラメータ
        self.xgb_params = dict(
            n_estimators=4000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=12,
            gamma=1.0,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=2.0,
            eval_metric="rmse",
        )

    def set_origin_data(self, df: pd.DataFrame):
        self.original_data_columns = df.columns
        self.origin_data = df

    # ---- Y を「室内温度」と「消費電力」に分割するヘルパ ----
    @staticmethod
    def _split_targets_temp_and_kwh(Y_features: pd.DataFrame):
        """
        Y_features の列名から
          - 室内温度ターゲット
          - 室外機消費電力ターゲット
        を分割する。

        ※列名の規則に応じてフィルタ条件は調整してください。
        ここでは例として:
          - "Indoor Temp." を含む列 → 温度
          - "total_kwh" を含む列 → 消費電力
        """
        temp_cols = [
            c for c in Y_features.columns if "Indoor Temp." in c or "indoor_temp" in c
        ]
        kwh_cols = [c for c in Y_features.columns if "total_kwh" in c]

        Y_temp = Y_features[temp_cols].copy() if temp_cols else None
        Y_kwh = Y_features[kwh_cols].copy() if kwh_cols else None

        return Y_temp, Y_kwh

    def learn_two_stage(self, area_unit_names: list, visualize: bool = False):
        """
        Step1: 室内機の温度を予測するモデルを学習（X から「lag1h__」始まりのカラムを除外）
        Step2: 室外機の消費電力を予測するモデルを学習

        各エリアごとに、
          models/{area}__temp.joblib
          models/{area}__kwh.joblib
        として res（モデル一式）を保存する。
        """
        if self.origin_data is None:
            raise ValueError(
                "origin_data がセットされていません。set_origin_data(...) を先に呼んでください。"
            )

        for unit_name in area_unit_names:
            print(f"{unit_name} の2段階学習中...")

            # --- このエリアに対応する列だけ抽出 ---
            unit_columns = CreaBuilding.get_columns_by_area_units(
                self.original_data_columns, unit_name
            )
            df_area = self.origin_data[
                unit_columns + self.building_info.weather_colimns
            ]

            # --- 特徴量・ターゲットを構築（残差化ロジックは既存関数に任せる） ---
            X_features = build_features_residualized(
                df_area,
                include_weather_raw=True,
            )
            Y_features = build_targets_residualized(df_area)

            # --- Y を 温度ターゲット / kWhターゲット に分割 ---
            Y_temp, Y_kwh = self._split_targets_temp_and_kwh(Y_features)

            # ========== Step1: 室内温度モデル ==========
            if Y_temp is not None and not Y_temp.empty:
                # lag1h__ から始まる特徴量を除外
                X_step1 = X_features.loc[
                    :, ~X_features.columns.str.startswith("lag1h__")
                ]

                print(
                    f"  Step1: 温度モデル学習 (X:{X_step1.shape}, Y_temp:{Y_temp.shape})"
                )

                res_temp = fit_predict_eval_xgb_residual_only(
                    X_step1,
                    Y_temp,
                    xgb_params=self.xgb_params,
                    split_mode="last_months",
                    test_last_months=1,
                    purge_days=0,
                    early_stopping_rounds=50,
                    verbose=False,
                )

                # 可視化（任意）
                if visualize:
                    print("    → 温度モデルの残差メトリクス / 重要度プロット")
                    plot_residual_metrics(res_temp, sort_by="R2", figsize=(16, 6))
                    plot_feature_importance_for_targets_compat(
                        res_temp,
                        X_step1,
                        targets=Y_temp.columns.tolist(),
                        importance_type="weight",
                        top_n=10,
                    )

                # モデル保存
                dump_residual_model(res_temp, f"models/{unit_name}__temp.joblib")
            else:
                print(
                    f"  ⚠ {unit_name}: 室内温度ターゲットが見つからなかったため Step1 をスキップします。"
                )

            # ========== Step2: 消費電力モデル ==========
            if Y_kwh is not None and not Y_kwh.empty:
                print(
                    f"  Step2: 消費電力モデル学習 (X:{X_features.shape}, Y_kwh:{Y_kwh.shape})"
                )

                res_kwh = fit_predict_eval_xgb_residual_only(
                    X_features,
                    Y_kwh,
                    xgb_params=self.xgb_params,
                    split_mode="last_months",
                    test_last_months=1,
                    purge_days=0,
                    early_stopping_rounds=50,
                    verbose=False,
                )

                if visualize:
                    print("    → 消費電力モデルの残差メトリクス / 重要度プロット")
                    plot_residual_metrics(res_kwh, sort_by="R2", figsize=(16, 6))
                    plot_feature_importance_for_targets_compat(
                        res_kwh,
                        X_features,
                        targets=Y_kwh.columns.tolist(),
                        importance_type="weight",
                        top_n=10,
                    )

                dump_residual_model(res_kwh, f"models/{unit_name}__kwh.joblib")
            else:
                print(
                    f"  ⚠ {unit_name}: total_kwh 系ターゲットが見つからなかったため Step2 をスキップします。"
                )

    @property
    def area_info(self):
        return self.building_info.AREA_UNITS
