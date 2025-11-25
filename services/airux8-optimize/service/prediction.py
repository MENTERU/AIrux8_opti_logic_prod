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
        self.origin_data = None
        self.original_data_columns = None
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
            eval_metric="rmse",  # ★ ここに入れる（fitには渡さない）
        )

    def set_origin_data(self, df: pd.DataFrame):
        self.original_data_columns = df.columns
        self.origin_data = df

    def learn(self, area_unit_names: list, visualize: bool = False):
        for unit_name in area_unit_names:
            print(unit_name, "の学習中...")
            unit_columns = CreaBuilding.get_columns_by_area_units(
                self.original_data_columns, unit_name
            )
            X_features = build_features_residualized(
                self.origin_data[unit_columns],
                include_weather_raw=True,
            )
            Y_features = build_targets_residualized(self.origin_data[unit_columns])
            res = fit_predict_eval_xgb_residual_only(
                X_features,
                Y_features,
                xgb_params=self.xgb_params,
                split_mode="last_months",
                test_last_months=1,
                purge_days=0,
                early_stopping_rounds=50,
                verbose=False,
            )
            if visualize:
                plot_residual_metrics(res, sort_by="R2", figsize=(16, 6))
                plot_feature_importance_for_targets_compat(
                    res,
                    X_features,
                    targets=Y_features.columns.tolist(),  # or ["targetA", "targetB"] で絞り込み
                    importance_type="weight",  # "weight", "cover", "total_gain", "total_cover" も可
                    top_n=10,
                )
            dump_residual_model(res, f"models/{unit_name}.joblib")

    @property
    def area_info(self):
        return self.building_info.AREA_UNITS
