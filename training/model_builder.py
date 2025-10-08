import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from config.config_train import BASE_FEATURES, POWER_FEATURE_COLS, TEMP_FEATURE_COLS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization.helper_functions import print_model_evaluation_table


# =============================
# STEP2: 予測モデル（制御エリア別）
# =============================
@dataclass
class EnvPowerModels:
    temp_model: Any
    hum_model: Optional[Any]
    power_model: Any
    multi_output_model: Optional[Any]  # マルチアウトプットモデル
    feature_cols: List[str]


class ModelBuilder:
    def __init__(self, store_name: str):
        self.store_name = store_name

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _shap_on_pipeline(
        pipeline: Any, X: pd.DataFrame, out_png: str, title: str
    ) -> None:
        try:
            # Handle both Pipeline and direct model objects
            if isinstance(pipeline, Pipeline):
                model = pipeline.named_steps.get("model")
                if model is None:
                    print(f"[ModelBuilder] No model found in pipeline")
                    return
                # Use pipeline's scaler if available
                scaler = pipeline.named_steps.get("scaler")
                data_for_model = X
                if scaler is not None:
                    try:
                        data_for_model = scaler.transform(X)
                    except Exception as e:
                        print(f"[ModelBuilder] Scaler transform failed: {e}")
                        data_for_model = X
            else:
                # Direct model object (e.g., XGBRegressor)
                model = pipeline
                data_for_model = X
            
            # サンプリング（重すぎ防止）
            Xs = X.sample(min(len(X), 2000), random_state=42) if len(X) > 0 else X
            if Xs is None or Xs.empty:
                print(f"[ModelBuilder] Empty or None data for SHAP")
                return
            
            # Use sampled data for model input
            if isinstance(pipeline, Pipeline) and scaler is not None:
                try:
                    data_for_model = scaler.transform(Xs)
                except Exception:
                    data_for_model = Xs
            else:
                data_for_model = Xs
            
            print(f"[ModelBuilder] Generating SHAP plot for {title} with {len(Xs)} samples")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_for_model)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                data_for_model,
                feature_names=Xs.columns,
                show=False,
                plot_type="bar",
            )
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[ModelBuilder] SHAP plot saved to: {out_png}")
        except Exception as e:
            print(f"[ModelBuilder] SHAP plot failed: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _split_xy(df: pd.DataFrame, target: str, feature_cols: List[str]):
        """特徴量と単一ターゲットを返す。欠損を含む行は除外する。"""
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = df[target]
        combined = pd.concat([X, y], axis=1)
        combined = combined.dropna()
        X_clean = combined[feature_cols].astype(float)
        y_clean = combined[target].astype(float)
        
        # Preserve the original datetime index
        if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
            # Use the index from the cleaned combined dataframe
            X_clean.index = combined.index
            y_clean.index = combined.index
        elif 'Datetime' in df.columns:
            # If Datetime is a column, use it as index
            datetime_values = df.loc[combined.index, 'Datetime']
            X_clean.index = pd.to_datetime(datetime_values)
            y_clean.index = pd.to_datetime(datetime_values)
        
        return X_clean, y_clean

    @staticmethod
    def _split_multi(df: pd.DataFrame, targets: List[str], feature_cols: List[str]):
        """特徴量と複数ターゲットを返す。欠損を含む行は除外する。"""
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        Y = df[targets]
        combined = pd.concat([X, Y], axis=1)
        combined = combined.dropna()
        X_clean = combined[feature_cols].astype(float)
        Y_clean = combined[targets].astype(float)
        
        # Preserve the original datetime index
        if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
            # Use the index from the cleaned combined dataframe
            X_clean.index = combined.index
            Y_clean.index = combined.index
        elif 'Datetime' in df.columns:
            # If Datetime is a column, use it as index
            datetime_values = df.loc[combined.index, 'Datetime']
            X_clean.index = pd.to_datetime(datetime_values)
            Y_clean.index = pd.to_datetime(datetime_values)
        
        return X_clean, Y_clean

    @staticmethod
    def _compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        """追加特徴量を作成（ON連続長, Hour×DayOfWeek など）"""
        out = df.copy()
        # HourOfWeek = DayOfWeek*24 + Hour
        if all(c in out.columns for c in ["DayOfWeek", "Hour"]):
            out["HourOfWeek"] = out["DayOfWeek"] * 24 + out["Hour"]
        # ON/OFF連続長（A/C ON/OFFが存在する場合）
        if "A/C ON/OFF" in out.columns and "Datetime" in out.columns:
            out = out.sort_values("Datetime")
            onoff = out["A/C ON/OFF"].fillna(0).astype(int).values
            run = 0
            runs = []
            for v in onoff:
                if v > 0:
                    run += 1
                else:
                    run = 0
                runs.append(run)
            out["OnRunLength"] = runs
        return out

    def train_by_zone(
        self, area_df: pd.DataFrame, master: dict
    ) -> Dict[str, EnvPowerModels]:
        print(
            f"[ModelBuilder] Starting train_by_zone. "
            f"Input shape: {area_df.shape if area_df is not None else 'None'}"
        )
        if area_df is None or area_df.empty:
            print("[ModelBuilder] Area data is None or empty")
            return {}
        models: Dict[str, EnvPowerModels] = {}
        zones = sorted(area_df["zone"].dropna().unique().tolist())
        print(f"[ModelBuilder] Found zones: {zones}")
        for z in zones:
            sub = area_df[area_df["zone"] == z].copy()
            if "Datetime" in sub.columns:
                sub = self._compute_additional_features(sub)
            print(f"[ModelBuilder] Zone {z}: {len(sub)} records")
            if len(sub) < 20:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient data: {len(sub)} < 20)"
                )
                continue

            # Temperature and humidity models use TEMP_FEATURE_COLS
            temp_feats = [c for c in TEMP_FEATURE_COLS if c in sub.columns]
            if temp_feats:
                nonnull_ratio = sub[temp_feats].notna().mean()
                # Keep important features even if they have some missing values (>= 30% non-null)
                # Critical features like A/C Fan Speed should always be included
                critical_features = [
                    "A/C Fan Speed",
                    "A/C Status",  # Combined ON/OFF and Mode
                    # "A/C Mode",  # Now using A/C Status
                    # "A/C ON/OFF",  # Now using A/C Status
                    "A/C Set Temperature",
                ]
                temp_feats = [
                    c
                    for c in temp_feats
                    if (c in critical_features and nonnull_ratio.get(c, 0.0) >= 0.3)
                    or nonnull_ratio.get(c, 0.0) >= 0.9
                ]
            if len(temp_feats) < 5:
                print(
                    f"[ModelBuilder] Zone {z}: insufficient temp features {temp_feats}"
                )

            # 温度
            X_t, y_t = self._split_xy(sub, "Indoor Temp.", temp_feats)
            if len(X_t) < 5:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient temp samples after dropping NaNs: {len(X_t)} < 5)"
                )
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X_t, y_t, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Print train/test date ranges
            print(f"[ModelBuilder] Zone {z} - Train: {pd.to_datetime(Xtr.index.min())} to {pd.to_datetime(Xtr.index.max())} | Test: {pd.to_datetime(Xte.index.min())} to {pd.to_datetime(Xte.index.max())}")
            temp_model = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=400,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            temp_model.fit(Xtr, ytr)
            # 湿度
            hum_model = None
            if "室内湿度" in sub.columns and sub["室内湿度"].notna().sum() > 10:
                X_h, y_h = self._split_xy(sub, "室内湿度", temp_feats)
                Xtrh, Xteh, ytrh, yteh = train_test_split(
                    X_h, y_h, test_size=0.2, random_state=42, shuffle=False
                )
                
                # Print humidity train/test date ranges
                print(f"[ModelBuilder] Zone {z} - Humidity Train: {pd.to_datetime(Xtrh.index.min())} to {pd.to_datetime(Xtrh.index.max())} | Test: {pd.to_datetime(Xteh.index.min())} to {pd.to_datetime(Xteh.index.max())}") 
                hum_model = Pipeline(
                    steps=[
                        ("scaler", MinMaxScaler()),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=400,
                                max_depth=6,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
                hum_model.fit(Xtrh, ytrh)
            # 電力
            print(f"[ModelBuilder] Zone {z}: Checking adjusted_power column...")
            if "adjusted_power" not in sub.columns:
                print(
                    f"[ModelBuilder] Zone {z}: No adjusted_power column found. Available columns: {list(sub.columns)}"
                )
                continue
            power_count = sub["adjusted_power"].notna().sum()
            print(
                f"[ModelBuilder] Zone {z}: adjusted_power non-null values: {power_count}"
            )
            if power_count < 10:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient power data: {power_count} < 10)"
                )
                continue
            # Power model uses POWER_FEATURE_COLS
            power_feats = [c for c in POWER_FEATURE_COLS if c in sub.columns]
            print(
                f"[ModelBuilder] Zone {z}: Initial power features count: {len(power_feats)}"
            )
            if power_feats:
                nonnull_ratio = sub[power_feats].notna().mean()
                # Keep important features even if they have some missing values (>= 30% non-null)
                # Critical features like A/C Fan Speed should always be included
                critical_features = [
                    "A/C Fan Speed",
                    "A/C Status",  # Combined ON/OFF and Mode
                    # "A/C Mode",  # Now using A/C Status
                    # "A/C ON/OFF",  # Now using A/C Status
                    "A/C Set Temperature",
                ]
                power_feats = [
                    c
                    for c in power_feats
                    if (c in critical_features and nonnull_ratio.get(c, 0.0) >= 0.3)
                    or nonnull_ratio.get(c, 0.0) >= 0.9
                ]
            print(
                f"[ModelBuilder] Zone {z}: Final power features count: {len(power_feats)}"
            )
            print(
                f"[ModelBuilder] Zone {z}: Power features (first 10): {power_feats[:min(10, len(power_feats))]}"
            )
            if len(power_feats) < 5:
                print(
                    f"[ModelBuilder] Zone {z}: insufficient power features {power_feats}"
                )

            X_p, y_p = self._split_xy(sub, "adjusted_power", power_feats)
            if len(X_p) < 5:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient power samples after dropping NaNs: {len(X_p)} < 5)"
                )
                continue

            # Debug: Print target statistics
            print(f"[ModelBuilder] Zone {z}: Power target (y_p) statistics:")
            print(f"  - Min: {y_p.min():.2f} W")
            print(f"  - Max: {y_p.max():.2f} W")
            print(f"  - Mean: {y_p.mean():.2f} W")
            print(f"  - Std: {y_p.std():.2f} W")
            print(f"  - Shape: {y_p.shape}")

            sample_weight_series = None
            if "A/C ON/OFF" in sub.columns:
                try:
                    onoff_aligned = sub.loc[X_p.index, "A/C ON/OFF"].fillna(0)
                    sample_weight_series = pd.Series(
                        np.where(onoff_aligned > 0, 1.0, 0.2), index=X_p.index
                    )
                except Exception:
                    sample_weight_series = None
            if sample_weight_series is not None:
                Xtrp, Xtep, ytrp, ytep, wtrp, wtep = train_test_split(
                    X_p, y_p, sample_weight_series, test_size=0.2, random_state=42, shuffle=False
                )
            else:
                Xtrp, Xtep, ytrp, ytep = train_test_split(
                    X_p, y_p, test_size=0.2, random_state=42, shuffle=False
                )
                wtrp = None
            
            # Print power train/test date ranges
            print(f"[ModelBuilder] Zone {z} - Power Train: {pd.to_datetime(Xtrp.index.min())} to {pd.to_datetime(Xtrp.index.max())} | Test: {pd.to_datetime(Xtep.index.min())} to {pd.to_datetime(Xtep.index.max())}")
            # Remove MinMaxScaler - it's causing constant predictions
            power_model = XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",  # Changed to squarederror for better stability
            )

            if wtrp is not None:
                power_model.fit(Xtrp, ytrp, sample_weight=wtrp)
            else:
                power_model.fit(Xtrp, ytrp)

            # Debug: Check predictions on training set
            train_pred = power_model.predict(Xtrp)
            print(f"[ModelBuilder] Zone {z}: Training predictions:")
            print(f"  - Min: {train_pred.min():.2f} W")
            print(f"  - Max: {train_pred.max():.2f} W")
            print(f"  - Mean: {train_pred.mean():.2f} W")
            print(f"  - Training targets min/max: {ytrp.min():.2f}/{ytrp.max():.2f} W")

            # マルチ出力
            multi_output_model = None
            try:
                multi_targets = ["Indoor Temp.", "adjusted_power"]
                if all(t in sub.columns for t in multi_targets):
                    Xm, Ym = self._split_multi(sub, multi_targets, temp_feats)
                    if len(Xm) >= 20:
                        Xm_tr, Xm_te, Ym_tr, Ym_te = train_test_split(
                            Xm, Ym, test_size=0.2, random_state=42, shuffle=False
                        )
                        
                        # Print multi-output train/test date ranges
                        print(f"[ModelBuilder] Zone {z} - Multi-output Train: {pd.to_datetime(Xm_tr.index.min())} to {pd.to_datetime(Xm_tr.index.max())} | Test: {pd.to_datetime(Xm_te.index.min())} to {pd.to_datetime(Xm_te.index.max())}")
                        # Remove MinMaxScaler - XGBoost handles features internally
                        base_xgb = XGBRegressor(
                            n_estimators=500,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.9,
                            random_state=42,
                            n_jobs=-1,
                            objective="reg:squarederror",  # Changed to squarederror for better stability
                        )
                        multi_output_model = MultiOutputRegressor(base_xgb)
                        multi_output_model.fit(Xm_tr, Ym_tr)
                        Ym_hat = pd.DataFrame(
                            multi_output_model.predict(Xm_te),
                            index=Ym_te.index,
                            columns=multi_targets,
                        )
                        temp_mae = mean_absolute_error(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        power_mae = mean_absolute_error(
                            Ym_te["adjusted_power"] / 1000.0,
                            Ym_hat["adjusted_power"] / 1000.0,
                        )
                        temp_mape = mean_absolute_percentage_error(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        nonzero_mask = (Ym_te["adjusted_power"] / 1000.0) != 0
                        power_mape = (
                            mean_absolute_percentage_error(
                                (Ym_te.loc[nonzero_mask, "adjusted_power"] / 1000.0),
                                (Ym_hat.loc[nonzero_mask, "adjusted_power"] / 1000.0),
                            )
                            if nonzero_mask.any()
                            else np.nan
                        )
                        temp_r2 = r2_score(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        power_r2 = r2_score(
                            Ym_te["adjusted_power"], Ym_hat["adjusted_power"]
                        )
                        # Print multi-output model evaluation in table format
                        print(f"\n{'='*60}")
                        print(f"📊 Multi-Output Model Evaluation - {z}")
                        print(f"{'='*60}")
                        print(f"{'Metric':<15} | {'Temperature':<15} | {'Power':<15}")
                        print(f"{'-'*15}-+-{'-'*15}-+-{'-'*15}")
                        print(
                            f"{'MAE':<15} | {temp_mae:<15.2f} | {power_mae:<15.3f} kW"
                        )
                        print(
                            f"{'MAPE':<15} | {temp_mape*100:<15.1f}% | {(power_mape*100 if not np.isnan(power_mape) else np.nan):<15.1f}%"
                        )
                        print(f"{'R2':<15} | {temp_r2:<15.3f} | {power_r2:<15.3f}")
                        print(f"{'='*60}")
            except Exception as me:
                print(f"[ModelBuilder] Multi-output model error for zone {z}: {me}")
            # 評価
            y_hat_t = temp_model.predict(Xte)
            temp_mae = mean_absolute_error(yte, y_hat_t)
            temp_mape = mean_absolute_percentage_error(yte, y_hat_t)

            # Calculate temperature metrics
            temp_metrics = {
                "MAE": temp_mae,
                "MAPE": temp_mape * 100,
                "R2": r2_score(yte, y_hat_t),
            }

            # Calculate power metrics
            y_hat_p = power_model.predict(Xtep)
            ytep_kw = ytep / 1000.0
            yhat_kw = pd.Series(y_hat_p, index=ytep.index) / 1000.0
            nonzero_mask_p = ytep_kw != 0
            power_mape = (
                mean_absolute_percentage_error(
                    ytep_kw[nonzero_mask_p], yhat_kw[nonzero_mask_p]
                )
                if nonzero_mask_p.any()
                else np.nan
            )
            power_metrics = {
                "MAE": mean_absolute_error(ytep_kw, yhat_kw),
                "MAPE": power_mape * 100 if not np.isnan(power_mape) else np.nan,
                "R2": r2_score(ytep, y_hat_p),
            }

            # Print combined model evaluation table
            print_model_evaluation_table(z, temp_metrics, power_metrics)
            # SHAP
            out_dir = os.path.join("analysis", "output", self.store_name, z)
            ModelBuilder._ensure_dir(out_dir)
            try:
                self._shap_on_pipeline(
                    temp_model,
                    Xte,
                    os.path.join(out_dir, "shap_temp.png"),
                    f"SHAP - Temp ({z})",
                )
            except Exception as e:
                print(f"[ModelBuilder] SHAP temp failed ({z}): {e}")
            try:
                self._shap_on_pipeline(
                    (
                        power_model.pipeline
                        if hasattr(power_model, "pipeline")
                        else power_model
                    ),
                    Xtep,
                    os.path.join(out_dir, "shap_power.png"),
                    f"SHAP - Power ({z})",
                )
            except Exception as e:
                print(f"[ModelBuilder] SHAP power failed ({z}): {e}")
            models[z] = EnvPowerModels(
                temp_model=temp_model,
                hum_model=hum_model,
                power_model=power_model,
                multi_output_model=multi_output_model,
                feature_cols=temp_feats,  # Use temp_feats for compatibility
            )
        # 保存
        from config.utils import get_data_path

        mdir = os.path.join(get_data_path("models_path"), self.store_name)
        os.makedirs(mdir, exist_ok=True)
        for z, pack in models.items():
            joblib.dump(
                {
                    "temp_model": pack.temp_model,
                    "hum_model": pack.hum_model,
                    "power_model": pack.power_model,
                    "multi_output_model": pack.multi_output_model,
                    "feature_cols": pack.feature_cols,
                },
                os.path.join(mdir, f"models_{z}.pkl"),
            )

        # Save validation results with input features and predictions
        self._save_validation_results(area_df, models)

        return models

    def _save_validation_results(self, area_df: pd.DataFrame, models: Dict[str, Any]):
        """
        Save input features and predictions to valid_results.csv for each area

        Args:
            area_df: Area data with features
            models: Trained models for each zone
        """
        from config.utils import get_data_path

        # Create output directory
        output_dir = os.path.join(get_data_path("valid_results_path"), self.store_name)
        os.makedirs(output_dir, exist_ok=True)

        # Get all zones
        zones = sorted(area_df["zone"].dropna().unique().tolist())

        for zone in zones:
            print(f"[ModelBuilder] Saving validation results for zone: {zone}")

            # Get zone data
            zone_data = area_df[area_df["zone"] == zone].copy()

            if zone not in models:
                print(f"[ModelBuilder] No model found for zone {zone}, skipping...")
                continue

            model_pack = models[zone]

            # Prepare features for prediction
            temp_feats = [c for c in TEMP_FEATURE_COLS if c in zone_data.columns]
            power_feats = [c for c in POWER_FEATURE_COLS if c in zone_data.columns]

            # Create results dataframe starting with input features
            results_df = zone_data.copy()

            # Add temperature predictions
            if model_pack.temp_model is not None and len(temp_feats) > 0:
                try:
                    # Filter features and handle missing values
                    X_temp = zone_data[temp_feats].fillna(0)
                    temp_pred = model_pack.temp_model.predict(X_temp)
                    results_df[f"{zone}_temp_pred"] = temp_pred
                    print(f"[ModelBuilder] Added temperature predictions for {zone}")
                except Exception as e:
                    print(
                        f"[ModelBuilder] Temperature prediction failed for {zone}: {e}"
                    )
                    results_df[f"{zone}_temp_pred"] = np.nan

            # Add power predictions
            if model_pack.power_model is not None and len(power_feats) > 0:
                try:
                    # Filter features and handle missing values
                    X_power = zone_data[power_feats].fillna(0)
                    power_pred = model_pack.power_model.predict(X_power)

                    # Clip negative predictions to 0 (power cannot be negative)
                    power_pred = np.clip(power_pred, 0.0, None)

                    # Debug: Print power prediction statistics
                    print(f"[ModelBuilder] Power predictions for {zone}:")
                    print(f"  - Min: {power_pred.min():.3f}")
                    print(f"  - Max: {power_pred.max():.3f}")
                    print(f"  - Mean: {power_pred.mean():.3f}")
                    print(f"  - Std: {power_pred.std():.3f}")
                    print(f"  - Unique values: {len(np.unique(power_pred))}")

                    results_df[f"{zone}_power_pred"] = power_pred
                    print(f"[ModelBuilder] Added power predictions for {zone}")
                except Exception as e:
                    print(f"[ModelBuilder] Power prediction failed for {zone}: {e}")
                    results_df[f"{zone}_power_pred"] = np.nan

            # Save to CSV with proper number formatting
            output_file = os.path.join(output_dir, f"valid_results_{zone}.csv")

            # Format power prediction columns to avoid scientific notation
            for col in results_df.columns:
                if col.endswith("_power_pred"):
                    results_df[col] = results_df[col].round(
                        3
                    )  # Round to 3 decimal places

            results_df.to_csv(output_file, index=False, float_format="%.3f")
            print(f"[ModelBuilder] Saved validation results to: {output_file}")

        print(f"[ModelBuilder] Validation results saved for {len(zones)} zones")

