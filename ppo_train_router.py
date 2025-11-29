import pandas as pd

from deep_reinforcement_learning.environment.prediction.model import load_residual_model
from input_info.crea_building_information import CreaBuilding
from service.ppo_train import AircontrolPPOTrainConfig, AircontrolPPOTrainer


def extract_set_and_target_from_cols(unit_df, room_names):
    unit_temp_range_list = []
    target_temp_list = []
    for room in room_names:
        set_low = unit_df.loc[unit_df["環境予測区分"] == room, "設定温度下限"].iloc[0]
        set_high = unit_df.loc[unit_df["環境予測区分"] == room, "設定温度上限"].iloc[0]
        target = unit_df.loc[unit_df["環境予測区分"] == room, "目標室内温度"].iloc[0]
        unit_temp_range_list.append((set_low, set_high))
        target_temp_list.append(target)
    return unit_temp_range_list, target_temp_list


class Trainer:
    def __init__(self, train_data_path, train_weather_path, area_name):
        self.train_data_path = train_data_path
        self.train_weather_path = train_weather_path
        self.area_name = area_name

    def setup(self):
        hvac_master = pd.read_excel("./data/master/MASTER_Clea.xlsx", sheet_name=None)
        master = hvac_master.get("MASTER").set_index(keys="制御区分", drop=True)
        master.index = master.index.str.replace(r"\s+", "", regex=True)
        # 学習済みの機械学習モデルから情報を抽出
        elec_model_dir = "./models/{area}__kwh.joblib"
        temp_model_dir = "./models/{area}__temp.joblib"

        self.elec_model = load_residual_model(
            elec_model_dir.format(area=self.area_name)
        )
        self.temp_model = load_residual_model(
            temp_model_dir.format(area=self.area_name)
        )
        area_master = master.loc[self.area_name]
        indoor_mode_columns = CreaBuilding.pick_cols(self.elec_model.x_cols, "A/C Mode")
        self.room_list = [c.split("__")[-1] for c in indoor_mode_columns]
        self.unit_temp_range_list, self.target_temp_list = (
            extract_set_and_target_from_cols(area_master, self.room_list)
        )
        # train dataの読み込み
        self.train_data = pd.read_csv(self.train_data_path)
        self.train_data["Datetime_hour"] = pd.to_datetime(
            self.train_data["Datetime_hour"]
        )
        self.train_data = self.train_data.set_index("Datetime_hour", drop=True)
        # weather dataの読み込み (期間はtraindataと同じである必要があります。)
        self.weather_data = pd.read_csv(self.train_weather_path)
        self.weather_data["Datetime_hour"] = pd.to_datetime(
            self.weather_data["Datetime_hour"]
        )

        self.weather_data = self.weather_data.set_index("Datetime_hour", drop=True)
        use_col = CreaBuilding.get_columns_by_area_units(
            self.train_data.columns, self.area_name
        )
        self.train_data = self.train_data[use_col]
        # 学習済みのモデルに保存されている室内機ユニットの順番と同じ順番に揃える↓
        ordered_cols = []
        for room in self.room_list:
            suffix = f"__{room}"
            ordered_cols.extend(
                [c for c in self.train_data.columns if c.endswith(suffix)]
            )
        rest_cols = [c for c in self.train_data.columns if c not in ordered_cols]
        ordered_cols.extend(rest_cols)
        self.train_data = self.train_data.reindex(columns=ordered_cols)

    def load(self, start_term: pd.Timestamp, end_term: pd.Timestamp):
        cfg = AircontrolPPOTrainConfig()
        cfg.log_dir_root = cfg.log_dir_root.format(area=self.area_name)
        self.trainer = AircontrolPPOTrainer(
            cfg=cfg,
            base_df=self.train_data,
            weather_df=self.weather_data,
            start_term=start_term,
            end_term=end_term,
            temp_model_or_path=self.temp_model,
            elec_model_or_path=self.elec_model,
            unit_temp_range_list=self.unit_temp_range_list,
            target_temp_list=self.target_temp_list,
        )

    def train_run(self):
        self.trainer.run(area_name=self.area_name)


if __name__ == "__main__":
    start = pd.Timestamp("2025-09-10 07:00:00")
    end = pd.Timestamp("2025-09-11 07:00:00")
    ppo = Trainer("data/base/hourly_filled.csv", "data/base/hourly_filled.csv", "Area1")
    ppo.setup()
    ppo.load(start_term=start, end_term=end)
    ppo.train_run()
